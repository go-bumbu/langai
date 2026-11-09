package onnx

import (
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"sync"
)

// runtimeAPI is what loading ONNX Runtime takes; the library is loaded once
// per process whichever Embedder opens first.
type runtimeAPI interface {
	// load points the bindings at the shared library and initializes the
	// runtime environment.
	load(libPath string, logger *slog.Logger) error
	// version is the version string the loaded library reports.
	version() string
}

// sessionAPI is the slice of ONNX Runtime an Embedder drives; ortAPI
// (ort.go) is the real implementation and backend_test.go has a fake.
type sessionAPI interface {
	runtimeAPI
	// open creates a session over the model file that feeds opts.input and
	// reads opts.output, and reports the vector length the model file
	// declares for one picture, 0 when it leaves it dynamic. Unknown tensor
	// names are an error that lists the model's.
	open(path string, opts sessionOptions) (session, int, error)
}

// sessionOptions is how open sets a session up.
type sessionOptions struct {
	input, output string
	provider      Provider
	deviceID      int
	threads       int
}

// session is one loaded model. Embedder serializes access to it.
type session interface {
	// run feeds n pictures of side×side pixels (planar RGB float32,
	// n*3*side*side values) and returns the output tensor's values, as a
	// fresh slice, with its shape ([n, dims] or [n, tokens, dims]).
	run(pixels []float32, n, side int) (values []float32, shape []int64, err error)
	// close frees the session; called at most once.
	close()
}

// api is the ONNX Runtime surface Open drives; tests swap in a fake.
var api sessionAPI = ortAPI{}

// libs is the process-wide runtime loader Open goes through.
var libs = &runtimeLoader{}

// runtimeLoader loads the shared library once per process (the bindings
// keep it in package globals and never unload it) and remembers the outcome:
// a failure is permanent, which matches sense.LazyEmbedder's cached-error
// contract, and a second library is refused.
type runtimeLoader struct {
	once sync.Once
	// mu guards the outcome below for loadedDir, which an install asks while
	// another goroutine may be loading
	mu      sync.Mutex
	err     error
	path    string
	version string // what the loaded library reported, kept for Open's load log line
}

// load loads the library at libPath through rt on first call and checks it
// is the pinned release. Later calls succeed only for the same path.
func (r *runtimeLoader) load(rt runtimeAPI, libPath string, logger *slog.Logger) error {
	r.once.Do(func() {
		err := rt.load(libPath, logger)
		var v string
		if err == nil {
			if v = rt.version(); !strings.HasPrefix(v, libMajorMinor+".") && v != libMajorMinor {
				err = fmt.Errorf("onnxruntime at %s reports version %q but provider/onnx is pinned to %s: the Go bindings (onnxruntime_go %s) match that release's C API", libPath, v, LibVersion, pinnedBinding)
			}
		}
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.err = err; err == nil {
			r.path, r.version = libPath, v
		}
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return r.err
	}
	if r.path != libPath {
		return fmt.Errorf("onnxruntime is already loaded from %s; cannot load %s in the same process", r.path, libPath)
	}
	return nil
}

// loadedDir is the directory the library was loaded from, "" while it has
// not been (or failed to).
func (r *runtimeLoader) loadedDir() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.path == "" {
		return ""
	}
	return filepath.Dir(r.path)
}

package yzma

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
)

// llamaAPI is the slice of llama.cpp this package drives, with handles as
// plain integers so that only llama.go names the bindings' types. ffiAPI
// (llama.go) is the real implementation; backend_test.go has a fake.
type llamaAPI interface {
	// load loads the shared libraries from dir, routes llama.cpp's log lines
	// to logger and initializes the backends.
	load(dir string, logger *slog.Logger) error
	// version is the version string the loaded libraries report.
	version() string
	// loadModel loads a GGUF file; a zero handle is an error.
	loadModel(path string) (model uintptr, err error)
	freeModel(model uintptr)
	nCtxTrain(model uintptr) int32
	nEmbd(model uintptr) int32
	// newContext creates an embedding context whose context, batch and
	// micro-batch sizes are all nCtx, so one text of up to nCtx tokens fits a
	// single batch (required for encoder models), pooling token outputs the
	// given way. limit is the smallest of the sizes llama.cpp actually
	// granted; mem is 0 when the model has no KV cache (BERT encoders).
	newContext(model uintptr, nCtx int32, threads int, pooling Pooling) (ctx uintptr, limit int, mem uintptr, err error)
	freeContext(ctx uintptr)
	// tokenize converts text to tokens, [CLS]/[SEP] included, treating
	// special-token spellings inside text as ordinary text.
	tokenize(model uintptr, text string) []int32
	memoryClear(mem uintptr) error
	// decode runs one batch; ret is llama_decode's return value (0 = ok).
	decode(ctx uintptr, tokens []int32) (ret int32, err error)
	// embeddings returns the pooled vector of sequence 0 as a view into
	// llama.cpp memory that the next decode overwrites; nil when absent.
	embeddings(ctx uintptr, n int32) ([]float32, error)
}

// libs is the process-wide runtime loader Open goes through.
var libs = &runtimeLoader{}

// runtimeLoader loads the shared libraries once per process (yzma binds them
// into package globals and they are never unloaded) and remembers the
// outcome: a failure is permanent, which matches semsearch.LazyEmbedder's
// cached-error contract, and a second directory is refused.
type runtimeLoader struct {
	once    sync.Once
	err     error
	dir     string
	version string // what the loaded libraries reported, kept for Open's load log line
}

// load loads the libraries in libDir through api on first call and checks
// they are the pinned build. Later calls succeed only for the same directory.
func (r *runtimeLoader) load(api llamaAPI, libDir string, logger *slog.Logger) error {
	r.once.Do(func() {
		// On an assertion failure ggml forks gdb/lldb to print a backtrace
		// before aborting; in a multithreaded server that can hang instead
		// of crashing. Disable it unless the operator chose otherwise.
		if os.Getenv("GGML_NO_BACKTRACE") == "" {
			_ = os.Setenv("GGML_NO_BACKTRACE", "1")
		}
		if err := api.load(libDir, logger); err != nil {
			r.err = err
			return
		}
		v := api.version()
		if !strings.HasPrefix(v, libVersion) {
			r.err = fmt.Errorf("llama.cpp in %s reports version %q but semsearch/yzma is pinned to %s (build %s): the Go bindings only match that build", libDir, v, libVersion, LibTag)
			return
		}
		r.dir, r.version = libDir, v
	})
	if r.err != nil {
		return r.err
	}
	if r.dir != libDir {
		return fmt.Errorf("llama.cpp is already loaded from %s; cannot load %s in the same process", r.dir, libDir)
	}
	return nil
}

// llamaBackend is the backend over one loaded model and one llama.cpp
// context. Embedder serializes access to it.
type llamaBackend struct {
	api   llamaAPI
	model uintptr
	ctx   uintptr
	mem   uintptr // 0 for models without a KV cache
	embd  int32
	lim   int
}

// openLlama loads the libraries (once, through rt) and the model, and builds
// an embedding context sized to the model's training window capped at
// maxTokens (defaultMaxTokens when <= 0), so one text of up to that many
// tokens fits a single batch, which llama.cpp requires for non-causal
// (encoder) models. Whatever it created is freed again when a later step
// fails.
func openLlama(api llamaAPI, rt *runtimeLoader, libDir, modelPath string, threads int, pooling Pooling, maxTokens int, logger *slog.Logger) (*llamaBackend, error) {
	pooling, err := pooling.normalized()
	if err != nil {
		return nil, err
	}
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	if err := rt.load(api, libDir, logger); err != nil {
		return nil, err
	}
	model, err := api.loadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load model %s: %w", modelPath, err)
	}
	nCtxTrain := api.nCtxTrain(model)
	if nCtxTrain <= 0 {
		api.freeModel(model)
		return nil, fmt.Errorf("load model %s: no training context length in the GGUF metadata", modelPath)
	}
	// a model trained on 8192 tokens would otherwise get multi-gigabyte batch
	// buffers for inputs the callers keep far shorter
	nCtx := int32(min(int(nCtxTrain), maxTokens))
	ctx, granted, mem, err := api.newContext(model, nCtx, threads, pooling)
	if err != nil {
		api.freeModel(model)
		return nil, fmt.Errorf("create llama.cpp context: %w", err)
	}
	// truncateTokens reads a limit <= 0 as "no limit", which would let an
	// oversized batch through to llama.cpp's abort: refuse such a context
	lim := min(granted, int(nCtx))
	if lim <= 0 {
		api.freeContext(ctx)
		api.freeModel(model)
		return nil, fmt.Errorf("llama.cpp granted a %d-token batch for %s", lim, modelPath)
	}
	return &llamaBackend{
		api:   api,
		model: model,
		ctx:   ctx,
		mem:   mem,
		embd:  api.nEmbd(model),
		lim:   lim,
	}, nil
}

// close frees the context, then the model it was built on.
func (l *llamaBackend) close() {
	l.api.freeContext(l.ctx)
	l.api.freeModel(l.model)
}

func (l *llamaBackend) limit() int { return l.lim }

func (l *llamaBackend) tokenize(text string) []int32 { return l.api.tokenize(l.model, text) }

func (l *llamaBackend) embed(tokens []int32) ([]float32, error) {
	// llama.cpp asserts (and aborts the process) on a batch larger than
	// n_ubatch; Embedder truncates, this is the last line of defense.
	if len(tokens) == 0 || len(tokens) > l.lim {
		return nil, fmt.Errorf("batch of %d tokens is outside 1..%d", len(tokens), l.lim)
	}
	if l.mem != 0 {
		if err := l.api.memoryClear(l.mem); err != nil {
			return nil, fmt.Errorf("clear kv cache: %w", err)
		}
	}
	ret, err := l.api.decode(l.ctx, tokens)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if ret != 0 {
		return nil, fmt.Errorf("decode returned %d", ret)
	}
	view, err := l.api.embeddings(l.ctx, l.embd)
	if err != nil {
		return nil, fmt.Errorf("read embeddings: %w", err)
	}
	if view == nil {
		return nil, errors.New("model returned no vector")
	}
	// view aliases llama.cpp's buffer, which the next decode overwrites
	out := make([]float32, len(view))
	copy(out, view)
	return out, nil
}

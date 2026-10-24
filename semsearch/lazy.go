package semsearch

import (
	"context"
	"fmt"
	"sync"
)

// LazyEmbedder defers building the real embedder (with the yzma embedder:
// its model and library downloads too) to the first Embed call rather than
// server startup, unless the caller asks for it earlier through Load.
// Dimensions answers from the configured dims without loading, so an Index
// or a vector store (which needs a dimensionality to size its tables) can be
// built at startup before the model is ever touched. A load failure is
// cached: it is returned unchanged by every later Embed call instead of
// retried, so a broken or unreachable model degrades gracefully instead of
// re-attempting a download on every request; State exposes it to a health
// endpoint.
type LazyEmbedder struct {
	dims int
	// load builds the underlying embedder; see NewLazyEmbedder.
	load func() (Embedder, error)

	once      sync.Once
	mu        sync.Mutex // guards the fields below, written once inside once.Do
	inner     Embedder
	err       error
	attempted bool
}

// verify the interface is actually satisfied
var _ Embedder = (*LazyEmbedder)(nil)

// NewLazyEmbedder returns a LazyEmbedder that calls load on the first Embed
// call and keeps its result for the process lifetime. dims must match the
// loaded embedder's actual output dimensionality (yzma.Spec reports it before
// a model loads); a mismatch discovered at load time becomes a cached error
// rather than a panic.
func NewLazyEmbedder(load func() (Embedder, error), dims int) *LazyEmbedder {
	return &LazyEmbedder{dims: dims, load: load}
}

// Dimensions implements Embedder. It returns the configured dimensionality
// without triggering a load, so callers can size a vector store before the
// model is loaded.
func (l *LazyEmbedder) Dimensions() int { return l.dims }

// Load builds the embedder now instead of on the first Embed call, for a
// server that would rather pay for the downloads and the model load at
// startup than on the first request. It is the same once-guarded load Embed
// performs: whichever of the two runs first loads, the other waits for and
// reuses its result, and a failure is cached identically. Calling it again
// just returns the cached outcome.
func (l *LazyEmbedder) Load() error {
	l.ensureLoaded()
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.err
}

// Embed implements Embedder. It loads the model on the first call (unless
// Load already did) and reuses it, or its cached load error, on every later
// call; the role is passed through to the loaded embedder.
func (l *LazyEmbedder) Embed(ctx context.Context, role Role, texts []string) ([][]float32, error) {
	l.ensureLoaded()
	l.mu.Lock()
	inner, err := l.inner, l.err
	l.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return inner.Embed(ctx, role, texts)
}

// ensureLoaded runs the loader exactly once per process and records its
// outcome; every later call returns as soon as the first one has finished.
func (l *LazyEmbedder) ensureLoaded() {
	l.once.Do(func() {
		inner, err := l.load()
		if err == nil && inner.Dimensions() != l.dims {
			inner, err = nil, errDimMismatch(inner.Dimensions(), l.dims)
		}
		l.mu.Lock()
		l.inner, l.err, l.attempted = inner, err, true
		l.mu.Unlock()
	})
}

// State reports whether a load has been attempted yet and, if it failed, the
// cached error every Embed call now returns.
func (l *LazyEmbedder) State() (attempted bool, err error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.attempted, l.err
}

// errDimMismatch reports that a loaded embedder's actual output
// dimensionality does not match the dimensionality it was configured for.
func errDimMismatch(got, want int) error {
	return fmt.Errorf("semsearch: embedder produced %d-dimensional vectors, configured for %d", got, want)
}

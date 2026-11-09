package langai

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"reflect"
	"runtime/debug"
	"sync"
)

// LazyState is where a lazy model stands: LazyEmbedder, LazyImageEmbedder
// and LazyCompleter report it through State.
type LazyState int

const (
	// LazyNotStarted means that nothing has asked for the model yet.
	LazyNotStarted LazyState = iota
	// LazyLoading means that the load is running.
	LazyLoading
	// LazyLoaded means that the model is loaded and calls reach it.
	LazyLoaded
	// LazyFailed means that the load failed. The failure is kept: every call
	// returns it and nothing loads again.
	LazyFailed
	// LazyClosed means that Close was called. Every call fails.
	LazyClosed
)

// String implements fmt.Stringer for log messages and health endpoints.
func (s LazyState) String() string {
	switch s {
	case LazyNotStarted:
		return "not started"
	case LazyLoading:
		return "loading"
	case LazyLoaded:
		return "loaded"
	case LazyFailed:
		return "failed"
	case LazyClosed:
		return "closed"
	}
	return fmt.Sprintf("LazyState(%d)", int(s))
}

// LoadPanicError is the load error of a lazy model whose load function
// panicked: the panic is kept from taking the process down and fails the load
// instead.
type LoadPanicError struct {
	// Value is what the load function panicked with.
	Value any
	// Stack is the stack of the goroutine that panicked.
	Stack []byte
}

// Error implements error. The stack is left out; read it from Stack.
func (e *LoadPanicError) Error() string {
	return fmt.Sprintf("langai: the model's load function panicked: %v", e.Value)
}

// errLazyClosed is what the error of every call on a closed lazy model
// wraps.
var errLazyClosed = errors.New("closed")

// lazy loads a value of type T on first use and hands it to everyone who
// asks, for the three lazy models to wrap.
//
// The load runs once, on a goroutine of its own, under a context that
// belongs to the lazy value and not to any caller: only close cancels it.
// Callers wait for it under their own contexts, so one that gives up leaves
// the load running for the others. The outcome is final: a failure is kept
// and returned to every later caller, never retried, so that a model that
// cannot be loaded costs one attempt and not one per request.
type lazy[T any] struct {
	// what names the value in error messages, e.g. "embedder".
	what string
	load func(ctx context.Context) (T, error)
	// check, when set, has the last word on a loaded value: its error fails
	// the load.
	check func(T) error

	closing sync.Once
	// loading is the load goroutine.
	loading sync.WaitGroup
	// closeErr is what closing a value whose load ended after close
	// returned. The load goroutine writes it and close reads it once that
	// goroutine is done.
	closeErr error

	mu    sync.Mutex // guards the fields below
	state LazyState
	value T     // set in LazyLoaded
	err   error // what calls return in LazyFailed and LazyClosed
	// settled is closed when state leaves LazyLoading.
	settled chan struct{}
	// cancel ends the load's context; nil in LazyNotStarted.
	cancel context.CancelFunc
}

// get returns the loaded value, starting the load when nobody has and
// waiting for it while ctx lasts. ctx bounds the wait only.
func (l *lazy[T]) get(ctx context.Context) (T, error) {
	var zero T
	for {
		l.mu.Lock()
		if l.state == LazyNotStarted {
			if err := ctx.Err(); err != nil {
				l.mu.Unlock()
				return zero, fmt.Errorf("langai: lazy %s: %w", l.what, err)
			}
			l.start()
		}
		state, value, err, settled := l.state, l.value, l.err, l.settled
		l.mu.Unlock()
		if state != LazyLoading {
			return value, err
		}
		select {
		case <-settled: // look again: the state is final now
		case <-ctx.Done():
			return zero, fmt.Errorf("langai: waiting for the %s to load: %w", l.what, ctx.Err())
		}
	}
}

// start begins the load. The caller holds mu.
func (l *lazy[T]) start() {
	ctx, cancel := context.WithCancel(context.Background()) //nolint:gosec // cancel is kept in l.cancel: a failed load and close call it, and until then the loaded value may still work under ctx
	l.state, l.settled, l.cancel = LazyLoading, make(chan struct{}), cancel
	l.loading.Go(func() {
		value, err := l.run(ctx)
		if l.settle(value, err) {
			return
		}
		// close came first, so nobody will ever get the value
		if err == nil {
			l.closeErr = closeValue(value)
		}
	})
}

// run calls load and checks what it returns. A panic and a nil value are
// errors, so that no caller is ever handed a value it cannot call.
func (l *lazy[T]) run(ctx context.Context) (value T, err error) {
	var zero T
	defer func() {
		if r := recover(); r != nil {
			value, err = zero, &LoadPanicError{Value: r, Stack: debug.Stack()}
		}
	}()
	value, err = l.load(ctx)
	if err != nil {
		return zero, err
	}
	if isNil(value) {
		return zero, fmt.Errorf("langai: the %s's load function returned neither a value nor an error", l.what)
	}
	if l.check != nil {
		if err := l.check(value); err != nil {
			// the value is refused, so it is released here
			return zero, errors.Join(err, closeValue(value))
		}
	}
	return value, nil
}

// settle records the load's outcome and lets the waiting callers go. It
// reports false, and records nothing, when close got there first.
func (l *lazy[T]) settle(value T, err error) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.state != LazyLoading {
		return false
	}
	if err != nil {
		l.cancel() // nothing runs under the load's context any more
		l.state, l.err = LazyFailed, err
	} else {
		l.state, l.value = LazyLoaded, value
	}
	close(l.settled)
	return true
}

// status returns the state and the error calls fail with in it, nil while
// they can still succeed.
func (l *lazy[T]) status() (LazyState, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.state, l.err
}

// close ends the lazy value for good: it fails the callers waiting for a
// load and every later one, cancels a running load and waits for it to
// return, and closes the loaded value when it is an io.Closer, whose error
// it returns. It does not wait for calls already inside the loaded value:
// closing under them is that value's Close to make safe. Only the first call
// does anything; the others wait for it and return nil.
func (l *lazy[T]) close() error {
	var err error
	l.closing.Do(func() {
		var zero T
		l.mu.Lock()
		state, value := l.state, l.value
		l.state, l.value = LazyClosed, zero
		l.err = fmt.Errorf("langai: lazy %s: %w", l.what, errLazyClosed)
		if l.cancel != nil {
			l.cancel()
		}
		if state == LazyLoading {
			close(l.settled)
		}
		l.mu.Unlock()

		l.loading.Wait()
		if state == LazyLoaded {
			err = closeValue(value)
		} else {
			err = l.closeErr
		}
	})
	return err
}

// closeValue closes v when it is an io.Closer.
func closeValue(v any) error {
	if c, ok := v.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

// isNil reports whether v is nil or holds a nil pointer, map, slice, func or
// channel: a load function that returns a nil *T as its port hands over an
// interface that is not nil and cannot be called.
func isNil(v any) bool {
	if v == nil {
		return true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan:
		return rv.IsNil()
	}
	return false
}

// checkVectors is the check of the two lazy embedders: the loaded model must
// make vectors of the configured length and, when both it and the
// configuration name a model, be that model.
func checkVectors(loaded interface{ Dimensions() int }, dims int, model string) error {
	if got := loaded.Dimensions(); got != dims {
		return fmt.Errorf("langai: embedder produced %d-dimensional vectors, configured for %d", got, dims)
	}
	if id, ok := loaded.(Identified); ok && model != "" {
		if got := id.Model(); got != "" && got != model {
			return fmt.Errorf("langai: loaded the model %q, configured for %q: their vectors must not mix", got, model)
		}
	}
	return nil
}

// LazyEmbedder is an Embedder that defers building the real one (with the
// yzma embedder: its model and library downloads too) to the first Embed
// call rather than server startup, unless the caller asks for it earlier
// through Load. Model and Dimensions answer from the configuration without
// loading, so an Index or a vector store, which needs both to set up its
// collection, can be built at startup before the model is ever touched.
//
// The load runs once, under a context of the LazyEmbedder's own that only
// Close cancels: a caller whose context ends stops waiting, and the load
// goes on for the others. A load failure is kept: it is returned unchanged
// by every later call instead of retried, so a broken or unreachable model
// degrades gracefully instead of re-attempting a download on every request.
// State exposes all of it to a health endpoint. A LazyEmbedder is safe for
// concurrent use and must not be copied.
type LazyEmbedder struct {
	model string
	dims  int
	lazy  lazy[Embedder]
}

// verify the interfaces are actually satisfied
var (
	_ Embedder   = (*LazyEmbedder)(nil)
	_ Identified = (*LazyEmbedder)(nil)
	_ io.Closer  = (*LazyEmbedder)(nil)
)

// NewLazyEmbedder returns a LazyEmbedder that calls load on first use and
// keeps what it returns until Close. model is what Model reports, the name
// the loaded embedder gives itself ("" leaves the model unknown and
// unchecked), and dims what Dimensions reports (yzma.Spec has both before a
// model loads). A loaded embedder with another dimensionality, or one that
// names another model, fails the load and is closed.
//
// load gets a context that Close cancels and should stop for it. A load that
// panics or returns a nil Embedder fails like one that returns an error.
func NewLazyEmbedder(model string, dims int, load func(ctx context.Context) (Embedder, error)) *LazyEmbedder {
	return &LazyEmbedder{model: model, dims: dims, lazy: lazy[Embedder]{
		what:  "embedder",
		load:  load,
		check: func(e Embedder) error { return checkVectors(e, dims, model) },
	}}
}

// Model implements Identified. It returns the configured model name without
// triggering a load.
func (l *LazyEmbedder) Model() string { return l.model }

// Dimensions implements Embedder. It returns the configured dimensionality
// without triggering a load, so callers can size a vector store before the
// model is loaded.
func (l *LazyEmbedder) Dimensions() int { return l.dims }

// Embed implements Embedder. It loads the model on the first call (unless
// Load already did), waits for a load in progress while ctx lasts, and
// passes the call, role included, to the loaded embedder.
func (l *LazyEmbedder) Embed(ctx context.Context, role Role, texts []string) ([][]float32, error) {
	inner, err := l.lazy.get(ctx)
	if err != nil {
		return nil, err
	}
	return inner.Embed(ctx, role, texts)
}

// Load builds the embedder now instead of on the first Embed call, for a
// server that would rather pay for the downloads and the model load at
// startup than on the first request. It is the same load Embed performs:
// whichever of the two comes first starts it, the other waits for and reuses
// its result. ctx bounds the wait, not the load. Load returns nil once the
// model is loaded, and otherwise what Embed would fail with.
func (l *LazyEmbedder) Load(ctx context.Context) error {
	_, err := l.lazy.get(ctx)
	return err
}

// State reports where the embedder stands and, when its calls can no longer
// succeed, the error they return: the load's in LazyFailed, the closed one
// in LazyClosed.
func (l *LazyEmbedder) State() (LazyState, error) { return l.lazy.status() }

// Close releases the model: it cancels a load in progress and waits for it
// to return, and closes the loaded embedder when it is an io.Closer. Every
// call waiting for the load and every later one fails. Calls already inside
// the loaded embedder are its Close's to deal with. Closing again does
// nothing.
func (l *LazyEmbedder) Close() error { return l.lazy.close() }

// LazyImageEmbedder is an ImageEmbedder that defers building the real one to
// its first use. It behaves as LazyEmbedder does: one load under a context
// only Close cancels, a kept failure, and Model and Dimensions answered from
// the configuration without loading.
type LazyImageEmbedder struct {
	model string
	dims  int
	lazy  lazy[ImageEmbedder]
}

// verify the interfaces are actually satisfied
var (
	_ ImageEmbedder = (*LazyImageEmbedder)(nil)
	_ Identified    = (*LazyImageEmbedder)(nil)
	_ io.Closer     = (*LazyImageEmbedder)(nil)
)

// NewLazyImageEmbedder returns a LazyImageEmbedder that calls load on first
// use and keeps what it returns until Close. model, dims and load are
// NewLazyEmbedder's.
func NewLazyImageEmbedder(model string, dims int, load func(ctx context.Context) (ImageEmbedder, error)) *LazyImageEmbedder {
	return &LazyImageEmbedder{model: model, dims: dims, lazy: lazy[ImageEmbedder]{
		what:  "image embedder",
		load:  load,
		check: func(e ImageEmbedder) error { return checkVectors(e, dims, model) },
	}}
}

// Model implements Identified. It returns the configured model name without
// triggering a load.
func (l *LazyImageEmbedder) Model() string { return l.model }

// Dimensions implements ImageEmbedder. It returns the configured
// dimensionality without triggering a load.
func (l *LazyImageEmbedder) Dimensions() int { return l.dims }

// EmbedImages implements ImageEmbedder. It loads the model on the first call
// (unless Load already did), waits for a load in progress while ctx lasts,
// and passes the call to the loaded embedder.
func (l *LazyImageEmbedder) EmbedImages(ctx context.Context, images []image.Image) ([][]float32, error) {
	inner, err := l.lazy.get(ctx)
	if err != nil {
		return nil, err
	}
	return inner.EmbedImages(ctx, images)
}

// Load builds the embedder now instead of on the first EmbedImages call.
// ctx bounds the wait, not the load. See LazyEmbedder.Load.
func (l *LazyImageEmbedder) Load(ctx context.Context) error {
	_, err := l.lazy.get(ctx)
	return err
}

// State reports where the embedder stands and, when its calls can no longer
// succeed, the error they return. See LazyEmbedder.State.
func (l *LazyImageEmbedder) State() (LazyState, error) { return l.lazy.status() }

// Close releases the model and fails every waiting and later call. See
// LazyEmbedder.Close.
func (l *LazyImageEmbedder) Close() error { return l.lazy.close() }

// LazyCompleter is a Completer that defers building the real one, a chat
// model of a few gigabytes, to its first use. It behaves as LazyEmbedder
// does: one load under a context only Close cancels, and a kept failure.
type LazyCompleter struct {
	lazy lazy[Completer]
}

// verify the interfaces are actually satisfied
var (
	_ Completer = (*LazyCompleter)(nil)
	_ io.Closer = (*LazyCompleter)(nil)
)

// NewLazyCompleter returns a LazyCompleter that calls load on first use and
// keeps what it returns until Close. load gets a context that Close cancels
// and should stop for it; a load that panics or returns a nil Completer
// fails like one that returns an error.
func NewLazyCompleter(load func(ctx context.Context) (Completer, error)) *LazyCompleter {
	return &LazyCompleter{lazy: lazy[Completer]{what: "completer", load: load}}
}

// Complete implements Completer. It loads the model on the first call
// (unless Load already did), waits for a load in progress while ctx lasts,
// and passes the prompt to the loaded model.
func (l *LazyCompleter) Complete(ctx context.Context, system, user string) (string, error) {
	inner, err := l.lazy.get(ctx)
	if err != nil {
		return "", err
	}
	return inner.Complete(ctx, system, user)
}

// Load builds the model now instead of on the first Complete call. ctx
// bounds the wait, not the load. See LazyEmbedder.Load.
func (l *LazyCompleter) Load(ctx context.Context) error {
	_, err := l.lazy.get(ctx)
	return err
}

// State reports where the model stands and, when its calls can no longer
// succeed, the error they return. See LazyEmbedder.State.
func (l *LazyCompleter) State() (LazyState, error) { return l.lazy.status() }

// Close releases the model and fails every waiting and later call. See
// LazyEmbedder.Close.
func (l *LazyCompleter) Close() error { return l.lazy.close() }

package langai_test

import (
	"context"
	"errors"
	"image"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
)

// The loaded values of the tests: the fakes, with a Close that counts.

type closingEmbedder struct {
	langaitest.FakeEmbedder
	closes *atomic.Int32
}

func (e closingEmbedder) Close() error { e.closes.Add(1); return nil }

type closingImageEmbedder struct {
	langaitest.FakeImageEmbedder
	closes *atomic.Int32
}

func (e closingImageEmbedder) Close() error { e.closes.Add(1); return nil }

type closingCompleter struct {
	*langaitest.FakeCompleter
	closes *atomic.Int32
}

func (c closingCompleter) Close() error { c.closes.Add(1); return nil }

// subject is one of the three lazy models, reduced to what they share.
type subject struct {
	// call makes one call through the port: Embed, EmbedImages or Complete.
	call  func(ctx context.Context) error
	load  func(ctx context.Context) error
	state func() (langai.LazyState, error)
	close func() error
}

// loader is the load function of a subject under test: gate, when set, runs
// first and its error fails the load. It counts the loads and the times the
// loaded value was closed.
type loader struct {
	gate   func(ctx context.Context) error
	loads  atomic.Int32
	closes atomic.Int32
}

func (l *loader) run(ctx context.Context) error {
	l.loads.Add(1)
	if l.gate != nil {
		return l.gate(ctx)
	}
	return nil
}

// kinds builds each of the three lazy models around a loader, so that one
// test covers the behaviour they share.
var kinds = map[string]func(l *loader) subject{
	"LazyEmbedder": func(l *loader) subject {
		emb := langai.NewLazyEmbedder(langaitest.FakeEmbedderModel, 4, func(ctx context.Context) (langai.Embedder, error) {
			if err := l.run(ctx); err != nil {
				return nil, err
			}
			return closingEmbedder{langaitest.FakeEmbedder{Dims: 4}, &l.closes}, nil
		})
		return subject{load: emb.Load, state: emb.State, close: emb.Close, call: func(ctx context.Context) error {
			vecs, err := emb.Embed(ctx, langai.RoleDocument, []string{"x"})
			if err == nil && (len(vecs) != 1 || len(vecs[0]) != 4) {
				return errors.New("Embed did not return one 4-dimensional vector")
			}
			return err
		}}
	},
	"LazyImageEmbedder": func(l *loader) subject {
		dims := langaitest.FakeImageEmbedder{}.Dimensions()
		emb := langai.NewLazyImageEmbedder(langaitest.FakeImageEmbedderModel, dims, func(ctx context.Context) (langai.ImageEmbedder, error) {
			if err := l.run(ctx); err != nil {
				return nil, err
			}
			return closingImageEmbedder{langaitest.FakeImageEmbedder{}, &l.closes}, nil
		})
		return subject{load: emb.Load, state: emb.State, close: emb.Close, call: func(ctx context.Context) error {
			vecs, err := emb.EmbedImages(ctx, []image.Image{image.NewRGBA(image.Rect(0, 0, 8, 8))})
			if err == nil && (len(vecs) != 1 || len(vecs[0]) != dims) {
				return errors.New("EmbedImages did not return one vector")
			}
			return err
		}}
	},
	"LazyCompleter": func(l *loader) subject {
		chat := langai.NewLazyCompleter(func(ctx context.Context) (langai.Completer, error) {
			if err := l.run(ctx); err != nil {
				return nil, err
			}
			return closingCompleter{&langaitest.FakeCompleter{Replies: []string{"reply"}}, &l.closes}, nil
		})
		return subject{load: chat.Load, state: chat.State, close: chat.Close, call: func(ctx context.Context) error {
			reply, err := chat.Complete(ctx, "system", "user")
			if err == nil && reply != "reply" {
				return errors.New("Complete did not return the model's reply")
			}
			return err
		}}
	},
}

// eachKind runs test once per lazy model.
func eachKind(t *testing.T, test func(t *testing.T, l *loader, s subject)) {
	t.Helper()
	for name, build := range kinds {
		t.Run(name, func(t *testing.T) {
			l := &loader{}
			test(t, l, build(l))
		})
	}
}

// wantState fails the test unless the subject is in the given state.
func wantState(t *testing.T, s subject, want langai.LazyState) error {
	t.Helper()
	got, err := s.state()
	if got != want {
		t.Fatalf("State() = %v (%v), want %v", got, err, want)
	}
	return err
}

// hold makes the loader block until the returned release is called (or the
// load's context ends, which fails the load) and returns a channel that is
// closed once the load is in there.
func hold(l *loader) (started <-chan struct{}, release func()) {
	in, out := make(chan struct{}), make(chan struct{})
	l.gate = func(ctx context.Context) error {
		close(in)
		select {
		case <-out:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return in, sync.OnceFunc(func() { close(out) })
}

var errBoom = errors.New("boom")

// TestLazy_DoesNotLoadOnConstruction is the guard behind the "lazy" in the
// name: building one, and asking where it stands, must not call the loader,
// so a server can construct it at startup without paying for (or failing on)
// the downloads.
func TestLazy_DoesNotLoadOnConstruction(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		if err := wantState(t, s, langai.LazyNotStarted); err != nil {
			t.Errorf("State() reports %v before anything happened", err)
		}
		if n := l.loads.Load(); n != 0 {
			t.Errorf("the loader ran %d times at construction", n)
		}
	})
}

// TestLazy_ConcurrentFirstCallsLoadOnce: the first requests of a server
// arrive together, and so may a preload. They share one load.
func TestLazy_ConcurrentFirstCallsLoadOnce(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		started, release := hold(l)
		const callers = 16
		errs := make([]error, callers)
		var wg sync.WaitGroup
		for i := range callers {
			wg.Go(func() {
				if i%4 == 0 {
					errs[i] = s.load(context.Background())
					return
				}
				errs[i] = s.call(context.Background())
			})
		}
		<-started
		if err := wantState(t, s, langai.LazyLoading); err != nil {
			t.Errorf("State() reports %v while loading", err)
		}
		release()
		wg.Wait()
		for i, err := range errs {
			if err != nil {
				t.Errorf("caller %d: %v", i, err)
			}
		}
		if n := l.loads.Load(); n != 1 {
			t.Errorf("the loader ran %d times, want 1", n)
		}
		if err := wantState(t, s, langai.LazyLoaded); err != nil {
			t.Errorf("State() reports %v once loaded", err)
		}
		// later calls and a late Load reuse the model
		if err := errors.Join(s.call(context.Background()), s.load(context.Background())); err != nil || l.loads.Load() != 1 {
			t.Errorf("after the load: %v, %d loads", err, l.loads.Load())
		}
	})
}

// TestLazy_CallerDeadlineDoesNotCancelTheLoad: a request that gives up
// during a slow load returns with its own context's error, and the load
// goes on for the request that is still waiting.
func TestLazy_CallerDeadlineDoesNotCancelTheLoad(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		started, release := hold(l)
		defer release()

		patient := make(chan error, 1)
		go func() { patient <- s.call(context.Background()) }()
		<-started

		short, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		if err := s.call(short); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("the impatient caller got %v, want its deadline", err)
		}
		if err := s.load(short); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("the impatient Load got %v, want its deadline", err)
		}
		if err := wantState(t, s, langai.LazyLoading); err != nil {
			t.Errorf("State() reports %v: the caller's deadline reached the load", err)
		}

		release()
		if err := <-patient; err != nil {
			t.Errorf("the patient caller got %v, want the result", err)
		}
		if n := l.loads.Load(); n != 1 {
			t.Errorf("the loader ran %d times, want 1", n)
		}
	})
}

// TestLazy_DoneContextDoesNotStartALoad: a request that is already over
// must not set off a multi-gigabyte load nobody is waiting for.
func TestLazy_DoneContextDoesNotStartALoad(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := s.call(ctx); !errors.Is(err, context.Canceled) {
			t.Errorf("call = %v, want context.Canceled", err)
		}
		_ = wantState(t, s, langai.LazyNotStarted)
		if n := l.loads.Load(); n != 0 {
			t.Errorf("the loader ran %d times", n)
		}
	})
}

// TestLazy_FailedLoadIsKept: a load failure is final. Every later call and
// Load returns it without loading again, and State shows it to a health
// endpoint.
func TestLazy_FailedLoadIsKept(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		l.gate = func(context.Context) error { return errBoom }
		for i := range 3 {
			if err := s.call(context.Background()); !errors.Is(err, errBoom) {
				t.Fatalf("call %d = %v, want the load's error", i, err)
			}
		}
		if err := s.load(context.Background()); !errors.Is(err, errBoom) {
			t.Errorf("Load = %v, want the load's error", err)
		}
		if n := l.loads.Load(); n != 1 {
			t.Errorf("the loader ran %d times, want 1: a failure is not retried", n)
		}
		if err := wantState(t, s, langai.LazyFailed); !errors.Is(err, errBoom) {
			t.Errorf("State() reports %v, want the load's error", err)
		}
		if err := s.close(); err != nil {
			t.Errorf("Close after a failed load: %v", err)
		}
		if n := l.closes.Load(); n != 0 {
			t.Errorf("closed %d values, but none was loaded", n)
		}
	})
}

// TestLazy_PanickingLoadIsAnError: a panic in the loader fails the load like
// an error would, with the stack kept, instead of killing the process or
// leaving a nil model behind for the next call to trip over.
func TestLazy_PanickingLoadIsAnError(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		l.gate = func(context.Context) error { panic("the model file is corrupt") }
		for i := range 2 {
			err := s.call(context.Background())
			var panicked *langai.LoadPanicError
			if !errors.As(err, &panicked) {
				t.Fatalf("call %d = %v, want a *LoadPanicError", i, err)
			}
			if panicked.Value != "the model file is corrupt" || !strings.Contains(string(panicked.Stack), "lazy_test.go") {
				t.Errorf("call %d: panic value %v, stack of %d bytes without the loader in it", i, panicked.Value, len(panicked.Stack))
			}
			if !strings.Contains(err.Error(), "the model file is corrupt") || strings.Contains(err.Error(), "goroutine") {
				t.Errorf("call %d: message %q, want the panic value and no stack", i, err)
			}
		}
		if err := wantState(t, s, langai.LazyFailed); err == nil {
			t.Error("State() reports no error after a panic")
		}
		if n := l.loads.Load(); n != 1 {
			t.Errorf("the loader ran %d times, want 1", n)
		}
	})
}

// TestLazy_CloseDuringLoad: Close cancels the load's context, releases the
// callers waiting for it and returns only once the load has.
func TestLazy_CloseDuringLoad(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		var loadCtxErr atomic.Value
		entered, returned := make(chan struct{}), make(chan struct{})
		l.gate = func(ctx context.Context) error {
			close(entered)
			<-ctx.Done() // only Close ends this load
			loadCtxErr.Store(ctx.Err())
			close(returned)
			return ctx.Err()
		}
		waiting := make(chan error, 1)
		go func() { waiting <- s.call(context.Background()) }()
		<-entered

		if err := s.close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		select {
		case <-returned:
		default:
			t.Fatal("Close returned while the load was still running")
		}
		if err, _ := loadCtxErr.Load().(error); !errors.Is(err, context.Canceled) {
			t.Errorf("the load's context ended with %v, want cancellation", err)
		}
		if err := <-waiting; !errors.Is(err, langai.ErrClosed) {
			t.Errorf("the waiting caller got %v, want the closed error", err)
		}
		if err := wantState(t, s, langai.LazyClosed); !errors.Is(err, langai.ErrClosed) {
			t.Errorf("State() reports %v, want the closed error", err)
		}
	})
}

// TestLazy_CloseDuringLoadClosesWhatTheLoadStillReturns: a loader that does
// not look at its context finishes with a model nobody will ever use. Close
// lets its waiting callers go at once, waits for the load, and releases that
// model.
func TestLazy_CloseDuringLoadClosesWhatTheLoadStillReturns(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		entered, release := make(chan struct{}), make(chan struct{})
		l.gate = func(context.Context) error {
			close(entered)
			<-release
			return nil
		}
		waiting := make(chan error, 1)
		go func() { waiting <- s.call(context.Background()) }()
		<-entered

		closed := make(chan error, 1)
		go func() { closed <- s.close() }()
		if err := <-waiting; !errors.Is(err, langai.ErrClosed) {
			t.Errorf("the waiting caller got %v while the load was still running, want the closed error", err)
		}
		select {
		case err := <-closed:
			t.Fatalf("Close returned %v while the load was still running", err)
		case <-time.After(20 * time.Millisecond):
		}
		close(release)
		if err := <-closed; err != nil {
			t.Fatalf("Close: %v", err)
		}
		if n := l.closes.Load(); n != 1 {
			t.Errorf("the model the load returned was closed %d times, want 1", n)
		}
		_ = wantState(t, s, langai.LazyClosed)
	})
}

// TestLazy_CloseAfterLoad: Close closes the loaded model once, however often
// and from however many goroutines it is called, and every later call fails
// with the closed error without loading again.
func TestLazy_CloseAfterLoad(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		if err := s.call(context.Background()); err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		for range 8 {
			wg.Go(func() {
				if err := s.close(); err != nil {
					t.Errorf("Close: %v", err)
				}
			})
		}
		wg.Wait()
		if n := l.closes.Load(); n != 1 {
			t.Errorf("the model was closed %d times, want 1", n)
		}
		if err := s.call(context.Background()); !errors.Is(err, langai.ErrClosed) {
			t.Errorf("call after Close = %v, want the closed error", err)
		}
		if err := s.load(context.Background()); !errors.Is(err, langai.ErrClosed) {
			t.Errorf("Load after Close = %v, want the closed error", err)
		}
		if err := wantState(t, s, langai.LazyClosed); !errors.Is(err, langai.ErrClosed) {
			t.Errorf("State() reports %v, want the closed error", err)
		}
		if n := l.loads.Load(); n != 1 {
			t.Errorf("the loader ran %d times, want 1", n)
		}
	})
}

// TestLazy_CloseBeforeAnyLoad: closing a model that was never needed loads
// nothing, then or later.
func TestLazy_CloseBeforeAnyLoad(t *testing.T) {
	eachKind(t, func(t *testing.T, l *loader, s subject) {
		if err := s.close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		if err := s.call(context.Background()); !errors.Is(err, langai.ErrClosed) {
			t.Errorf("call after Close = %v, want the closed error", err)
		}
		_ = wantState(t, s, langai.LazyClosed)
		if n := l.loads.Load(); n != 0 {
			t.Errorf("the loader ran %d times", n)
		}
	})
}

// failingCloser is a model whose Close fails.
type failingCloser struct{ langaitest.FakeEmbedder }

func (failingCloser) Close() error { return errBoom }

// TestLazyEmbedder_CloseReturnsTheModelsError: the first Close reports what
// closing the model returned; the ones after it have nothing left to report.
func TestLazyEmbedder_CloseReturnsTheModelsError(t *testing.T) {
	emb := langai.NewLazyEmbedder("", 4, func(context.Context) (langai.Embedder, error) {
		return failingCloser{langaitest.FakeEmbedder{Dims: 4}}, nil
	})
	if err := emb.Load(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := emb.Close(); !errors.Is(err, errBoom) {
		t.Errorf("Close = %v, want the model's error", err)
	}
	if err := emb.Close(); err != nil {
		t.Errorf("second Close = %v, want nil", err)
	}
}

// TestLazy_NilModelIsAnError: a loader that returns no model and no error,
// or a nil pointer behind the interface, fails the load. Without the check
// the first call would dereference nil.
func TestLazy_NilModelIsAnError(t *testing.T) {
	var noEmbedder *closingEmbedder
	var noImageEmbedder *closingImageEmbedder
	var noCompleter *langaitest.FakeCompleter
	ctx := context.Background()
	cases := map[string]subject{}
	add := func(name string, load func(ctx context.Context) error, state func() (langai.LazyState, error)) {
		cases[name] = subject{load: load, state: state}
	}
	e1 := langai.NewLazyEmbedder("", 4, func(context.Context) (langai.Embedder, error) { return nil, nil })
	add("embedder, nil", e1.Load, e1.State)
	e2 := langai.NewLazyEmbedder("", 4, func(context.Context) (langai.Embedder, error) { return noEmbedder, nil })
	add("embedder, nil pointer", e2.Load, e2.State)
	i1 := langai.NewLazyImageEmbedder("", 4, func(context.Context) (langai.ImageEmbedder, error) { return nil, nil })
	add("image embedder, nil", i1.Load, i1.State)
	i2 := langai.NewLazyImageEmbedder("", 4, func(context.Context) (langai.ImageEmbedder, error) { return noImageEmbedder, nil })
	add("image embedder, nil pointer", i2.Load, i2.State)
	c1 := langai.NewLazyCompleter(func(context.Context) (langai.Completer, error) { return nil, nil })
	add("completer, nil", c1.Load, c1.State)
	c2 := langai.NewLazyCompleter(func(context.Context) (langai.Completer, error) { return noCompleter, nil })
	add("completer, nil pointer", c2.Load, c2.State)
	c3 := langai.NewLazyCompleter(nil)
	add("completer, no load function", c3.Load, c3.State)

	for name, s := range cases {
		t.Run(name, func(t *testing.T) {
			first, second := s.load(ctx), s.load(ctx)
			if first == nil || second == nil {
				t.Fatalf("Load = %v, then %v; want an error both times", first, second)
			}
			if err := wantState(t, s, langai.LazyFailed); err == nil {
				t.Error("State() reports no error")
			}
		})
	}
	if _, err := e2.Embed(ctx, langai.RoleQuery, []string{"x"}); err == nil {
		t.Error("Embed on a nil model: want the load's error")
	}
	if _, err := i2.EmbedImages(ctx, []image.Image{image.NewRGBA(image.Rect(0, 0, 1, 1))}); err == nil {
		t.Error("EmbedImages on a nil model: want the load's error")
	}
	if _, err := c2.Complete(ctx, "s", "u"); err == nil {
		t.Error("Complete on a nil model: want the load's error")
	}
}

// named is a FakeEmbedder that reports another model name.
type named struct {
	closingEmbedder
	model string
}

func (n named) Model() string { return n.model }

// anonymous is an embedder that names no model at all.
type anonymous struct{ inner langaitest.FakeEmbedder }

func (a anonymous) Dimensions() int { return a.inner.Dimensions() }

func (a anonymous) Embed(ctx context.Context, role langai.Role, texts []string) ([][]float32, error) {
	return a.inner.Embed(ctx, role, texts)
}

// TestLazyEmbedder_ModelAndDimensions: Model and Dimensions answer from the
// configuration before any load, because a store records both when an index
// is built. A loaded model that contradicts either fails the load, for good,
// and is closed: its vectors must never reach a collection built for the
// configured model.
func TestLazyEmbedder_ModelAndDimensions(t *testing.T) {
	cases := []struct {
		name        string
		model       string // configured
		dims        int    // configured
		loadedModel string // "-" for a model that is not Identified
		loadedDims  int
		wantErr     string
	}{
		{name: "the same model", model: "m1", dims: 4, loadedModel: "m1", loadedDims: 4},
		{name: "another model", model: "m1", dims: 4, loadedModel: "m2", loadedDims: 4, wantErr: `loaded the model "m2", configured for "m1"`},
		{name: "other dimensions", model: "m1", dims: 4, loadedModel: "m1", loadedDims: 8, wantErr: "8-dimensional vectors, configured for 4"},
		{name: "no model configured", model: "", dims: 4, loadedModel: "m2", loadedDims: 4},
		{name: "the loaded model names none", model: "m1", dims: 4, loadedModel: "", loadedDims: 4},
		{name: "the loaded model is not Identified", model: "m1", dims: 4, loadedModel: "-", loadedDims: 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var loads, closes atomic.Int32
			emb := langai.NewLazyEmbedder(tc.model, tc.dims, func(context.Context) (langai.Embedder, error) {
				loads.Add(1)
				fake := langaitest.FakeEmbedder{Dims: tc.loadedDims}
				if tc.loadedModel == "-" {
					return anonymous{fake}, nil
				}
				return named{closingEmbedder{fake, &closes}, tc.loadedModel}, nil
			})
			if emb.Model() != tc.model || emb.Dimensions() != tc.dims || loads.Load() != 0 {
				t.Fatalf("before the load: Model() = %q, Dimensions() = %d, %d loads; want the configuration and no load",
					emb.Model(), emb.Dimensions(), loads.Load())
			}
			ctx := context.Background()
			_, first := emb.Embed(ctx, langai.RoleDocument, []string{"x"})
			_, second := emb.Embed(ctx, langai.RoleDocument, []string{"y"})
			if tc.wantErr == "" {
				if first != nil || second != nil {
					t.Fatalf("Embed = %v, then %v", first, second)
				}
			} else {
				if first == nil || second == nil || !strings.Contains(first.Error(), tc.wantErr) || second.Error() != first.Error() {
					t.Fatalf("Embed = %v, then %v; want an error saying %q both times", first, second, tc.wantErr)
				}
				if state, _ := emb.State(); state != langai.LazyFailed {
					t.Errorf("State() = %v, want failed", state)
				}
				if closes.Load() != 1 {
					t.Errorf("the refused model was closed %d times, want 1", closes.Load())
				}
			}
			if loads.Load() != 1 {
				t.Errorf("the loader ran %d times, want 1", loads.Load())
			}
			// what the store was told at construction still holds
			if emb.Model() != tc.model || emb.Dimensions() != tc.dims {
				t.Errorf("after the load: Model() = %q, Dimensions() = %d", emb.Model(), emb.Dimensions())
			}
		})
	}
}

// TestLazyImageEmbedder_ModelAndDimensions: the image embedder checks the
// loaded model the same way.
func TestLazyImageEmbedder_ModelAndDimensions(t *testing.T) {
	dims := langaitest.FakeImageEmbedder{}.Dimensions()
	load := func(context.Context) (langai.ImageEmbedder, error) { return langaitest.FakeImageEmbedder{}, nil }
	cases := []struct {
		name    string
		model   string
		dims    int
		wantErr string
	}{
		{name: "the same model", model: langaitest.FakeImageEmbedderModel, dims: dims},
		{name: "no model configured", model: "", dims: dims},
		{name: "another model", model: "siglip", dims: dims, wantErr: `configured for "siglip"`},
		{name: "other dimensions", model: langaitest.FakeImageEmbedderModel, dims: 768, wantErr: "configured for 768"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			emb := langai.NewLazyImageEmbedder(tc.model, tc.dims, load)
			if emb.Model() != tc.model || emb.Dimensions() != tc.dims {
				t.Fatalf("Model() = %q, Dimensions() = %d; want the configuration", emb.Model(), emb.Dimensions())
			}
			if state, _ := emb.State(); state != langai.LazyNotStarted {
				t.Fatalf("Model and Dimensions loaded the model: %v", state)
			}
			err := emb.Load(context.Background())
			if (tc.wantErr == "") != (err == nil) || (err != nil && !strings.Contains(err.Error(), tc.wantErr)) {
				t.Errorf("Load = %v, want an error saying %q", err, tc.wantErr)
			}
		})
	}
}

// roleRecorder wraps an Embedder and records each call's role.
type roleRecorder struct {
	inner langai.Embedder
	roles []langai.Role
}

func (r *roleRecorder) Dimensions() int { return r.inner.Dimensions() }

func (r *roleRecorder) Embed(ctx context.Context, role langai.Role, texts []string) ([][]float32, error) {
	r.roles = append(r.roles, role)
	return r.inner.Embed(ctx, role, texts)
}

// TestLazyEmbedder_ForwardsRole: Embed hands the caller's role to the loaded
// embedder untouched. An asymmetric model prefixes queries and documents
// differently, so a LazyEmbedder that fixed the role (or dropped it) would
// silently skew every search made through it.
func TestLazyEmbedder_ForwardsRole(t *testing.T) {
	rec := &roleRecorder{inner: langaitest.FakeEmbedder{Dims: 4}}
	l := langai.NewLazyEmbedder("", 4, func(context.Context) (langai.Embedder, error) { return rec, nil })
	ctx := context.Background()
	if _, err := l.Embed(ctx, langai.RoleQuery, []string{"x"}); err != nil {
		t.Fatalf("Embed as query: %v", err)
	}
	if _, err := l.Embed(ctx, langai.RoleDocument, []string{"x"}); err != nil {
		t.Fatalf("Embed as document: %v", err)
	}
	want := []langai.Role{langai.RoleQuery, langai.RoleDocument}
	if !slices.Equal(rec.roles, want) {
		t.Fatalf("the loaded embedder saw roles %v, want %v", rec.roles, want)
	}
}

// TestLazyCompleter_ForwardsThePrompt: the loaded model gets the system and
// user messages as they came.
func TestLazyCompleter_ForwardsThePrompt(t *testing.T) {
	fake := &langaitest.FakeCompleter{Replies: []string{"bread"}}
	chat := langai.NewLazyCompleter(func(context.Context) (langai.Completer, error) { return fake, nil })
	reply, err := chat.Complete(context.Background(), "write tags", "sourdough")
	if err != nil || reply != "bread" {
		t.Fatalf("Complete = %q, %v", reply, err)
	}
	if want := []langaitest.CompleterCall{{System: "write tags", User: "sourdough"}}; !slices.Equal(fake.Calls(), want) {
		t.Errorf("the loaded model saw %+v, want %+v", fake.Calls(), want)
	}
}

func TestLazyState_String(t *testing.T) {
	want := map[langai.LazyState]string{
		langai.LazyNotStarted: "not started",
		langai.LazyLoading:    "loading",
		langai.LazyLoaded:     "loaded",
		langai.LazyFailed:     "failed",
		langai.LazyClosed:     "closed",
		langai.LazyState(42):  "LazyState(42)",
	}
	for state, text := range want {
		if state.String() != text {
			t.Errorf("LazyState(%d).String() = %q, want %q", int(state), state.String(), text)
		}
	}
}

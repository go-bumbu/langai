package tagging

import (
	"context"
	"sync"
)

// flight runs one function per key at a time, for every caller of that key:
// the engine's singleflight, so many Suggests on one stale space rebuild it
// once. The zero value is ready to use.
type flight struct {
	mu    sync.Mutex
	calls map[string]*flightCall
}

// flightCall is one run of a function; err is set before done is closed.
type flightCall struct {
	done chan struct{}
	err  error
}

// do runs fn unless a run for key is in flight, and returns the error of the
// run it waits for. fn runs on its own goroutine with
// context.WithoutCancel(ctx) of the caller that started it, so a caller that
// gives up does not fail the others: each caller returns early with its own
// ctx.Err() when its ctx ends, and the run carries on for the rest. A caller
// that arrives once a run has ended starts a new one, so a failure is retried.
func (f *flight) do(ctx context.Context, key string, fn func(context.Context) error) error {
	f.mu.Lock()
	c, ok := f.calls[key]
	if !ok {
		if f.calls == nil {
			f.calls = map[string]*flightCall{}
		}
		c = &flightCall{done: make(chan struct{})}
		f.calls[key] = c
		go f.run(context.WithoutCancel(ctx), key, c, fn)
	}
	f.mu.Unlock()
	select {
	case <-c.done:
		return c.err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// run runs fn for c and then ends the flight of key.
func (f *flight) run(ctx context.Context, key string, c *flightCall, fn func(context.Context) error) {
	defer func() {
		f.mu.Lock()
		delete(f.calls, key)
		f.mu.Unlock()
		close(c.done)
	}()
	c.err = fn(ctx)
}

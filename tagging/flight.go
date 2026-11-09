package tagging

import (
	"context"
	"fmt"
	"runtime/debug"
	"sync"

	"github.com/go-bumbu/langai"
)

// errClosed is returned by every call on a Tagger after its Close, and
// wrapped by the error of a call that Close cut short. It wraps
// langai.ErrClosed, the module's one sentinel for a closed value.
var errClosed = fmt.Errorf("tagging: tagger: %w", langai.ErrClosed)

// flight runs one function per key at a time, for every caller of that key:
// the engine's singleflight, keyed by space, so the rebuilds of one space
// never overlap and many Suggests on one stale space rebuild it once. The
// zero value is ready to use.
type flight struct {
	mu      sync.Mutex
	calls   map[spaceKey]*flightCall
	closed  bool
	running sync.WaitGroup
	// onPanic, when set, is told of a panic in a run, with its stack, once
	onPanic func(value any, stack []byte)
}

// flightCall is one run of a function; err is set before done is closed. The
// other fields are guarded by the flight's mutex.
type flightCall struct {
	tag     string // what the run builds: callers that want another look again
	done    chan struct{}
	err     error
	cancel  context.CancelFunc
	waiters int
	// left is set once the last waiter has gone and the run was cancelled:
	// nobody joins it any more, but it still holds its key until it ends
	left bool
}

// panicError is the error of a run that panicked.
type panicError struct{ value any }

func (p *panicError) Error() string { return fmt.Sprintf("panic: %v", p.value) }

// do runs fn unless a run for key is in flight, and waits for that run. With
// again false, err is the outcome for the caller: the error of a run for tag
// (its own, or one another caller started for the same tag), its own
// ctx.Err() when its ctx ended first, or errClosed. With again true, the run
// it waited for was for another tag and has ended, without an outcome for
// this caller: it looks at what that run left and calls do again if it still
// needs to. So the runs of one key follow each other, whatever they build.
//
// fn runs on its own goroutine, with a context that keeps the values of the
// ctx of the caller that started it but not its cancellation, so a caller that
// gives up does not fail the others: it returns early and the run carries on
// for the rest. When the last of them has gone, the run's context is
// cancelled: nobody waits for what it builds. Callers that come while it winds
// down wait for it to end, and then start a run of their own. A panic in fn
// does not take the process down: it is the run's error, with the value
// panicked with, and is reported to onPanic with its stack. A caller that
// arrives once a run has ended starts a new one, so a failure is retried.
func (f *flight) do(ctx context.Context, key spaceKey, tag string, fn func(context.Context) error) (again bool, err error) {
	for {
		f.mu.Lock()
		if f.closed {
			f.mu.Unlock()
			return false, errClosed
		}
		c, ok := f.calls[key]
		if ok && c.left {
			f.mu.Unlock()
			select {
			case <-c.done:
				continue
			case <-ctx.Done():
				return false, ctx.Err()
			}
		}
		if !ok {
			c = f.start(ctx, key, tag, fn)
		}
		c.waiters++
		f.mu.Unlock()
		select {
		case <-c.done:
			if c.tag != tag {
				return true, nil
			}
			return false, c.err
		case <-ctx.Done():
			f.leave(c)
			return false, ctx.Err()
		}
	}
}

// start begins a run of fn for key. The caller holds the mutex.
func (f *flight) start(ctx context.Context, key spaceKey, tag string, fn func(context.Context) error) *flightCall {
	if f.calls == nil {
		f.calls = map[spaceKey]*flightCall{}
	}
	runCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	c := &flightCall{tag: tag, done: make(chan struct{}), cancel: cancel}
	f.calls[key] = c
	f.running.Add(1)
	go f.run(runCtx, key, c, fn)
	return c
}

// leave takes a caller whose ctx ended off c, and cancels the run when it was
// the last one waiting.
func (f *flight) leave(c *flightCall) {
	f.mu.Lock()
	defer f.mu.Unlock()
	c.waiters--
	if c.waiters == 0 {
		c.left = true
		c.cancel()
	}
}

// run runs fn for c, turning a panic into the run's error, and then ends the
// flight of key.
func (f *flight) run(ctx context.Context, key spaceKey, c *flightCall, fn func(context.Context) error) {
	defer f.running.Done()
	defer func() {
		if v := recover(); v != nil {
			c.err = &panicError{value: v}
			if f.onPanic != nil {
				f.onPanic(v, debug.Stack())
			}
		}
		c.cancel()
		f.mu.Lock()
		delete(f.calls, key)
		f.mu.Unlock()
		close(c.done)
	}()
	c.err = fn(ctx)
}

// close stops the flight: no run starts after it, the ones in flight are
// cancelled, and it returns when they have ended. A run whose function does
// not return on a cancelled context holds it up for as long.
func (f *flight) close() {
	f.mu.Lock()
	f.closed = true
	for _, c := range f.calls {
		c.cancel()
	}
	f.mu.Unlock()
	f.running.Wait()
}

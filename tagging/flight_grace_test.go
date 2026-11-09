package tagging

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/go-bumbu/langai"
)

// within is run under a ctx that ends after d.
func within(d time.Duration, f *flight, fn func(context.Context) error) error {
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	return run(ctx, f, fn)
}

// untilCancelled is a run that ends only when its ctx does, and says so.
func untilCancelled(ended *atomic.Bool) func(context.Context) error {
	return func(ctx context.Context) error {
		<-ctx.Done()
		ended.Store(true)
		return ctx.Err()
	}
}

// TestFlight_AnAbandonedRunSurvivesAGracePeriod: a run that takes longer than
// each of its callers waits used to be cancelled as the last of them left, and
// started over by the next, for ever. With a grace period it goes on for a
// while with nobody waiting: a caller that comes within it joins the same run
// and gets its result, and only a run nobody came back for is cancelled.
func TestFlight_AnAbandonedRunSurvivesAGracePeriod(t *testing.T) {
	t.Run("the next caller joins the run and gets its result", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			f := flight{grace: 30 * time.Second}
			var runs atomic.Int32
			slow := func(ctx context.Context) error {
				runs.Add(1)
				select {
				case <-time.After(60 * time.Millisecond):
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			if err := within(40*time.Millisecond, &f, slow); !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("the first caller: err = %v, want its deadline", err)
			}
			if err := within(40*time.Millisecond, &f, slow); err != nil {
				t.Errorf("the second caller: err = %v, want the result of the run the first one left", err)
			}
			if n := runs.Load(); n != 1 {
				t.Errorf("fn ran %d times, want once", n)
			}
		})
	})

	t.Run("a run nobody comes back for is cancelled when the period ends", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			f := flight{grace: 30 * time.Second}
			var ended atomic.Bool
			if err := within(time.Second, &f, untilCancelled(&ended)); !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("err = %v, want the caller's deadline", err)
			}
			time.Sleep(30*time.Second - time.Nanosecond)
			synctest.Wait()
			if ended.Load() {
				t.Error("the run was cancelled before the grace period ended")
			}
			time.Sleep(time.Nanosecond)
			synctest.Wait()
			if !ended.Load() {
				t.Error("the run goes on after the grace period, with nobody waiting for it")
			}
			// and the key is free again
			if err := run(context.Background(), &f, func(context.Context) error { return nil }); err != nil {
				t.Errorf("the run after it: %v", err)
			}
		})
	})

	t.Run("a caller that joins stops the clock, and leaving starts it again", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			f := flight{grace: 30 * time.Second}
			var ended atomic.Bool
			fn := untilCancelled(&ended)
			_ = within(time.Second, &f, fn) // left at 1s: cancelled at 31s unless somebody joins
			time.Sleep(19 * time.Second)
			_ = within(40*time.Second, &f, fn) // joins at 20s, leaves at 60s
			synctest.Wait()
			if ended.Load() {
				t.Fatal("the run was cancelled while a caller that joined it waited: the first period's clock was not stopped")
			}
			time.Sleep(30*time.Second - time.Nanosecond)
			synctest.Wait()
			if ended.Load() {
				t.Error("the run was cancelled before the second grace period ended")
			}
			time.Sleep(time.Nanosecond)
			synctest.Wait()
			if !ended.Load() {
				t.Error("the run goes on after the second grace period")
			}
		})
	})

	t.Run("without a grace period the run is cancelled as its last caller leaves", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			var f flight
			var ended atomic.Bool
			_ = within(time.Second, &f, untilCancelled(&ended))
			synctest.Wait()
			if !ended.Load() {
				t.Error("the run goes on with nobody waiting for it")
			}
		})
	})
}

// TestFlight_CloseDuringTheGracePeriodCancelsAtOnce: close does not wait for
// the grace period of a run nobody waits for.
func TestFlight_CloseDuringTheGracePeriodCancelsAtOnce(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		f := flight{grace: 30 * time.Second}
		var ended atomic.Bool
		_ = within(time.Second, &f, untilCancelled(&ended))
		f.mu.Lock()
		c := f.calls[flightKey]
		f.mu.Unlock()
		if c == nil || c.timer == nil {
			t.Fatal("no run in its grace period to close")
		}
		start := time.Now()
		f.close()
		if !ended.Load() || time.Since(start) != 0 {
			t.Errorf("close returned after %v with the run ended: %v; want at once, and ended", time.Since(start), ended.Load())
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		if c.timer != nil || len(f.calls) != 0 {
			t.Errorf("after close the flight holds %d runs and the run's timer is %v; want none and nil", len(f.calls), c.timer)
		}
	})
}

// TestFlight_ARunThatEndsInItsGracePeriodLeavesNothingBehind: a run that
// finishes with nobody waiting stops its timer and frees its key, and no
// goroutine of the flight outlives it (synctest fails a bubble that leaves one
// blocked).
func TestFlight_ARunThatEndsInItsGracePeriodLeavesNothingBehind(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		f := flight{grace: 30 * time.Second}
		finish := make(chan struct{})
		_ = within(time.Second, &f, func(context.Context) error { <-finish; return nil })
		f.mu.Lock()
		c := f.calls[flightKey]
		f.mu.Unlock()
		close(finish)
		synctest.Wait()
		f.mu.Lock()
		if c == nil || c.timer != nil || len(f.calls) != 0 {
			t.Errorf("after the run ended the flight holds %d runs and the run's timer is set: %v", len(f.calls), c != nil && c.timer != nil)
		}
		f.mu.Unlock()
		// a caller that comes later starts a run of its own
		ran := false
		if err := run(context.Background(), &f, func(context.Context) error { ran = true; return nil }); err != nil || !ran {
			t.Errorf("the next caller: ran %v, err %v; want a new run", ran, err)
		}
	})
}

// TestFlight_NoGoroutineOutlivesClose: callers on deadlines shorter and longer
// than the run and the grace period come and go on the real clock, then the
// flight is closed. Every run and every timer of it is gone after that. Run
// it with -race.
func TestFlight_NoGoroutineOutlivesClose(t *testing.T) {
	before := runtime.NumGoroutine()
	f := flight{grace: 2 * time.Millisecond}
	fn := func(ctx context.Context) error {
		select {
		case <-time.After(time.Millisecond):
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	var wg sync.WaitGroup
	for g := range 16 {
		wg.Go(func() {
			for i := range 100 {
				d := time.Duration(100+(g*7+i*13)%2900) * time.Microsecond
				ctx, cancel := context.WithTimeout(context.Background(), d)
				_, err := f.do(ctx, Space{"k", string(rune('a' + g%3))}, "tag", fn)
				cancel()
				// context.Canceled is the error of a run that close cut short
				if err != nil && !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) && !errors.Is(err, langai.ErrClosed) {
					t.Errorf("err = %v, want nil, the caller's deadline, a closed run's context.Canceled or langai.ErrClosed", err)
				}
			}
		})
	}
	time.Sleep(20 * time.Millisecond)
	f.close()
	wg.Wait()
	f.mu.Lock()
	left := len(f.calls)
	f.mu.Unlock()
	if left != 0 {
		t.Errorf("after close the flight holds %d runs, want none", left)
	}
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > before && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if n := runtime.NumGoroutine(); n > before {
		t.Errorf("%d goroutines after close, %d before the flight: something of it is still running", n, before)
	}
}

// slowCatalog returns breadWithRecipes whose Tags takes d on the bubble's
// clock, each time it is read.
func slowCatalog(d time.Duration) *fakeCatalog {
	cat := breadWithRecipes("1")
	cat.delay = d
	return cat
}

// TestEngine_ARebuildLongerThanItsCallersDeadlinesCompletes: the catalog takes
// 60 ms to read and every request gives up after 40. Each rebuild used to be
// cancelled as its one caller left, so no call ever succeeded and the catalog
// was read once per call. Now the first caller's rebuild goes on, and every
// later caller finds the space built.
func TestEngine_ARebuildLongerThanItsCallersDeadlinesCompletes(t *testing.T) {
	for _, tc := range []struct {
		name string
		// pause is how long after one caller gave up the next one comes
		pause time.Duration
	}{
		{"the next caller joins the rebuild", 0},
		{"the rebuild ends with nobody waiting and records the revision", 100 * time.Millisecond},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				cat := slowCatalog(60 * time.Millisecond)
				r := newRig(t, engineCfg, cat, nil, nil)
				defer r.e.close()
				ok := 0
				for i := range 15 {
					ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
					ss, err := r.e.scores(ctx, "1", "10", "r1", croissantText)
					cancel()
					switch {
					case err == nil && len(ss) == 2:
						ok++
					case i == 0 && errors.Is(err, context.DeadlineExceeded):
					default:
						t.Errorf("call %d: %+v, %v", i+1, ss, err)
					}
					if i == 0 {
						time.Sleep(tc.pause)
					}
				}
				if ok != 14 {
					t.Errorf("%d of 15 calls succeeded, want 14: all but the one that started the rebuild", ok)
				}
				if tags, _ := cat.counts(); tags != 1 {
					t.Errorf("the catalog's tags were read %d times, want once", tags)
				}
			})
		})
	}
}

// TestEngine_RebuildGrace: the grace period is TaggerConfig.RebuildGrace, 30
// seconds when zero and none when negative, and Close does not wait for it.
func TestEngine_RebuildGrace(t *testing.T) {
	for _, tc := range []struct {
		name  string
		grace time.Duration
		want  time.Duration // when the hung rebuild is cancelled, after its caller left
	}{
		{"the default", 0, DefaultRebuildGrace},
		{"a period of its own", 5 * time.Second, 5 * time.Second},
		{"none", -1, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				cfg := engineCfg
				cfg.RebuildGrace = tc.grace
				r := newRig(t, cfg, &fakeCatalog{tags: breadAndJapan("1")}, nil, nil)
				defer r.e.close()
				r.emb.hangOn(langai.RoleDocument, true)
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				if _, err := r.e.scores(ctx, "1", "10", "1", diaryText); !errors.Is(err, context.DeadlineExceeded) {
					t.Errorf("err = %v, want the caller's deadline", err)
				}
				if tc.want > 0 {
					time.Sleep(tc.want - time.Nanosecond)
					synctest.Wait()
					if n := r.emb.gaveUp(); n != 0 {
						t.Errorf("the rebuild was cancelled before %v had passed", tc.want)
					}
					time.Sleep(time.Nanosecond)
				}
				synctest.Wait()
				if n := r.emb.gaveUp(); n != 1 {
					t.Errorf("%d hanging Embed calls were cancelled %v after the caller left, want 1: the rebuild's", n, tc.want)
				}
			})
		})
	}

	t.Run("Close does not wait for it", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			r := newRig(t, engineCfg, &fakeCatalog{tags: breadAndJapan("1")}, nil, nil)
			r.emb.hangOn(langai.RoleDocument, true)
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			_, _ = r.e.scores(ctx, "1", "10", "1", diaryText)
			start := time.Now()
			r.e.close()
			if n := r.emb.gaveUp(); n != 1 || time.Since(start) != 0 {
				t.Errorf("close returned after %v with %d hanging Embed calls cancelled; want at once and 1", time.Since(start), n)
			}
		})
	})
}

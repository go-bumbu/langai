package tagging

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/go-bumbu/langai"
)

// flightKey is the key the flight tests run under.
var flightKey = spaceKey{"k", ""}

// run is flight.do for one key and tag, for the tests that have no use for
// again.
func run(ctx context.Context, f *flight, fn func(context.Context) error) error {
	_, err := f.do(ctx, flightKey, "tag", fn)
	return err
}

// TestFlight_RunsOncePerKey holds the run until all ten callers are provably
// blocked, one in fn and nine on its run, so none can arrive after it ended
// and start a second one.
func TestFlight_RunsOncePerKey(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var f flight
		var runs, finished, others atomic.Int32
		release := make(chan struct{})
		fn := func(context.Context) error {
			runs.Add(1)
			<-release
			return nil
		}
		var wg sync.WaitGroup
		for range 10 {
			wg.Go(func() {
				_ = run(context.Background(), &f, fn)
				finished.Add(1)
			})
		}
		synctest.Wait()
		// not Fatalf: the callers must get to finish for the bubble to end
		if n := runs.Load(); n != 1 {
			t.Errorf("with every caller blocked, fn ran %d times, want once", n)
		}
		if n := finished.Load(); n != 0 {
			t.Errorf("%d callers finished before the run did", n)
		}
		// another key runs on its own, while the first is in flight
		other := func(context.Context) error { others.Add(1); return nil }
		if _, err := f.do(context.Background(), spaceKey{"other", ""}, "tag", other); err != nil {
			t.Error(err)
		}
		if others.Load() != 1 {
			t.Error("the other key did not run")
		}
		close(release)
		wg.Wait()
		if n := runs.Load(); n != 1 {
			t.Fatalf("fn ran %d times for one key, want once", n)
		}
	})
}

func TestFlight_SharesTheErrorAndRunsAgainAfter(t *testing.T) {
	var f flight
	if err := run(context.Background(), &f, func(context.Context) error { return errBoom }); !errors.Is(err, errBoom) {
		t.Fatalf("err = %v, want boom", err)
	}
	ran := false
	if err := run(context.Background(), &f, func(context.Context) error { ran = true; return nil }); err != nil || !ran {
		t.Fatalf("second do: ran %v, err %v; want a fresh run", ran, err)
	}
}

// TestFlight_TheRunEndsWhenItsLastCallerLeaves: a caller that gives up leaves
// the run to the others, and it carries on with its context alive. When the
// last one has gone nobody waits for the run, and its context is cancelled: a
// run used to go on under context.WithoutCancel whoever was left, so a hung
// embedder held its goroutine for ever.
func TestFlight_TheRunEndsWhenItsLastCallerLeaves(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var f flight
		var ended atomic.Bool
		fn := func(ctx context.Context) error {
			<-ctx.Done()
			ended.Store(true)
			return ctx.Err()
		}
		first, leaveFirst := context.WithCancel(context.Background())
		second, leaveSecond := context.WithCancel(context.Background())
		errs := make(chan error, 2)
		go func() { errs <- run(first, &f, fn) }()
		synctest.Wait()
		go func() { errs <- run(second, &f, func(context.Context) error { return errors.New("a second run") }) }()
		synctest.Wait()

		leaveFirst()
		if err := <-errs; !errors.Is(err, context.Canceled) {
			t.Errorf("the caller that left: err = %v, want context.Canceled", err)
		}
		synctest.Wait()
		if ended.Load() {
			t.Error("the run was cancelled although a caller still waits for it")
		}
		leaveSecond()
		if err := <-errs; !errors.Is(err, context.Canceled) {
			t.Errorf("the last caller: err = %v, want context.Canceled", err)
		}
		synctest.Wait()
		if !ended.Load() {
			t.Error("the run goes on with nobody waiting for it")
		}
	})
}

// TestFlight_ANewCallerWaitsForARunThatWasLeft: a run whose callers have all
// gone still holds its key until it returns, so the next caller's run does not
// overlap with it.
func TestFlight_ANewCallerWaitsForARunThatWasLeft(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var f flight
		var second atomic.Bool
		finish := make(chan struct{})
		slow := func(context.Context) error { <-finish; return nil } // takes no notice of its ctx
		ctx, leave := context.WithCancel(context.Background())
		gone := make(chan error, 1)
		go func() { gone <- run(ctx, &f, slow) }()
		synctest.Wait()
		leave()
		<-gone

		done := make(chan error, 1)
		go func() {
			done <- run(context.Background(), &f, func(context.Context) error { second.Store(true); return nil })
		}()
		synctest.Wait()
		if second.Load() {
			t.Error("a second run started while the one that was left still runs")
		}
		close(finish)
		if err := <-done; err != nil || !second.Load() {
			t.Errorf("the new caller: ran %v, err %v; want its own run once the other ended", second.Load(), err)
		}

		// and it does not wait for ever: its own ctx still counts
		finish = make(chan struct{})
		ctx, leave = context.WithCancel(context.Background())
		go func() { gone <- run(ctx, &f, slow) }()
		synctest.Wait()
		leave()
		<-gone
		short, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := run(short, &f, func(context.Context) error { return nil }); !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("a caller behind a run that never ends: err = %v, want its deadline", err)
		}
		close(finish)
	})
}

// TestFlight_APanicIsTheRunsError: fn runs on the flight's goroutine, where no
// caller can recover a panic, so the flight does: the callers get an error
// with the value, the stack goes to onPanic, and the key is free again.
func TestFlight_APanicIsTheRunsError(t *testing.T) {
	var f flight
	var stacks []string
	f.onPanic = func(_ any, stack []byte) { stacks = append(stacks, string(stack)) }
	err := run(context.Background(), &f, func(context.Context) error { panic("kaboom") })
	var panicked *panicError
	if !errors.As(err, &panicked) || panicked.value != "kaboom" || !strings.Contains(err.Error(), "kaboom") {
		t.Fatalf("err = %v, want one holding the panic's value", err)
	}
	if len(stacks) != 1 || !strings.Contains(stacks[0], "TestFlight_APanicIsTheRunsError") {
		t.Errorf("onPanic got %d stacks, want one that shows where it panicked", len(stacks))
	}
	if err := run(context.Background(), &f, func(context.Context) error { return nil }); err != nil {
		t.Fatalf("the run after a panic: %v", err)
	}
}

// TestFlight_RunsOfOneKeyFollowEachOtherWhateverTheTag: a caller that wants
// another tag than the run in flight waits for it and is told to look again;
// its own fn has not run.
func TestFlight_RunsOfOneKeyFollowEachOtherWhateverTheTag(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var f flight
		var other atomic.Bool
		release := make(chan struct{})
		first := make(chan error, 1)
		go func() {
			_, err := f.do(context.Background(), flightKey, "a", func(context.Context) error { <-release; return errBoom })
			first <- err
		}()
		synctest.Wait()
		type outcome struct {
			again bool
			err   error
		}
		second := make(chan outcome, 1)
		go func() {
			again, err := f.do(context.Background(), flightKey, "b", func(context.Context) error { other.Store(true); return nil })
			second <- outcome{again, err}
		}()
		synctest.Wait()
		if other.Load() {
			t.Error("the run for another tag started beside the one in flight")
		}
		close(release)
		if err := <-first; !errors.Is(err, errBoom) {
			t.Errorf("the first caller: err = %v, want its run's", err)
		}
		if got := <-second; !got.again || got.err != nil || other.Load() {
			t.Errorf("the caller of another tag: again %v, err %v, ran %v; want again, no error and no run", got.again, got.err, other.Load())
		}
	})
}

// TestFlight_CloseCancelsTheRunsAndWaitsForThem: after close nothing of the
// flight runs, and nothing starts.
func TestFlight_CloseCancelsTheRunsAndWaitsForThem(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var f flight
		var ended atomic.Bool
		caller := make(chan error, 1)
		go func() {
			caller <- run(context.Background(), &f, func(ctx context.Context) error {
				<-ctx.Done()
				ended.Store(true)
				return ctx.Err()
			})
		}()
		synctest.Wait()
		f.close()
		if !ended.Load() {
			t.Error("close returned before the run ended")
		}
		if err := <-caller; !errors.Is(err, context.Canceled) {
			t.Errorf("the caller of the run that was closed: err = %v, want its run's context.Canceled", err)
		}
		ran := false
		if err := run(context.Background(), &f, func(context.Context) error { ran = true; return nil }); !errors.Is(err, ErrClosed) || ran {
			t.Errorf("do after close: ran %v, err %v; want ErrClosed and no run", ran, err)
		}
		f.close() // closing twice is fine
	})
}

// TestEngine_ARebuildNobodyWaitsForIsCancelled: the embedder hangs on the tag
// names, and the one caller gives up at its deadline. The rebuild used to stay
// blocked in the embedder for ever, on a context nothing could cancel; now it
// is cancelled as the caller leaves, and the next call starts afresh.
func TestEngine_ARebuildNobodyWaitsForIsCancelled(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, engineCfg, &fakeCatalog{tags: breadAndJapan("1")}, nil, nil)
		r.emb.hangOn(langai.RoleDocument, true)
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if _, err := r.e.scores(ctx, "1", "10", "1", diaryText); !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("err = %v, want the caller's deadline", err)
		}
		synctest.Wait()
		if n := r.emb.gaveUp(); n != 1 {
			t.Errorf("%d hanging Embed calls were cancelled, want 1: the rebuild's", n)
		}
		r.emb.hangOn(langai.RoleDocument, false)
		if got := picked(r.scores(t, "1", diaryText)); len(got) != 1 || got[0] != "japan" {
			t.Errorf("the next call: %v, want [japan]", got)
		}
	})
}

// TestEngine_APanicInARebuildIsAnError: a Catalog or an Embedder that panics
// in a rebuild used to end the process, from a goroutine no caller could
// recover on. It is now the error of the callers, wrapping ErrNotRebuilt and
// saying what was panicked with, and the stack is logged.
func TestEngine_APanicInARebuildIsAnError(t *testing.T) {
	cfg, rec := loggedCfg()
	cat := &fakeCatalog{tags: breadAndJapan("1")}
	r := newRig(t, cfg, cat, nil, nil)

	cat.setPanic("catalog exploded")
	ss, err := r.e.scores(context.Background(), "1", "10", "1", diaryText)
	if !errors.Is(err, ErrNotRebuilt) || !strings.Contains(err.Error(), "catalog exploded") || ss != nil {
		t.Fatalf("with a catalog that panics: %+v, %v; want no scores and an error wrapping ErrNotRebuilt with the value", ss, err)
	}
	cat.setPanic(nil)

	r.emb.atCall(func(int) error { panic(errBoom) }) // the item is cached: the next Embed is the rebuild's
	_, err = r.e.scores(context.Background(), "1", "10", "1", diaryText)
	if !errors.Is(err, ErrNotRebuilt) || !strings.Contains(err.Error(), errBoom.Error()) {
		t.Fatalf("with an embedder that panics: err = %v, want one wrapping ErrNotRebuilt with the value", err)
	}
	r.emb.atCall(nil)

	if n := len(rec.matching("ERROR", "panic", "stack=")); n != 2 {
		t.Errorf("%d panics were logged with their stack, want 2; log:\n%s", n, strings.Join(rec.lines(), "\n"))
	}
	if got := picked(r.scores(t, "1", diaryText)); len(got) != 1 || got[0] != "japan" {
		t.Errorf("after the panics: %v, want [japan]", got)
	}
}

// TestEngine_CallersAtTwoRevisionsRebuildOneAfterTheOther: two instances of an
// application, one a revision behind, ask about one space together. Their
// rebuilds used to run side by side, one flight per revision, and the second
// Apply found the space moved and left it recorded as nobody's, so every later
// call read the catalog again. Now the second waits for the first, and the
// space ends recorded at the revision of whichever came last.
func TestEngine_CallersAtTwoRevisionsRebuildOneAfterTheOther(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cat := breadWithRecipes("1")
		cat.release = make(chan struct{})
		r := newRig(t, engineCfg, cat, nil, nil)
		errs := make([]error, 2)
		var wg sync.WaitGroup
		for i, rev := range []string{"r1", "r2"} {
			wg.Go(func() { _, errs[i] = r.e.scores(context.Background(), "1", "10", rev, croissantText) })
		}
		synctest.Wait()
		// not Fatalf: the callers must get to finish for the bubble to end
		if tags, _ := cat.counts(); tags != 1 {
			t.Errorf("with both callers blocked, %d rebuilds read the catalog, want 1: the other waits", tags)
		}
		close(cat.release)
		wg.Wait()
		if errs[0] != nil || errs[1] != nil {
			t.Fatalf("errs = %v", errs)
		}
		if n := cat.overlap(); n != 1 {
			t.Errorf("%d rebuilds of one space ran together, want one at a time", n)
		}
		state, _, ok, err := r.ix.State(context.Background(), "1", "10")
		if err != nil || !ok || (state != r.e.state("r1") && state != r.e.state("r2")) {
			t.Errorf("the space is recorded at %q (known %v, %v), want the state of r1 or of r2", state, ok, err)
		}
	})
}

// TestTagger_Close: Close cancels the rebuild in flight and waits for it, its
// caller learns the Tagger was closed, and every call after it is refused.
func TestTagger_Close(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		emb := &tallyEmbedder{}
		tg, err := NewTagger(emb, &fakeCatalog{tags: breadAndJapan("1")}, NewMemCache(), engineCfg)
		if err != nil {
			t.Fatal(err)
		}
		space := tg.Scope("1", "10")
		ctx := context.Background()
		emb.hangOn(langai.RoleDocument, true)
		caller := make(chan error, 1)
		go func() {
			_, err := space.Suggest(ctx, "1", diaryText)
			caller <- err
		}()
		synctest.Wait()

		if err := tg.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
		if n := emb.gaveUp(); n != 1 {
			t.Errorf("Close returned with %d hanging Embed calls cancelled, want 1: it waits for the rebuild", n)
		}
		if err := <-caller; !errors.Is(err, ErrClosed) {
			t.Errorf("the call Close cut short: err = %v, want one wrapping ErrClosed", err)
		}
		emb.hangOn(langai.RoleDocument, false)
		embedded := emb.total()
		for name, call := range map[string]func() error{
			"Suggest": func() error { _, err := space.Suggest(ctx, "1", diaryText); return err },
			"Scores":  func() error { _, err := space.Scores(ctx, "1", diaryText); return err },
			"Warm":    func() error { return space.Warm(ctx, "1", guideText) },
			"Touch":   func() error { return space.Touch(ctx) },
			"Prune":   func() error { _, err := tg.Prune(ctx, time.Now().Add(-48*time.Hour)); return err },
		} {
			if err := call(); !errors.Is(err, ErrClosed) {
				t.Errorf("%s after Close: err = %v, want ErrClosed", name, err)
			}
		}
		if n := emb.total() - embedded; n != 0 {
			t.Errorf("calls after Close embedded %d texts, want none", n)
		}
		if err := tg.Close(); err != nil {
			t.Errorf("a second Close: %v", err)
		}
	})
}

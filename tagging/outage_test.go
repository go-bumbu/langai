package tagging

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
)

// outageRig is a rig over a langaitest.FakeEmbedder a test takes down and
// brings back: while down holds an error every Embed fails with it, and a text
// containing "boom" always fails with errBoom, alone or in a batch. calls
// counts the Embed calls, the failed ones included.
type outageRig struct {
	*rig
	calls atomic.Int64
	down  atomic.Pointer[error]
}

// newOutageRig returns an outageRig over cat, with no Logger: what it checks
// must be observable without one.
func newOutageRig(t *testing.T, cat *fakeCatalog) *outageRig {
	t.Helper()
	o := &outageRig{rig: newRig(t, engineCfg, cat, nil, nil)}
	o.emb.inner = langaitest.FakeEmbedder{Calls: &o.calls, Fail: func(text string) error {
		if cause := o.down.Load(); cause != nil {
			return *cause
		}
		if strings.Contains(text, "boom") {
			return errBoom
		}
		return nil
	}}
	return o
}

// fail takes the embedder down with cause; nil brings it back.
func (o *outageRig) fail(cause error) {
	if cause == nil {
		o.down.Store(nil)
		return
	}
	o.down.Store(&cause)
}

// recorded returns the state the index records for the rig's space and the
// ids of the tags it holds as built without one of their members.
func (o *outageRig) recorded(t *testing.T) (state string, skipped []string) {
	t.Helper()
	state, held, _, err := o.ix.State(context.Background(), "1", "10")
	if err != nil {
		t.Fatal(err)
	}
	for id, fp := range held {
		if strings.HasSuffix(fp, skippedMark) {
			skipped = append(skipped, id)
		}
	}
	slices.Sort(skipped)
	return state, skipped
}

// breadAt is ExampleTagger's catalog with the bread tag at revision rev and
// the given members.
func breadAt(rev string, members []string) ([]CatalogTag, map[string][]string) {
	return []CatalogTag{{"bread", "cooking / bread", rev}, {"japan", "travel / japan", "1"}}, map[string][]string{"bread": members}
}

// TestEngine_AnEmbedderOutageIsNotRecordedAsRebuilt: a tag whose name is
// cached gets 40 new members while the embedder is down. The rebuild used to
// call it once per text, take every member for a text it cannot embed, and
// record the revision as built from the name alone: the tag scored wrong, with
// a nil error, until the space's next revision. An embedder that fails on
// everything, the canary included, is down: the rebuild fails after a handful
// of calls, records nothing, and the call after the outage builds the tag
// from all of its members.
func TestEngine_AnEmbedderOutageIsNotRecordedAsRebuilt(t *testing.T) {
	deadline := fmt.Errorf("embedder: post: %w", context.DeadlineExceeded)
	for _, tc := range []struct {
		name     string
		cause    error
		maxCalls int64
	}{
		// the batch, the texts tried alone, and the canary
		{"connection refused", errors.New("dial tcp 10.0.0.7:8080: connect: connection refused"), 1 + maxEmbedFailuresInARow + 1},
		// a timeout of the embedder's own is no bad text: nothing is tried again
		{"a timeout of the embedder's own", deadline, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			o := newOutageRig(t, breadWithRecipes("1"))
			o.scores(t, "r1", croissantText)
			built := o.e.state("r1")

			members := append(slices.Clone(croissantRecipes), recipes(40)...)
			o.cat.set(breadAt("2", members))
			o.fail(tc.cause)
			before := o.calls.Load()
			ss, err := o.e.scores(ctx, "1", "10", "r2", croissantText)
			if !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, tc.cause) || ss != nil {
				t.Fatalf("during the outage: %+v, %v; want no scores and an error wrapping ErrNotRebuilt and the cause", ss, err)
			}
			if n := o.calls.Load() - before; n > tc.maxCalls {
				t.Errorf("the rebuild called a dead embedder %d times, want at most %d", n, tc.maxCalls)
			}
			if state, skipped := o.recorded(t); state != built || len(skipped) != 0 {
				t.Errorf("after the outage the space is recorded at %q with %v built without members; want it left at r1's %q", state, skipped, built)
			}
			if ids, err := o.e.incomplete(ctx, "1", "10"); err != nil || len(ids) != 0 {
				t.Errorf("incomplete = %v, %v; want none: nothing was built", ids, err)
			}

			o.fail(nil)
			got := o.scores(t, "r2", croissantText)
			healthy := newRig(t, engineCfg, &fakeCatalog{}, nil, nil)
			healthy.cat.set(breadAt("2", members))
			if want := healthy.scores(t, "r2", croissantText); !reflect.DeepEqual(got, want) {
				t.Errorf("after the outage: %+v, want a healthy build's %+v", got, want)
			}
			if state, skipped := o.recorded(t); state != o.e.state("r2") || len(skipped) != 0 {
				t.Errorf("after the recovery the space is at %q with %v built without members; want r2's state and none", state, skipped)
			}
		})
	}
}

// TestEngine_AnOutageThatStartsMidRebuildIsNotRecorded: the embedder took the
// first batches of members and then went down. That it worked earlier in the
// call says nothing about the texts that failed after: only an Embed that
// works after them does. The batches embedded before stay cached, and the
// call after the outage embeds the rest.
func TestEngine_AnOutageThatStartsMidRebuildIsNotRecorded(t *testing.T) {
	refused := errors.New("connection refused")
	for _, tc := range []struct {
		name  string
		added int
	}{
		{"with a batch of texts left", 2*embedBatchSize + 20},
		{"with one text left", 2*embedBatchSize + 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			o := newOutageRig(t, breadWithRecipes("1"))
			o.scores(t, "r1", croissantText)
			built := o.e.state("r1")

			added := recipes(tc.added)
			o.cat.set(breadAt("2", append(slices.Clone(croissantRecipes), added...)))
			var calls atomic.Int64
			o.emb.inner = langaitest.FakeEmbedder{Calls: &calls, Fail: func(string) error {
				if calls.Load() > 2 { // the third Embed of the rebuild and every one after
					return refused
				}
				return nil
			}}
			if _, err := o.e.scores(ctx, "1", "10", "r2", croissantText); !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, refused) {
				t.Fatalf("err = %v, want one wrapping ErrNotRebuilt and the cause", err)
			}
			if n, most := calls.Load(), int64(2+1+maxEmbedFailuresInARow+1); n > most {
				t.Errorf("%d Embed calls, want at most %d", n, most)
			}
			if state, skipped := o.recorded(t); state != built || len(skipped) != 0 {
				t.Errorf("the space is recorded at %q with %v built without members; want it left at r1's %q", state, skipped, built)
			}
			for i, ok := range cached(t, o.cache, added...) {
				if want := slices.Index(added, i) < 2*embedBatchSize; ok != want {
					t.Errorf("cached(%q) = %v, want %v: the two batches before the outage only", i, ok, want)
				}
			}

			o.emb.inner = langaitest.FakeEmbedder{}
			embedded := o.emb.count(langai.RoleQuery)
			o.scores(t, "r2", croissantText)
			if n, want := o.emb.count(langai.RoleQuery)-embedded, tc.added-2*embedBatchSize; n != want {
				t.Errorf("the call after the outage embedded %d members, want %d: the ones that were missing", n, want)
			}
			if state, skipped := o.recorded(t); state != o.e.state("r2") || len(skipped) != 0 {
				t.Errorf("after the recovery the space is at %q with %v built without members; want r2's state and none", state, skipped)
			}
		})
	}
}

// TestEngine_AClosedEmbedderFailsTheRebuild: an embedder that was closed says
// so with an error wrapping langai.ErrClosed, whatever the text. That is the
// whole call's error: nothing is tried again one text at a time, no member is
// skipped, and nothing is recorded.
func TestEngine_AClosedEmbedderFailsTheRebuild(t *testing.T) {
	ctx := context.Background()
	o := newOutageRig(t, breadWithRecipes("1"))
	o.scores(t, "r1", croissantText)
	built := o.e.state("r1")

	o.cat.set(breadAt("2", append(slices.Clone(croissantRecipes), recipes(40)...)))
	o.fail(fmt.Errorf("yzma: embedder: %w", langai.ErrClosed))
	before := o.calls.Load()
	ss, err := o.e.scores(ctx, "1", "10", "r2", croissantText)
	if !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, langai.ErrClosed) || ss != nil {
		t.Fatalf("with a closed embedder: %+v, %v; want no scores and an error wrapping ErrNotRebuilt and langai.ErrClosed", ss, err)
	}
	if n := o.calls.Load() - before; n != 1 {
		t.Errorf("a closed embedder was called %d times, want once", n)
	}
	if state, skipped := o.recorded(t); state != built || len(skipped) != 0 {
		t.Errorf("the space is recorded at %q with %v built without members; want it left at r1's %q", state, skipped, built)
	}

	// the same for the text of the call itself, and for a Warm
	if _, err := o.e.scores(ctx, "1", "10", "r1", "a text nobody asked about"); !errors.Is(err, langai.ErrClosed) {
		t.Errorf("scores of a new text: err = %v, want one wrapping langai.ErrClosed", err)
	}
	before = o.calls.Load()
	if err := o.e.warm(ctx, "1", "10", "r1", recipes(100)); !errors.Is(err, langai.ErrClosed) {
		t.Errorf("warm: err = %v, want one wrapping langai.ErrClosed", err)
	}
	if n := o.calls.Load() - before; n != 1 {
		t.Errorf("warm called a closed embedder %d times, want once", n)
	}
}

// TestEngine_ALazyEmbedderThatCannotLoadFailsTheRebuild: a LazyEmbedder keeps
// its load failure and returns it from every Embed. That is an embedder that
// is down, not a space full of texts it cannot take.
func TestEngine_ALazyEmbedderThatCannotLoadFailsTheRebuild(t *testing.T) {
	notFound := errors.New("model file not found")
	var loads atomic.Int64
	lazy := langai.NewLazyEmbedder("fake", 64, func(context.Context) (langai.Embedder, error) {
		loads.Add(1)
		return nil, notFound
	})
	t.Cleanup(func() { _ = lazy.Close() })
	ix := NewMemIndex(0)
	e, err := newEngine(lazy, breadWithRecipes("1"), NewMemCache(), ix, engineCfg)
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := e.warm(context.Background(), "1", "10", "r1", nil); !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, notFound) {
			t.Fatalf("warm: err = %v, want one wrapping ErrNotRebuilt and the load failure", err)
		}
		if _, _, ok, err := ix.State(context.Background(), "1", "10"); ok || err != nil {
			t.Fatalf("the space was recorded (known %v, %v), want nothing", ok, err)
		}
	}
	if n := loads.Load(); n != 1 {
		t.Errorf("the model was loaded %d times, want once", n)
	}
}

// TestEngine_ABadTextAmongGoodOnesIsStillSkipped: the embedder works and
// fails on some texts only. Each of those is skipped once an Embed after it
// has worked, a later text's or the canary's, so one member the embedder
// cannot take does not block its space, wherever it stands among the others.
// The skip can be seen without a Logger, and the member gets another try at
// the space's next revision.
func TestEngine_ABadTextAmongGoodOnesIsStillSkipped(t *testing.T) {
	poison := func(i int) string { return fmt.Sprintf("A recipe the embedder chokes on, boom %d", i) }
	good := recipes(40)
	for _, tc := range []struct {
		name    string
		added   []string
		skipped int
	}{
		{"in the middle of a batch", slices.Insert(slices.Clone(good), 20, poison(0)), 1},
		{"the last text of the call", append(slices.Clone(good), poison(0)), 1},
		{"the only text of the call", []string{poison(0)}, 1},
		{"every text of the call", []string{poison(0), poison(1), poison(2), poison(3), poison(4), poison(5), poison(6)}, 7},
		{"more in a row than are tried before the canary", slices.Concat(good[:5], []string{poison(0), poison(1), poison(2), poison(3)}, good[5:]), 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			o := newOutageRig(t, breadWithRecipes("1"))
			o.scores(t, "r1", croissantText)

			members := append(slices.Clone(croissantRecipes), tc.added...)
			o.cat.set(breadAt("2", members))
			got := o.scores(t, "r2", croissantText)

			var taken []string
			for _, m := range members {
				if !strings.Contains(m, "boom") {
					taken = append(taken, m)
				}
			}
			if len(members)-len(taken) != tc.skipped {
				t.Fatalf("the case holds %d bad texts, want %d", len(members)-len(taken), tc.skipped)
			}
			healthy := newRig(t, engineCfg, &fakeCatalog{}, nil, nil)
			healthy.cat.set(breadAt("2", taken))
			if want := healthy.scores(t, "r2", croissantText); !reflect.DeepEqual(got, want) {
				t.Errorf("scores %+v, want those of the tag built from its other members, %+v", got, want)
			}
			for text, ok := range cached(t, o.cache, members...) {
				if ok == strings.Contains(text, "boom") {
					t.Errorf("cached(%q) = %v", text, ok)
				}
			}
			if state, skipped := o.recorded(t); state != o.e.state("r2") || !slices.Equal(skipped, []string{"bread"}) {
				t.Errorf("the space is at %q with %v built without members; want r2's state and [bread]", state, skipped)
			}
			if ids, err := o.e.incomplete(ctx, "1", "10"); err != nil || !slices.Equal(ids, []string{"bread"}) {
				t.Errorf("incomplete = %v, %v; want [bread]", ids, err)
			}

			// the next revision of the space tries the members again, although
			// the tag's own revision is the same
			o.emb.inner = langaitest.FakeEmbedder{}
			o.scores(t, "r3", croissantText)
			if ids, err := o.e.incomplete(ctx, "1", "10"); err != nil || len(ids) != 0 {
				t.Errorf("incomplete after the next revision = %v, %v; want none", ids, err)
			}
			for text, ok := range cached(t, o.cache, members...) {
				if !ok {
					t.Errorf("%q was not embedded at the next revision", text)
				}
			}
		})
	}
}

// TestEngine_ACachedTextVouchesForTheEmbedder: the one text a call has to
// embed fails, and nothing of the call embeds after it. A text the cache holds
// was taken by the embedder before, so embedding it once more tells a bad
// text from an embedder that is down: the error still names the tag whose new
// name could not be embedded, and a Warm the text it could not embed.
func TestEngine_ACachedTextVouchesForTheEmbedder(t *testing.T) {
	ctx := context.Background()
	o := newOutageRig(t, breadWithRecipes("1"))
	o.scores(t, "r1", croissantText)

	tags, members := breadAt("1", croissantRecipes)
	o.cat.set(append(tags, CatalogTag{"new", "a name that goes boom", "1"}), members)
	before := o.calls.Load()
	_, err := o.e.scores(ctx, "1", "10", "r2", croissantText)
	if !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, errBoom) || !strings.Contains(err.Error(), `"new"`) {
		t.Errorf("err = %v, want one wrapping ErrNotRebuilt and the cause, naming the tag new", err)
	}
	if n := o.calls.Load() - before; n != 2 {
		t.Errorf("%d Embed calls, want 2: the name, and a cached name once more", n)
	}

	err = o.e.warm(ctx, "1", "10", "r1", []string{croissantText, poisonText})
	if !errors.Is(err, errBoom) || !strings.Contains(err.Error(), "text 2 of 2") {
		t.Errorf("warm: err = %v, want one wrapping the cause and naming text 2 of 2", err)
	}

	// with the embedder down the same calls end with its error, naming nothing
	refused := errors.New("connection refused")
	o.fail(refused)
	if _, err := o.e.scores(ctx, "1", "10", "r2", croissantText); !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, refused) || strings.Contains(err.Error(), `"new"`) {
		t.Errorf("with the embedder down: err = %v, want one wrapping ErrNotRebuilt and the cause, and no tag named", err)
	}
	if err := o.e.warm(ctx, "1", "10", "r1", []string{croissantText, poisonText}); !errors.Is(err, refused) || strings.Contains(err.Error(), "text 2 of 2") {
		t.Errorf("warm with the embedder down: err = %v, want the cause and no text named", err)
	}
}

// TestScope_IncompleteNamesTheTagsBuiltWithoutAMember: through the Tagger,
// with no Logger, an application sees which tags were built without one of
// their members.
func TestScope_IncompleteNamesTheTagsBuiltWithoutAMember(t *testing.T) {
	ctx := context.Background()
	emb := langaitest.FakeEmbedder{Fail: func(text string) error {
		if strings.Contains(text, "boom") {
			return errBoom
		}
		return nil
	}}
	cat := &fakeCatalog{}
	cat.set(breadAt("1", append(slices.Clone(croissantRecipes), poisonText)))
	tg, err := NewTagger(emb, cat, NewMemCache(), engineCfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tg.Close() })
	space := tg.Scope(Space{"1", "10"})
	if ids, err := space.Incomplete(ctx); err != nil || len(ids) != 0 {
		t.Fatalf("Incomplete of a space never built = %v, %v; want none", ids, err)
	}
	if _, err := space.Suggest(ctx, "r1", croissantText); err != nil {
		t.Fatal(err)
	}
	if ids, err := space.Incomplete(ctx); err != nil || !slices.Equal(ids, []string{"bread"}) {
		t.Errorf("Incomplete = %v, %v; want [bread]", ids, err)
	}
	if _, err := tg.Scope(Space{"", "10"}).Incomplete(ctx); err == nil {
		t.Error("Incomplete with an empty index succeeded")
	}
}

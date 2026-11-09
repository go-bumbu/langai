package tagging

import (
	"context"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
)

// engineCfg is the configuration of ExampleTagger: FakeEmbedder's scores run
// lower than a real model's, so the two minimums are lowered to match.
var engineCfg = TaggerConfig{Model: "fake", NameMinScore: 0.3, MemberMinScore: 0.4}

// The texts of the croissant scenario of ExampleTagger.
const (
	guideText     = "Baking bread at home: a cooking guide"
	diaryText     = "A travel diary from Kyoto, Japan"
	croissantText = "Croissant dough with cold butter"
)

var croissantRecipes = []string{
	"Croissant dough: folding cold butter into layers",
	"Butter croissant dough, step by step",
	"Shaping croissant dough with butter",
}

// breadAndJapan is the two tags of ExampleTagger, both at revision rev.
func breadAndJapan(rev string) []CatalogTag {
	return []CatalogTag{{"bread", "cooking / bread", rev}, {"japan", "travel / japan", rev}}
}

// rig is an engine with the fakes behind it.
type rig struct {
	e     *engine
	emb   *tallyEmbedder
	cat   *fakeCatalog
	cache EmbeddingCache
	ix    TagIndex
}

// newRig returns an engine over a new tallyEmbedder and the given catalog,
// cache and index; a nil one is replaced by a new fake, MemCache or MemIndex.
func newRig(t *testing.T, cfg TaggerConfig, cat *fakeCatalog, cache EmbeddingCache, ix TagIndex) *rig {
	t.Helper()
	if cat == nil {
		cat = &fakeCatalog{}
	}
	if cache == nil {
		cache = NewMemCache()
	}
	if ix == nil {
		ix = NewMemIndex(0)
	}
	emb := &tallyEmbedder{}
	e, err := newEngine(emb, cat, cache, ix, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return &rig{e: e, emb: emb, cat: cat, cache: cache, ix: ix}
}

// scores returns the engine's scores of text in the space ("1", "10") at
// revision rev, failing the test on an error.
func (r *rig) scores(t *testing.T, rev, text string) []TagScore {
	t.Helper()
	return r.scoresIn(t, "1", rev, text)
}

// scoresIn is scores in the space (index, "10").
func (r *rig) scoresIn(t *testing.T, index, rev, text string) []TagScore {
	t.Helper()
	ss, err := r.e.scores(context.Background(), index, "10", rev, text)
	if err != nil {
		t.Fatalf("scores(%q, %q): %v", rev, text, err)
	}
	return ss
}

// picked returns the ids Suggest would return from ss.
func picked(ss []TagScore) []string {
	var ids []string
	for _, s := range ss {
		if s.Picked() {
			ids = append(ids, s.ID)
		}
	}
	return ids
}

// scoreOf returns the score of tag id in ss, failing the test when it is missing.
func scoreOf(t *testing.T, ss []TagScore, id string) TagScore {
	t.Helper()
	for _, s := range ss {
		if s.ID == id {
			return s
		}
	}
	t.Fatalf("no score for tag %q in %+v", id, ss)
	return TagScore{}
}

// breadWithRecipes is ExampleTagger's catalog once the croissant recipes are
// tagged with "cooking / bread" by hand.
func breadWithRecipes(rev string) *fakeCatalog {
	return &fakeCatalog{tags: breadAndJapan(rev), members: map[string][]string{"bread": croissantRecipes}}
}

func TestEngine_RequiresItsParts(t *testing.T) {
	emb, cat, cache, ix := langaitest.FakeEmbedder{}, &fakeCatalog{}, NewMemCache(), NewMemIndex(0)
	for _, tc := range []struct {
		name string
		err  func() error
	}{
		{"no embedder", func() error { _, err := newEngine(nil, cat, cache, ix, engineCfg); return err }},
		{"no catalog", func() error { _, err := newEngine(emb, nil, cache, ix, engineCfg); return err }},
		{"no cache", func() error { _, err := newEngine(emb, cat, nil, ix, engineCfg); return err }},
		{"no index", func() error { _, err := newEngine(emb, cat, cache, nil, engineCfg); return err }},
		{"no model", func() error { _, err := newEngine(emb, cat, cache, ix, TaggerConfig{}); return err }},
		{"NaN minimum", func() error {
			_, err := newEngine(emb, cat, cache, ix, TaggerConfig{Model: "m", NameMinScore: math.NaN()})
			return err
		}},
	} {
		if tc.err() == nil {
			t.Errorf("%s: newEngine succeeded, want an error", tc.name)
		}
	}
}

func TestEngine_NameOnlyTagsMatchByName(t *testing.T) {
	r := newRig(t, engineCfg, &fakeCatalog{tags: breadAndJapan("1")}, nil, nil)
	if got := picked(r.scores(t, "1", guideText)); !slices.Equal(got, []string{"bread"}) {
		t.Errorf("%q -> %v, want [bread]", guideText, got)
	}
	if got := picked(r.scores(t, "1", diaryText)); !slices.Equal(got, []string{"japan"}) {
		t.Errorf("%q -> %v, want [japan]", diaryText, got)
	}
}

func TestEngine_HandTaggedMembersTeachTheTag(t *testing.T) {
	cat := &fakeCatalog{tags: breadAndJapan("1")}
	r := newRig(t, engineCfg, cat, nil, nil)
	if got := picked(r.scores(t, "1", croissantText)); len(got) != 0 {
		t.Fatalf("before the recipes: %q -> %v, want []", croissantText, got)
	}
	cat.set([]CatalogTag{{"bread", "cooking / bread", "2"}, {"japan", "travel / japan", "1"}},
		map[string][]string{"bread": croissantRecipes})
	if got := picked(r.scores(t, "2", croissantText)); !slices.Equal(got, []string{"bread"}) {
		t.Fatalf("after the recipes: %q -> %v, want [bread]", croissantText, got)
	}
}

func TestEngine_UnchangedRevisionCallsNothing(t *testing.T) {
	r := newRig(t, engineCfg, breadWithRecipes("1"), nil, nil)
	first := r.scores(t, "1", croissantText)
	tags, members := r.cat.counts()
	embedded := r.emb.total()
	second := r.scores(t, "1", croissantText)
	t2, m2 := r.cat.counts()
	if t2 != tags || m2 != members || r.emb.total() != embedded {
		t.Fatalf("second call: %d Tags, %d Members calls and %d texts embedded, want 0 of each",
			t2-tags, m2-members, r.emb.total()-embedded)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("scores changed: %+v, then %+v", first, second)
	}
}

func TestEngine_RevisionChangeRebuildsOnlyChangedTags(t *testing.T) {
	tags := append(breadAndJapan("1"), CatalogTag{"rust", "dev / rust", "1"})
	cat := &fakeCatalog{tags: tags, members: map[string][]string{"bread": croissantRecipes}}
	r := newRig(t, engineCfg, cat, nil, nil)
	r.scores(t, "1", croissantText)
	_, members := r.cat.counts()
	names := r.emb.count(langai.RoleDocument)

	bumped := slices.Clone(tags)
	bumped[1].Revision = "2"
	cat.set(bumped, map[string][]string{"bread": croissantRecipes, "japan": {diaryText}})
	r.scores(t, "2", croissantText)
	if _, m := r.cat.counts(); m != members+1 {
		t.Fatalf("Members called %d times, want once", m-members)
	}
	if got := r.cat.lastMemberIDs(); !slices.Equal(got, []string{"japan"}) {
		t.Fatalf("Members asked for %v, want [japan]", got)
	}
	if got := r.emb.count(langai.RoleDocument); got != names {
		t.Fatalf("embedded %d names, want 0: the cache holds them", got-names)
	}
}

func TestEngine_RenameIsSeenWithoutATagRevision(t *testing.T) {
	cat := &fakeCatalog{tags: breadAndJapan("1")}
	r := newRig(t, engineCfg, cat, nil, nil)
	if got := picked(r.scores(t, "1", diaryText)); !slices.Equal(got, []string{"japan"}) {
		t.Fatalf("before the rename: %v, want [japan]", got)
	}
	renamed := []CatalogTag{{"bread", "cooking / bread", "1"}, {"japan", "sports / football", "1"}}
	cat.set(renamed, nil)
	got := r.scores(t, "2", diaryText)
	if n := r.emb.times(langai.RoleDocument, "sports / football"); n != 1 {
		t.Fatalf("the new name was embedded %d times, want once", n)
	}
	want := newRig(t, engineCfg, &fakeCatalog{tags: renamed}, nil, nil).scores(t, "1", diaryText)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("after the rename: %+v, want what a fresh engine gives, %+v", got, want)
	}
	if slices.Contains(picked(got), "japan") {
		t.Fatalf("after the rename the diary still gets japan: %+v", got)
	}
}

func TestEngine_DroppedTagDisappears(t *testing.T) {
	cat := &fakeCatalog{tags: breadAndJapan("1")}
	r := newRig(t, engineCfg, cat, nil, nil)
	scoreOf(t, r.scores(t, "1", diaryText), "japan")
	cat.set(breadAndJapan("1")[:1], nil)
	for _, s := range r.scores(t, "2", diaryText) {
		if s.ID == "japan" {
			t.Fatalf("dropped tag japan is still scored: %+v", s)
		}
	}
}

func TestEngine_CatalogErrorFailsAndRetries(t *testing.T) {
	boom := errors.New("catalog down")
	cat := &fakeCatalog{tags: breadAndJapan("1"), err: boom}
	r := newRig(t, engineCfg, cat, nil, nil)
	ss, err := r.e.scores(context.Background(), "1", "10", "1", diaryText)
	if !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, boom) {
		t.Fatalf("err = %v, want one wrapping ErrNotRebuilt and the cause", err)
	}
	if ss != nil {
		t.Fatalf("scores = %+v, want none on an error", ss)
	}
	cat.setErr(nil)
	if got := picked(r.scores(t, "1", diaryText)); !slices.Equal(got, []string{"japan"}) {
		t.Fatalf("retry: %v, want [japan]", got)
	}
	if tags, _ := cat.counts(); tags != 2 {
		t.Fatalf("Tags called %d times, want 2: the failure recorded no state", tags)
	}
}

func TestEngine_EmbedderErrorFailsAndRetries(t *testing.T) {
	boom := errors.New("embedder down")
	r := newRig(t, engineCfg, &fakeCatalog{tags: breadAndJapan("1")}, nil, nil)
	r.emb.failRole(langai.RoleDocument, boom)
	ss, err := r.e.scores(context.Background(), "1", "10", "1", diaryText)
	if !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, boom) {
		t.Fatalf("err = %v, want one wrapping ErrNotRebuilt and the cause", err)
	}
	if ss != nil {
		t.Fatalf("scores = %+v, want none on an error", ss)
	}
	r.emb.failRole(langai.RoleDocument, nil)
	items, names := r.emb.count(langai.RoleQuery), r.emb.count(langai.RoleDocument)
	if got := picked(r.scores(t, "1", diaryText)); !slices.Equal(got, []string{"japan"}) {
		t.Fatalf("retry: %v, want [japan]", got)
	}
	if n := r.emb.count(langai.RoleQuery) - items; n != 0 {
		t.Fatalf("retry embedded %d items, want 0: the item vector is cached", n)
	}
	if n := r.emb.count(langai.RoleDocument) - names; n != 2 {
		t.Fatalf("retry embedded %d names, want 2", n)
	}
}

// TestEngine_AMemberWithAPieceThatFailsIsSkippedWhole: one piece of a long
// member failing leaves the whole member out of its tag, since the rest of it
// is not the member, and loses none of the pieces that did embed, its own
// among them: once the embedder takes the piece, the next revision embeds
// that piece only.
func TestEngine_AMemberWithAPieceThatFailsIsSkippedWhole(t *testing.T) {
	const (
		short   = "Short bread"
		earlier = "A long pastry recipe that goes on past the chunk size"
		failing = "A long croissant recipe that boom goes on past the chunk size"
		item    = "bread at home"
	)
	cfg, rec := loggedCfg()
	cfg.ChunkBytes = 20
	cat := &fakeCatalog{tags: breadAndJapan("1"), members: map[string][]string{"bread": {short, earlier, failing}}}
	r := newRig(t, cfg, cat, nil, nil)
	r.emb.failContaining("boom", errBoom)
	got := r.scores(t, "1", item)

	pair := &fakeCatalog{tags: breadAndJapan("1"), members: map[string][]string{"bread": {short, earlier}}}
	if want := newRig(t, cfg, pair, nil, nil).scores(t, "1", item); !reflect.DeepEqual(got, want) {
		t.Errorf("scores %+v, want those of the tag without the failing member, %+v", got, want)
	}
	if n := len(rec.matching("WARN", "member", "tag=bread", "piece")); n != 1 {
		t.Errorf("the member was reported %d times, want once, naming the piece; log:\n%s", n, strings.Join(rec.lines(), "\n"))
	}
	r.emb.failContaining("", nil)
	embedded := r.emb.count(langai.RoleQuery)
	r.scores(t, "2", item)
	if n := r.emb.count(langai.RoleQuery) - embedded; n != 1 {
		t.Errorf("the next revision embedded %d pieces, want 1: only the piece that had failed", n)
	}
}

func TestEngine_RejectsBadInput(t *testing.T) {
	r := newRig(t, engineCfg, &fakeCatalog{tags: breadAndJapan("1")}, nil, nil)
	ctx := context.Background()
	for _, tc := range []struct{ name, index, rev, text string }{
		{"an empty index", "", "1", diaryText},
		{"an empty revision", "1", "", diaryText},
		{"a blank text", "1", "1", " \t\n"},
	} {
		ss, err := r.e.scores(ctx, tc.index, "10", tc.rev, tc.text)
		if err == nil || ss != nil {
			t.Errorf("scores with %s = %+v, %v; want no scores and an error", tc.name, ss, err)
		}
	}
	if err := r.e.warm(ctx, "", "10", "1", nil); err == nil {
		t.Error("warm with an empty index succeeded")
	}
	if err := r.e.warm(ctx, "1", "10", "", nil); err == nil {
		t.Error("warm with an empty revision succeeded")
	}
	if tags, members := r.cat.counts(); tags != 0 || members != 0 {
		t.Fatalf("%d Tags and %d Members calls, want none", tags, members)
	}
	if n := r.emb.total(); n != 0 {
		t.Fatalf("embedded %d texts, want none", n)
	}
}

func TestEngine_RejectsBadCatalogTags(t *testing.T) {
	for _, tc := range []struct {
		name  string
		tags  []CatalogTag
		named string
	}{
		{"an empty id", []CatalogTag{{"", "cooking / bread", "1"}}, "cooking / bread"},
		{"a duplicate id", []CatalogTag{{"bread", "cooking / bread", "1"}, {"bread", "baking", "1"}}, "bread"},
		{"a blank name", []CatalogTag{{"bread", " \t", "1"}}, "bread"},
		{"an empty revision", []CatalogTag{{"bread", "cooking / bread", ""}}, "bread"},
	} {
		r := newRig(t, engineCfg, &fakeCatalog{tags: tc.tags}, nil, nil)
		ss, err := r.e.scores(context.Background(), "1", "10", "1", diaryText)
		if !errors.Is(err, ErrNotRebuilt) || !strings.Contains(err.Error(), `"`+tc.named+`"`) {
			t.Errorf("%s: err = %v, want one wrapping ErrNotRebuilt naming %q", tc.name, err, tc.named)
		}
		if ss != nil {
			t.Errorf("%s: scores = %+v, want none", tc.name, ss)
		}
	}
}

func TestEngine_ToleratesOddMembers(t *testing.T) {
	const ghost = "a ghost member of a tag nobody asked for"
	cat := &fakeCatalog{
		tags: breadAndJapan("1"),
		members: map[string][]string{
			"bread": {croissantRecipes[0], "  ", croissantRecipes[0], croissantRecipes[1]},
			"japan": {"", " \t"},
		},
		extra: map[string][]string{"ghost": {ghost}},
	}
	r := newRig(t, engineCfg, cat, nil, nil)
	ss := r.scores(t, "1", croissantText)
	if n := r.emb.times(langai.RoleQuery, ghost); n != 0 {
		t.Errorf("a member of a tag not asked for was embedded %d times", n)
	}
	for _, blank := range []string{"", "  ", " \t"} {
		if n := r.emb.times(langai.RoleQuery, blank); n != 0 {
			t.Errorf("blank member %q was embedded %d times", blank, n)
		}
	}
	if n := r.emb.times(langai.RoleQuery, croissantRecipes[0]); n != 1 {
		t.Errorf("a member listed twice was embedded %d times, want once", n)
	}
	if got := scoreOf(t, ss, "japan").Cutoff; got != engineCfg.NameMinScore {
		t.Errorf("japan, with only blank members, has cutoff %v, want NameMinScore %v", got, engineCfg.NameMinScore)
	}
	if got := scoreOf(t, ss, "bread").Cutoff; got == engineCfg.NameMinScore {
		t.Errorf("bread, with members, has the name-only cutoff %v", got)
	}
}

func TestEngine_RefusesAMemberOfAnotherLength(t *testing.T) {
	r := newRig(t, engineCfg, breadWithRecipes("1"), nil, nil)
	r.emb.noDims = true
	r.emb.shortFor(croissantRecipes[1])
	ss, err := r.e.scores(context.Background(), "1", "10", "1", croissantText)
	if !errors.Is(err, ErrNotRebuilt) || !strings.Contains(err.Error(), `"bread"`) {
		t.Fatalf("err = %v, want one wrapping ErrNotRebuilt naming bread", err)
	}
	if ss != nil {
		t.Fatalf("scores = %+v, want none", ss)
	}
}

// TestEngine_RefusesABrokenEmbedder: an embedder that answers with too few
// vectors, one of another length than its Dimensions, or an empty one fails
// the call, which caches nothing.
func TestEngine_RefusesABrokenEmbedder(t *testing.T) {
	for _, tc := range []struct {
		name string
		emb  langai.Embedder
		want string
	}{
		{"no vectors", vectorlessEmbedder{}, "returned 0 vectors"},
		{"a short item", brokenEmbedder{dims: 64, query: []float32{1, 0, 0}}, "3 dimensions, want 64"},
		{"an empty item", brokenEmbedder{query: []float32{}}, "empty vector"},
	} {
		cache := NewMemCache()
		e, err := newEngine(tc.emb, &fakeCatalog{tags: breadAndJapan("1")}, cache, NewMemIndex(0), engineCfg)
		if err != nil {
			t.Fatal(err)
		}
		ss, err := e.scores(context.Background(), "1", "10", "1", diaryText)
		if err == nil || !strings.Contains(err.Error(), tc.want) || ss != nil {
			t.Errorf("%s: %+v, %v; want no scores and an error saying %q", tc.name, ss, err, tc.want)
		}
		got, err := cache.Get(context.Background(), "1", []CacheKey{KeyFor("fake", langai.RoleQuery, diaryText)})
		if err != nil || len(got) != 0 {
			t.Errorf("%s: the cache holds %v (%v), want nothing", tc.name, got, err)
		}
	}
}

// TestEngine_ConcurrentCallsRebuildOnce holds the one rebuild in the catalog
// until every caller is provably blocked. With none finished and one Tags call
// between the fifty of them, the others can only be waiting on it in the flight.
func TestEngine_ConcurrentCallsRebuildOnce(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cat := breadWithRecipes("1")
		cat.release = make(chan struct{})
		r := newRig(t, engineCfg, cat, nil, nil)
		const n = 50
		got := make([][]TagScore, n)
		errs := make([]error, n)
		var finished atomic.Int32
		var wg sync.WaitGroup
		for i := range n {
			wg.Go(func() {
				got[i], errs[i] = r.e.scores(context.Background(), "1", "10", "1", croissantText)
				finished.Add(1)
			})
		}
		synctest.Wait()
		// not Fatalf: the callers must get to finish for the bubble to end
		if tags, _ := cat.counts(); tags != 1 {
			t.Errorf("with every caller blocked, Tags was called %d times, want once", tags)
		}
		if k := finished.Load(); k != 0 {
			t.Errorf("%d callers finished before the rebuild did", k)
		}
		close(cat.release)
		wg.Wait()
		if tags, _ := cat.counts(); tags != 1 {
			t.Fatalf("Tags called %d times, want once", tags)
		}
		for i := range n {
			if errs[i] != nil {
				t.Fatalf("call %d: %v", i, errs[i])
			}
			if !reflect.DeepEqual(got[i], got[0]) {
				t.Fatalf("call %d scored %+v, call 0 %+v", i, got[i], got[0])
			}
		}
		if !slices.Equal(picked(got[0]), []string{"bread"}) {
			t.Fatalf("picked %v, want [bread]", picked(got[0]))
		}
	})
}

func TestEngine_CancelledLeaderDoesNotFailWaiters(t *testing.T) {
	cat := breadWithRecipes("1")
	cat.release, cat.entered = make(chan struct{}), make(chan struct{}, 1)
	r := newRig(t, engineCfg, cat, nil, nil)
	ctx, cancel := context.WithCancel(context.Background())
	leader := make(chan error, 1)
	go func() {
		_, err := r.e.scores(ctx, "1", "10", "1", croissantText)
		leader <- err
	}()
	<-cat.entered
	const n = 10
	errs := make([]error, n)
	got := make([][]TagScore, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Go(func() { got[i], errs[i] = r.e.scores(context.Background(), "1", "10", "1", croissantText) })
	}
	cancel()
	if err := <-leader; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled leader: err = %v, want context.Canceled", err)
	}
	close(cat.release)
	wg.Wait()
	for i := range n {
		if errs[i] != nil || got[i] == nil {
			t.Fatalf("waiter %d: %+v, %v; want scores", i, got[i], errs[i])
		}
	}
	if tags, _ := cat.counts(); tags != 1 {
		t.Fatalf("Tags called %d times, want once: the leader's cancel must not end the rebuild", tags)
	}
}

// evictOther fills another space of a tiny MemIndex, which evicts every other
// space. It may run on the flight's goroutine, so it does not stop the test.
func evictOther(t *testing.T, ix *MemIndex) {
	t.Helper()
	if err := ix.Apply(context.Background(), "1", "other", "", "s", []TagVector{{ID: "x", Vec: []float32{1}}}, nil); err != nil {
		t.Error(err)
	}
}

// TestEngine_SpaceEvictedMidRebuildIsNotRecordedCurrent: a space evicted
// between the engine's read of its state and its Apply comes back with only the
// tags that changed, so it must not be recorded current: the next call at that
// revision diffs again and scores every tag.
func TestEngine_SpaceEvictedMidRebuildIsNotRecordedCurrent(t *testing.T) {
	ix := NewMemIndex(1) // tiny budget: only the space just used is kept
	tags := append(breadAndJapan("1"), CatalogTag{"rust", "dev / rust", "1"})
	cat := &fakeCatalog{tags: tags}
	r := newRig(t, engineCfg, cat, nil, ix)
	if ss := r.scores(t, "1", diaryText); len(ss) != 3 {
		t.Fatalf("first call: %d scores, want 3", len(ss))
	}
	bumped := slices.Clone(tags)
	bumped[0].Revision = "2"
	cat.set(bumped, nil)
	cat.runAtMembers(func() { evictOther(t, ix) })
	r.scores(t, "2", diaryText)
	if ss := r.scores(t, "2", diaryText); len(ss) != 3 {
		t.Errorf("after an eviction mid-rebuild: %d scores at the same revision, want 3", len(ss))
	}
	tagCalls, _ := cat.counts()
	if ss := r.scores(t, "2", diaryText); len(ss) != 3 {
		t.Errorf("once recovered: %d scores, want 3", len(ss))
	}
	if n, _ := cat.counts(); n != tagCalls {
		t.Errorf("%d more Tags calls once recovered, want none: the state is recorded by now", n-tagCalls)
	}
}

// evictingIndex is a TagIndex that runs hook before Candidates: before the
// first call, or before every call with always.
type evictingIndex struct {
	TagIndex
	hook   func()
	always bool
	once   sync.Once
}

func (x *evictingIndex) Candidates(ctx context.Context, index, partition string, vec []float32) ([]TagVector, error) {
	if x.always {
		x.hook()
	} else {
		x.once.Do(x.hook)
	}
	return x.TagIndex.Candidates(ctx, index, partition, vec)
}

// TestEngine_SpaceEvictedBeforeCandidatesIsRebuiltOnce: a space evicted between
// the refresh and the Candidates would score nothing with no error, so the
// engine refreshes once more and reads the candidates again; it does not go on
// for ever when the eviction keeps happening.
func TestEngine_SpaceEvictedBeforeCandidatesIsRebuiltOnce(t *testing.T) {
	for _, tc := range []struct {
		name      string
		always    bool
		wantScore int
	}{{"once", false, 2}, {"every time", true, 0}} {
		t.Run(tc.name, func(t *testing.T) {
			mem := NewMemIndex(1)
			ix := &evictingIndex{TagIndex: mem, always: tc.always, hook: func() { evictOther(t, mem) }}
			cat := &fakeCatalog{tags: breadAndJapan("1")}
			r := newRig(t, engineCfg, cat, nil, ix)
			if ss := r.scores(t, "1", diaryText); len(ss) != tc.wantScore {
				t.Errorf("%d scores, want %d", len(ss), tc.wantScore)
			}
			if n, _ := cat.counts(); n != 2 {
				t.Errorf("%d Tags calls, want 2: the refresh and one more", n)
			}
		})
	}
}

// TestEngine_SpaceWithNoTagsIsNotRebuilt: no candidates from a space that is
// known is the right answer, not an eviction.
func TestEngine_SpaceWithNoTagsIsNotRebuilt(t *testing.T) {
	cat := &fakeCatalog{}
	r := newRig(t, engineCfg, cat, nil, nil)
	if ss := r.scores(t, "1", diaryText); len(ss) != 0 {
		t.Errorf("%d scores, want none", len(ss))
	}
	if n, _ := cat.counts(); n != 1 {
		t.Errorf("%d Tags calls, want 1", n)
	}
}

func TestEngine_SharedCacheColdStartEmbedsNothing(t *testing.T) {
	cache := NewMemCache()
	a := newRig(t, engineCfg, breadWithRecipes("1"), cache, nil)
	if err := a.e.warm(context.Background(), "1", "10", "1", []string{croissantText}); err != nil {
		t.Fatal(err)
	}
	b := newRig(t, engineCfg, breadWithRecipes("1"), cache, nil)
	got := b.scores(t, "1", croissantText)
	if n := b.emb.total(); n != 0 {
		t.Fatalf("engine B embedded %d texts, want 0", n)
	}
	if !slices.Equal(picked(got), []string{"bread"}) {
		t.Fatalf("engine B picked %v, want [bread]", picked(got))
	}
}

func TestEngine_ModelSwitchReembeds(t *testing.T) {
	cache := NewMemCache()
	newRig(t, engineCfg, breadWithRecipes("1"), cache, nil).scores(t, "1", croissantText)
	cfg := engineCfg
	cfg.Model = "fake-2"
	b := newRig(t, cfg, breadWithRecipes("1"), cache, nil)
	b.scores(t, "1", croissantText)
	for _, name := range []string{"cooking / bread", "travel / japan"} {
		if n := b.emb.times(langai.RoleDocument, name); n != 1 {
			t.Errorf("name %q embedded %d times under the new model, want once", name, n)
		}
	}
	for _, text := range append(slices.Clone(croissantRecipes), croissantText) {
		if n := b.emb.times(langai.RoleQuery, text); n != 1 {
			t.Errorf("text %q embedded %d times under the new model, want once", text, n)
		}
	}
}

func TestEngine_ChunkBytesChangeReembedsItemsOnly(t *testing.T) {
	cache := NewMemCache()
	newRig(t, engineCfg, breadWithRecipes("1"), cache, nil).scores(t, "1", croissantText)
	cfg := engineCfg
	cfg.ChunkBytes = 20
	b := newRig(t, cfg, breadWithRecipes("1"), cache, nil)
	b.scores(t, "1", croissantText)
	if n := b.emb.count(langai.RoleDocument); n != 0 {
		t.Errorf("embedded %d names, want 0: names are not chunked", n)
	}
	// every member and the item is longer than 20 bytes, so each is embedded
	// again, as its pieces
	var pieces []string
	for _, text := range append(slices.Clone(croissantRecipes), croissantText) {
		pieces = append(pieces, Chunk(text, cfg.ChunkBytes)...)
	}
	slices.Sort(pieces)
	pieces = slices.Compact(pieces)
	if n := b.emb.count(langai.RoleQuery); n != len(pieces) {
		t.Errorf("embedded %d item texts, want the %d distinct pieces of the item and the members", n, len(pieces))
	}
	for _, p := range pieces {
		if n := b.emb.times(langai.RoleQuery, p); n != 1 {
			t.Errorf("piece %q embedded %d times, want once", p, n)
		}
	}
}

func TestEngine_OtherSettingsOverOneIndexRebuild(t *testing.T) {
	cache, ix := NewMemCache(), NewMemIndex(0)
	cat := breadWithRecipes("1")
	newRig(t, engineCfg, cat, cache, ix).scores(t, "1", croissantText)
	tags, _ := cat.counts()
	cfg := engineCfg
	cfg.SmoothingK = 10
	b := newRig(t, cfg, cat, cache, ix)
	got := b.scores(t, "1", croissantText)
	if n, _ := cat.counts(); n != tags+1 {
		t.Fatalf("engine B made %d Tags calls, want 1", n-tags)
	}
	if n := b.emb.total(); n != 0 {
		t.Fatalf("engine B embedded %d texts, want 0", n)
	}
	want := newRig(t, cfg, breadWithRecipes("1"), nil, nil).scores(t, "1", croissantText)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("engine B scored %+v, want its own settings' %+v", got, want)
	}
}

func TestEngine_IndexesAreIsolated(t *testing.T) {
	r := newRig(t, engineCfg, &fakeCatalog{tags: breadAndJapan("1")}, nil, nil)
	r.scoresIn(t, "a", "1", diaryText)
	r.scoresIn(t, "b", "1", diaryText)
	if n := r.emb.times(langai.RoleQuery, diaryText); n != 2 {
		t.Fatalf("the text was embedded %d times across two indexes, want 2", n)
	}
}

func TestEngine_WarmFillsTheCacheAndTheIndex(t *testing.T) {
	r := newRig(t, engineCfg, breadWithRecipes("1"), nil, nil)
	if err := r.e.warm(context.Background(), "1", "10", "1", []string{guideText, diaryText, "  "}); err != nil {
		t.Fatal(err)
	}
	if n := r.emb.times(langai.RoleQuery, "  "); n != 0 {
		t.Errorf("warm embedded a blank text %d times", n)
	}
	tags, members := r.cat.counts()
	embedded := r.emb.total()
	if got := picked(r.scores(t, "1", diaryText)); !slices.Equal(got, []string{"japan"}) {
		t.Errorf("after warm: %v, want [japan]", got)
	}
	t2, m2 := r.cat.counts()
	if n := r.emb.total() - embedded; n != 0 || t2 != tags || m2 != members {
		t.Fatalf("after warm: %d texts embedded, %d Tags and %d Members calls, want none", n, t2-tags, m2-members)
	}
}

func TestEngine_RolesAreQueriesAndDocuments(t *testing.T) {
	r := newRig(t, engineCfg, breadWithRecipes("1"), nil, nil)
	r.scores(t, "1", croissantText)
	for _, name := range []string{"cooking / bread", "travel / japan"} {
		if r.emb.times(langai.RoleDocument, name) != 1 || r.emb.times(langai.RoleQuery, name) != 0 {
			t.Errorf("name %q: want it embedded once, as a document only", name)
		}
	}
	for _, text := range append(slices.Clone(croissantRecipes), croissantText) {
		if r.emb.times(langai.RoleQuery, text) != 1 || r.emb.times(langai.RoleDocument, text) != 0 {
			t.Errorf("text %q: want it embedded once, as a query only", text)
		}
	}
}

// TestEngine_OnePathUnlessAllPathLevels: "dev" and "dev / go" both clear their
// cutoffs, "dev / go" by more, so only it is picked and "dev" is shadowed by
// it; with AllPathLevels both are picked.
func TestEngine_OnePathUnlessAllPathLevels(t *testing.T) {
	emb := stubEmbedder{dims: 3, vecs: map[string][]float32{
		"dev":      unitVec(0.60, 0.80, 0),
		"dev / go": unitVec(0.70, 0, math.Sqrt(1-0.70*0.70)),
		"query":    unitVec(1, 0, 0),
	}}
	for _, tc := range []struct {
		all  bool
		want []string
	}{{false, []string{"go"}}, {true, []string{"go", "dev"}}} {
		cat := &fakeCatalog{tags: []CatalogTag{{"dev", "dev", "1"}, {"go", "dev / go", "1"}}}
		e, err := newEngine(emb, cat, NewMemCache(), NewMemIndex(0), TaggerConfig{Model: "stub", AllPathLevels: tc.all})
		if err != nil {
			t.Fatal(err)
		}
		scores, err := e.scores(context.Background(), "1", "10", "1", "query")
		if err != nil {
			t.Fatal(err)
		}
		if got := picked(scores); !slices.Equal(got, tc.want) {
			t.Errorf("AllPathLevels %v: picked %v, want %v", tc.all, got, tc.want)
		}
		wantShadow := "go"
		if tc.all {
			wantShadow = ""
		}
		for _, s := range scores {
			if s.ID == "dev" && (s.ShadowedBy != wantShadow || s.Picked() == !tc.all) {
				t.Errorf("AllPathLevels %v: Scores has dev %+v, want ShadowedBy %q", tc.all, s, wantShadow)
			}
		}
	}
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
				_ = f.do(context.Background(), "k", fn)
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
		// another key runs on its own, while "k" is in flight
		if err := f.do(context.Background(), "other", func(context.Context) error { others.Add(1); return nil }); err != nil {
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
	boom := errors.New("boom")
	if err := f.do(context.Background(), "k", func(context.Context) error { return boom }); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
	ran := false
	if err := f.do(context.Background(), "k", func(context.Context) error { ran = true; return nil }); err != nil || !ran {
		t.Fatalf("second do: ran %v, err %v; want a fresh run", ran, err)
	}
}

func TestFlight_CallerLeavesOnItsOwnCancelAndFnCarriesOn(t *testing.T) {
	var f flight
	release, started, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	ctx, cancel := context.WithCancel(context.Background())
	leader := make(chan error, 1)
	go func() {
		leader <- f.do(ctx, "k", func(ctx context.Context) error {
			close(started)
			<-release
			done <- ctx.Err()
			return nil
		})
	}()
	<-started
	waiter := make(chan error, 1)
	// the waiter may arrive after the flight ended and run fn of its own
	go func() { waiter <- f.do(context.Background(), "k", func(context.Context) error { return nil }) }()
	cancel()
	if err := <-leader; !errors.Is(err, context.Canceled) {
		t.Fatalf("leader: err = %v, want context.Canceled", err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatalf("fn saw its ctx end (%v): it must outlive its caller's", err)
	}
	if err := <-waiter; err != nil {
		t.Fatalf("waiter: %v", err)
	}
}

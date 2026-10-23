package autotag

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/semsearch"
)

// mustNew returns a Tagger, failing the test if New does.
func mustNew(t *testing.T, emb semsearch.Embedder, store Store, cfg Config) *Tagger {
	t.Helper()
	tg, err := New(emb, store, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return tg
}

// must fails the test on err.
func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// suggest returns the tags the scope suggests for the item, failing the test
// on an error.
func suggest(t *testing.T, space *Scope, itemID string) []string {
	t.Helper()
	ids, err := space.Suggest(context.Background(), itemID)
	if err != nil {
		t.Fatal(err)
	}
	return ids
}

func TestNew_RequiresEmbedderStoreAndModel(t *testing.T) {
	if _, err := New(nil, NewMemStore(), Config{Model: "m"}); err == nil {
		t.Error("nil embedder: New must fail")
	}
	if _, err := New(semsearch.FakeEmbedder{}, nil, Config{Model: "m"}); err == nil {
		t.Error("nil store: New must fail")
	}
	if _, err := New(semsearch.FakeEmbedder{}, NewMemStore(), Config{}); err == nil {
		t.Error("no model: New must fail")
	}
}

func TestNew_Defaults(t *testing.T) {
	tg := mustNew(t, semsearch.FakeEmbedder{}, NewMemStore(), Config{Model: "m"})
	if tg.cfg.NameMinScore != 0.46 {
		t.Fatalf("NameMinScore = %v, want default 0.46", tg.cfg.NameMinScore)
	}
	if tg.cfg.MemberMinScore != 0.70 {
		t.Fatalf("MemberMinScore = %v, want default 0.70", tg.cfg.MemberMinScore)
	}
	if tg.cfg.SmoothingK != 3 {
		t.Fatalf("SmoothingK = %v, want default 3", tg.cfg.SmoothingK)
	}
}

// TestSetTagAndSetItem_RejectEmptyIDsAndBlankText: a tag or an item needs an
// id, and text to embed. A write without them fails and stores nothing.
func TestSetTagAndSetItem_RejectEmptyIDsAndBlankText(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"SetTag with an empty id", space.SetTag(ctx, "", "rust")},
		{"SetTag with a blank name", space.SetTag(ctx, "1", " \t\n")},
		{"SetItem with an empty id", space.SetItem(ctx, "", "rust")},
		{"SetItem with blank text", space.SetItem(ctx, "a", "  ")},
	} {
		if tc.err == nil {
			t.Errorf("%s succeeded, want an error", tc.name)
		}
	}
	tags, _ := store.Tags(ctx, "1", "10")
	_, foundEmpty, _ := store.Item(ctx, "1", "10", "")
	_, foundA, _ := store.Item(ctx, "1", "10", "a")
	if len(tags) != 0 || foundEmpty || foundA {
		t.Fatalf("a refused write stored something: tags %+v, item \"\" found %v, item a found %v", tags, foundEmpty, foundA)
	}
}

// rustCooking is an item text that shares several words with tag "rust" and
// one with tag "cooking": under FakeEmbedder it scores ~0.61 against "rust"
// and ~0.20 against "cooking", so a NameMinScore of 0.15 clears both.
const rustCooking = "The Rust Programming Language\nA systems programming language focused on safety and speed\nrust-lang.org\n" +
	"Rust is a modern systems programming language. Rust emphasizes memory safety " +
	"without garbage collection. Learn Rust by building real projects in Rust. " +
	"I also enjoy cooking on weekends."

// TestSuggest_ColdStartTagsByName: tags without members are suggested by their
// names alone, every one whose score clears the name minimum, best first.
func TestSuggest_ColdStartTagsByName(t *testing.T) {
	ctx := context.Background()
	space := mustNew(t, semsearch.FakeEmbedder{}, NewMemStore(), Config{Model: "fake", NameMinScore: 0.15}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetTag(ctx, "2", "cooking"))
	must(t, space.SetItem(ctx, "doc", rustCooking))
	if got := suggest(t, space, "doc"); !slices.Equal(got, []string{"1", "2"}) {
		t.Fatalf("Suggest = %v, want [1 2]: rust first, it scores higher", got)
	}
}

// countingEmbedder wraps an Embedder, counts Embed calls (so tests can assert
// a cache actually avoided re-embedding) and records each call's role (so
// tests can assert what the Tagger embeds as a query and as a document).
type countingEmbedder struct {
	inner semsearch.Embedder
	calls int
	roles []semsearch.Role
}

func (c *countingEmbedder) Dimensions() int { return c.inner.Dimensions() }

func (c *countingEmbedder) Embed(ctx context.Context, role semsearch.Role, texts []string) ([][]float32, error) {
	c.calls++
	c.roles = append(c.roles, role)
	return c.inner.Embed(ctx, role, texts)
}

// TestEmbedRoles_ItemsAreQueriesAndNamesAreDocuments: an item is scored
// against the tag vectors, so its text is embedded as a query, and a tag's
// name as a document. An asymmetric model prefixes the two differently.
// Suggest finds both embedded already, and embeds nothing more.
func TestEmbedRoles_ItemsAreQueriesAndNamesAreDocuments(t *testing.T) {
	ctx := context.Background()
	emb := &countingEmbedder{inner: semsearch.FakeEmbedder{}}
	space := mustNew(t, emb, NewMemStore(), Config{Model: "fake", NameMinScore: 0.15}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetTag(ctx, "2", "cooking"))
	must(t, space.SetItem(ctx, "doc", rustCooking))
	suggest(t, space, "doc")
	want := []semsearch.Role{semsearch.RoleDocument, semsearch.RoleDocument, semsearch.RoleQuery}
	if !slices.Equal(emb.roles, want) {
		t.Fatalf("embed roles = %v, want %v (one document per tag name, then the item as a query)", emb.roles, want)
	}
	if emb.calls != 3 {
		t.Fatalf("embedded %d times, want 3: Suggest must embed nothing more", emb.calls)
	}
}

// stubEmbedder maps texts to fixed vectors. Unknown text is an error, so
// nothing is scored by accident.
type stubEmbedder struct {
	dims int
	vecs map[string][]float32
}

func (s stubEmbedder) Dimensions() int { return s.dims }

func (s stubEmbedder) Embed(_ context.Context, _ semsearch.Role, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		v, ok := s.vecs[text]
		if !ok {
			return nil, fmt.Errorf("stubEmbedder: no vector for %q", text)
		}
		out[i] = v
	}
	return out, nil
}

// taughtAndPlain is the embedder of the per-tag cutoff tests (see
// suggestTaughtAndPlain). The text "query" embeds as q and "member" as m. The
// name "taught" scores 0.55 against q and m scores 0.60, so a tag "taught"
// with three members of text "member" blends to a score of 0.705 against q.
// The name "plain" scores 0.50.
func taughtAndPlain() stubEmbedder {
	return stubEmbedder{dims: 4, vecs: map[string][]float32{
		"taught": unitVec(0.55, math.Sqrt(1-0.55*0.55), 0, 0),
		"plain":  unitVec(0.50, 0, 0, math.Sqrt(0.75)),
		"member": unitVec(0.60, 0, 0.80, 0),
		"query":  unitVec(1, 0, 0, 0),
	}}
}

// suggestTaughtAndPlain returns the tags suggested for the item "q" (text
// "query") in a scope of two tags: "1", named "taught", whose members are the
// items m1, m2 and m3 (text "member"), and "2", named "plain", with no members.
func suggestTaughtAndPlain(t *testing.T, cfg Config) []string {
	t.Helper()
	ctx := context.Background()
	space := mustNew(t, taughtAndPlain(), NewMemStore(), cfg).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "taught"))
	must(t, space.SetTag(ctx, "2", "plain"))
	for _, id := range []string{"m1", "m2", "m3"} {
		must(t, space.SetItem(ctx, id, "member"))
		must(t, space.TagByHand(ctx, id, "1"))
	}
	must(t, space.SetItem(ctx, "q", "query"))
	return suggest(t, space, "q")
}

// With the default minimums "taught" blends to 0.705, which clears a global
// 0.46 but not its own cutoff (0.711), since its name scores 0.55 against 0.46
// and its members 0.60 against 0.70. "plain" scores 0.50 against 0.46.
// "taught" comes first, so an implementation that stops at the first refused
// tag never reaches "plain".
func TestSuggest_JudgesEachTagAgainstItsOwnCutoff(t *testing.T) {
	if got := suggestTaughtAndPlain(t, Config{Model: "stub"}); !slices.Equal(got, []string{"2"}) {
		t.Fatalf("Suggest = %v, want [plain] only", got)
	}
}

// MemberMinScore reaches the Tagger, and moves the cutoff of a tag with
// members. The default of 0.70 hides it: a Tagger that used 0.70 whatever it
// was given would pass every other test. Lowered to 0.5, the cutoff of "taught"
// (3 members, so a blend weight of 0.5) drops from 0.711 to about 0.589, below
// its score of 0.705, so it is suggested along with "plain", and first.
func TestSuggest_MemberMinScoreSetsTheCutoffOfATaughtTag(t *testing.T) {
	if got := suggestTaughtAndPlain(t, Config{Model: "stub", MemberMinScore: 0.5}); !slices.Equal(got, []string{"1", "2"}) {
		t.Fatalf("Suggest = %v, want [taught plain]", got)
	}
}

func TestSuggest_AssignsEveryTagThatClearsItsCutoff(t *testing.T) {
	const n = 7 // more than any cap a tagger might be tempted to have
	ctx := context.Background()
	q := make([]float64, n+1)
	q[0] = 1
	vecs := map[string][]float32{"query": unitVec(q...)}
	var want []string
	for i := 1; i <= n; i++ {
		v := make([]float64, n+1)
		v[0], v[i] = 0.5, math.Sqrt(0.75) // each name scores 0.50
		vecs[fmt.Sprintf("tag-%d", i)] = unitVec(v...)
		want = append(want, fmt.Sprint(i))
	}
	space := mustNew(t, stubEmbedder{dims: n + 1, vecs: vecs}, NewMemStore(), Config{Model: "stub"}).Scope("1", "10")
	for i := 1; i <= n; i++ {
		must(t, space.SetTag(ctx, fmt.Sprint(i), fmt.Sprintf("tag-%d", i)))
	}
	must(t, space.SetItem(ctx, "q", "query"))
	got := suggest(t, space, "q")
	slices.Sort(got)
	if !slices.Equal(got, want) {
		t.Fatalf("Suggest = %v, want all seven", got)
	}
}

// TestSuggest_NoTagsReturnsNothing: a scope without tags has nothing to
// suggest, and that is no error.
func TestSuggest_NoTagsReturnsNothing(t *testing.T) {
	ctx := context.Background()
	space := mustNew(t, semsearch.FakeEmbedder{}, NewMemStore(), Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetItem(ctx, "doc", rustCooking))
	got, err := space.Suggest(ctx, "doc")
	if err != nil || got != nil {
		t.Fatalf("Suggest = %#v, %v; want no ids and no error", got, err)
	}
}

func TestSuggest_UnknownItem(t *testing.T) {
	ctx := context.Background()
	space := mustNew(t, semsearch.FakeEmbedder{}, NewMemStore(), Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	if _, err := space.Suggest(ctx, "missing"); !errors.Is(err, ErrUnknownItem) {
		t.Fatalf("Suggest of an item never set: err = %v, want ErrUnknownItem", err)
	}
}

// A cutoff that is not a number is reached by no score. A caller's
// configuration should refuse one, but a Tagger built any other way must not
// answer it by suggesting every tag: with a NaN NameMinScore neither "alpha",
// which scores 0.80 against the item, nor "beta", which scores below zero, is
// suggested.
func TestSuggest_NaNNameMinScoreAssignsNoTag(t *testing.T) {
	const dims = 4
	ctx := context.Background()
	stub := stubEmbedder{dims: dims, vecs: map[string][]float32{
		"alpha": scoreVec(dims, 1, 0.80),
		"beta":  scoreVec(dims, 2, -0.30),
		"query": unitVec(1, 0, 0, 0),
	}}
	space := mustNew(t, stub, NewMemStore(), Config{Model: "stub", NameMinScore: math.NaN()}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "alpha"))
	must(t, space.SetTag(ctx, "2", "beta"))
	must(t, space.SetItem(ctx, "q", "query"))
	if got := suggest(t, space, "q"); len(got) != 0 {
		t.Fatalf("Suggest = %v, want none: no score reaches a NaN cutoff", got)
	}
}

func TestTagByHand_UnknownItemOrTag(t *testing.T) {
	ctx := context.Background()
	space := mustNew(t, semsearch.FakeEmbedder{}, NewMemStore(), Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetItem(ctx, "a", "rust ownership"))
	for _, tc := range []struct {
		itemID, tagID string
		want          error
	}{
		{"missing", "1", ErrUnknownItem},
		{"a", "missing", ErrUnknownTag},
	} {
		if err := space.TagByHand(ctx, tc.itemID, tc.tagID); !errors.Is(err, tc.want) {
			t.Errorf("TagByHand(%q, %q) = %v, want %v", tc.itemID, tc.tagID, err, tc.want)
		}
	}
}

// TestTagByHand_TeachesTheTagAndUntagForgets: items tagged by hand teach a tag
// their words, so it fits an item its name alone did not, and untagging them
// takes the lesson back.
func TestTagByHand_TeachesTheTagAndUntagForgets(t *testing.T) {
	ctx := context.Background()
	space := mustNew(t, semsearch.FakeEmbedder{}, NewMemStore(),
		Config{Model: "fake", NameMinScore: 0.3, MemberMinScore: 0.5}).Scope("1", "10")
	must(t, space.SetTag(ctx, "bread", "cooking / bread"))
	must(t, space.SetTag(ctx, "japan", "travel / japan"))
	must(t, space.SetItem(ctx, "doc", "Croissant dough with cold butter"))
	if got := suggest(t, space, "doc"); len(got) != 0 {
		t.Fatalf("before any tag by hand, Suggest = %v, want none", got)
	}
	recipes := []string{"recipe-1", "recipe-2", "recipe-3"}
	for i, text := range []string{
		"Croissant dough: folding cold butter into layers",
		"Butter croissant dough, step by step",
		"Shaping croissant dough with butter",
	} {
		must(t, space.SetItem(ctx, recipes[i], text))
		must(t, space.TagByHand(ctx, recipes[i], "bread"))
	}
	if got := suggest(t, space, "doc"); !slices.Equal(got, []string{"bread"}) {
		t.Fatalf("with three recipes tagged by hand, Suggest = %v, want [bread]", got)
	}
	for _, id := range recipes {
		must(t, space.UntagByHand(ctx, id, "bread"))
	}
	if got := suggest(t, space, "doc"); len(got) != 0 {
		t.Fatalf("with the recipes untagged, Suggest = %v, want none", got)
	}
}

// TestSetItem_UnchangedTextEmbedsNothing: setting an item to the text it has
// keeps the embedding it has.
func TestSetItem_UnchangedTextEmbedsNothing(t *testing.T) {
	ctx := context.Background()
	emb := &countingEmbedder{inner: semsearch.FakeEmbedder{}}
	space := mustNew(t, emb, NewMemStore(), Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetItem(ctx, "a", rustCooking))
	must(t, space.SetItem(ctx, "a", rustCooking))
	if emb.calls != 1 {
		t.Fatalf("embedded %d times, want 1", emb.calls)
	}
}

// TestSetItem_EditRebuildsItsHandTags: an item's new text moves the vectors
// of the tags it is tagged with by hand, before SetItem returns.
func TestSetItem_EditRebuildsItsHandTags(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetItem(ctx, "a", "rust ownership"))
	must(t, space.TagByHand(ctx, "a", "1"))
	before, _, _ := store.Tag(ctx, "1", "10", "1")
	must(t, space.SetItem(ctx, "a", "cooking bread at home"))
	after, _, _ := store.Tag(ctx, "1", "10", "1")
	if after.Built != after.Version {
		t.Fatalf("the tag was left stale: Built %d, Version %d", after.Built, after.Version)
	}
	if slices.Equal(after.Vec, before.Vec) {
		t.Fatal("the tag's vector did not move with its member's new text")
	}
}

// TestSetTag_RenameReembedsTheName: a tag's name is embedded when the tag is
// created and when it is renamed, never when it is set to the name it has,
// and a rename changes which items the tag fits.
func TestSetTag_RenameReembedsTheName(t *testing.T) {
	ctx := context.Background()
	emb := &countingEmbedder{inner: semsearch.FakeEmbedder{}}
	space := mustNew(t, emb, NewMemStore(), Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetItem(ctx, "doc", "go"))
	setTag := func(name string, wantCalls int) {
		t.Helper()
		before := emb.calls
		must(t, space.SetTag(ctx, "1", name))
		if got := emb.calls - before; got != wantCalls {
			t.Fatalf("SetTag(%q) embedded %d times, want %d", name, got, wantCalls)
		}
	}
	setTag("rust", 1)
	setTag("rust", 0)
	if got := suggest(t, space, "doc"); len(got) != 0 {
		t.Fatalf("before the rename, Suggest = %v, want none", got)
	}
	setTag("go", 1)
	if got := suggest(t, space, "doc"); !slices.Equal(got, []string{"1"}) {
		t.Fatalf("after the rename, Suggest = %v, want [1]", got)
	}
}

// TestDeleteItem_RebuildsItsTags: a deleted member leaves its tags' vectors
// before DeleteItem returns.
func TestDeleteItem_RebuildsItsTags(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetItem(ctx, "a", "rust ownership and borrowing"))
	must(t, space.SetItem(ctx, "b", "cargo crates for rust programs"))
	must(t, space.TagByHand(ctx, "a", "1"))
	must(t, space.TagByHand(ctx, "b", "1"))
	must(t, space.DeleteItem(ctx, "a"))
	rec, _, _ := store.Tag(ctx, "1", "10", "1")
	ms, _ := store.Members(ctx, "1", "10", "1")
	if len(ms) != 1 || ms[0].ItemID != "b" {
		t.Fatalf("members = %+v, want b alone", ms)
	}
	if rec.Built != rec.Version {
		t.Fatalf("the tag was left stale: Built %d, Version %d", rec.Built, rec.Version)
	}
	if !slices.Equal(rec.Vec, blend(rec.NameVec, [][]float32{ms[0].Vec}, 3)) {
		t.Fatal("the tag's vector is not the blend of its name and its one member left")
	}
}

// TestDeleteTag_DropsItAndForgetsItsLinks: a deleted tag is suggested no
// more, and a tag created later under its id starts without its members.
func TestDeleteTag_DropsItAndForgetsItsLinks(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetTag(ctx, "2", "programming"))
	must(t, space.SetItem(ctx, "doc", "rust programming"))
	must(t, space.TagByHand(ctx, "doc", "1"))
	if got := suggest(t, space, "doc"); !slices.Contains(got, "1") {
		t.Fatalf("before the delete, Suggest = %v, want 1 among them", got)
	}
	must(t, space.DeleteTag(ctx, "1"))
	if got := suggest(t, space, "doc"); !slices.Equal(got, []string{"2"}) {
		t.Fatalf("after the delete, Suggest = %v, want [2]", got)
	}
	must(t, space.SetTag(ctx, "1", "rust"))
	if ms, _ := store.Members(ctx, "1", "10", "1"); len(ms) != 0 {
		t.Fatalf("the tag created again under the deleted one's id has members %+v, want none", ms)
	}
}

// TestScopes_AreIsolated: the same ids in two partitions of one index name
// different tags and different items.
func TestScopes_AreIsolated(t *testing.T) {
	ctx := context.Background()
	tg := mustNew(t, semsearch.FakeEmbedder{}, NewMemStore(), Config{Model: "fake"})
	rust, cooking := tg.Scope("1", "10"), tg.Scope("1", "11")
	must(t, rust.SetTag(ctx, "1", "rust"))
	must(t, cooking.SetTag(ctx, "1", "cooking"))
	for _, space := range []*Scope{rust, cooking} {
		must(t, space.SetItem(ctx, "doc", "rust programming language"))
	}
	if got := suggest(t, rust, "doc"); !slices.Equal(got, []string{"1"}) {
		t.Errorf("in scope (1, 10), Suggest = %v, want [1]", got)
	}
	if got := suggest(t, cooking, "doc"); len(got) != 0 {
		t.Errorf("in scope (1, 11), Suggest = %v, want none", got)
	}
}

// vectorlessEmbedder breaks the Embedder contract on purpose: every call
// succeeds without returning a single vector.
type vectorlessEmbedder struct{}

func (vectorlessEmbedder) Dimensions() int { return 64 }

func (vectorlessEmbedder) Embed(context.Context, semsearch.Role, []string) ([][]float32, error) {
	return nil, nil
}

// TestEmbedderErrors_FailTheWrite: an embedding that succeeds without a
// vector fails the write that needed it, and does not panic the caller.
func TestEmbedderErrors_FailTheWrite(t *testing.T) {
	ctx := context.Background()
	space := mustNew(t, vectorlessEmbedder{}, NewMemStore(), Config{Model: "fake"}).Scope("1", "10")
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"SetItem", space.SetItem(ctx, "a", "some text")},
		{"SetTag", space.SetTag(ctx, "1", "rust")},
	} {
		if tc.err == nil || !strings.Contains(tc.err.Error(), "returned 0 vectors") {
			t.Errorf("%s = %v, want the error of an embedder that returned 0 vectors", tc.name, tc.err)
		}
	}
}

// racingStore links item "b" to tag "1" the first time a rebuild of "1" tries
// to store its vectors once raced is false, so that rebuild was computed from
// an older Version.
type racingStore struct {
	*MemStore
	raced bool
}

func (r *racingStore) SetTagVectors(ctx context.Context, i, p, id string, v int64, tv TagVectors) (bool, error) {
	if !r.raced && id == "1" {
		r.raced = true
		if _, err := r.Link(ctx, i, p, "b", "1"); err != nil {
			return false, err
		}
	}
	return r.MemStore.SetTagVectors(ctx, i, p, id, v, tv)
}

// TestRebuild_KeepsAChangeThatLandsMidRebuild: a hand tag that lands while its
// tag is being rebuilt is not overwritten by that rebuild, which never saw it,
// and the next Suggest rebuilds the tag with it.
func TestRebuild_KeepsAChangeThatLandsMidRebuild(t *testing.T) {
	ctx := context.Background()
	store := &racingStore{MemStore: NewMemStore(), raced: true}
	space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetItem(ctx, "a", "rust ownership and borrowing"))
	must(t, space.SetItem(ctx, "b", "cargo crates for rust programs"))
	store.raced = false

	if err := space.TagByHand(ctx, "a", "1"); err != nil {
		t.Fatal(err)
	}
	rec, _, _ := store.Tag(ctx, "1", "10", "1")
	if rec.Built == rec.Version {
		t.Fatal("the rebuild that missed b must not have been stored")
	}
	if _, err := space.Suggest(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	rec, _, _ = store.Tag(ctx, "1", "10", "1")
	ms, _ := store.Members(ctx, "1", "10", "1")
	if rec.Built != rec.Version || len(ms) != 2 ||
		!slices.Equal(rec.Vec, blend(rec.NameVec, [][]float32{ms[0].Vec, ms[1].Vec}, 3)) {
		t.Fatalf("Suggest must rebuild the tag from both members: %+v", rec)
	}
}

// linkOnReadStore links item "b" to tag "1" just before the first read of tag
// "1" once armed, so the link lands between the two reads of a rebuild.
type linkOnReadStore struct {
	*MemStore
	armed bool
}

func (r *linkOnReadStore) Tag(ctx context.Context, i, p, id string) (TagRecord, bool, error) {
	if r.armed && id == "1" {
		r.armed = false
		if _, err := r.Link(ctx, i, p, "b", "1"); err != nil {
			return TagRecord{}, false, err
		}
	}
	return r.MemStore.Tag(ctx, i, p, id)
}

// TestRebuild_ReadsTheMembersAfterTheTag: a rebuild reads the tag before its
// members. A change that lands before the members are read is then among
// them; one that lands after bumps the tag past the Version the rebuild read,
// and SetTagVectors refuses its vectors. A rebuild that read the members first
// would store vectors missing b at the Version b's link set.
func TestRebuild_ReadsTheMembersAfterTheTag(t *testing.T) {
	ctx := context.Background()
	store := &linkOnReadStore{MemStore: NewMemStore()}
	space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetItem(ctx, "a", "rust ownership and borrowing"))
	must(t, space.SetItem(ctx, "b", "cargo crates for rust programs"))
	store.armed = true

	must(t, space.TagByHand(ctx, "a", "1"))
	rec, _, _ := store.Tag(ctx, "1", "10", "1")
	ms, _ := store.Members(ctx, "1", "10", "1")
	if rec.Built != rec.Version || len(ms) != 2 ||
		!slices.Equal(rec.Vec, blend(rec.NameVec, [][]float32{ms[0].Vec, ms[1].Vec}, 3)) {
		t.Fatalf("TagByHand must build the tag from both members: %+v", rec)
	}
}

// TestModelSwitch_ReembedsEverythingAndMixesNothing: a Tagger of another model
// over a filled store re-embeds what it needs from the texts the store keeps,
// each one once, and builds every tag from vectors of its own model alone.
// FakeEmbedder stands in for both models, so the suggestions stay the same.
func TestModelSwitch_ReembedsEverythingAndMixesNothing(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	before := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "a"}).Scope("1", "10")
	must(t, before.SetTag(ctx, "t1", "programming / rust"))
	must(t, before.SetTag(ctx, "t2", "programming / go"))
	must(t, before.SetItem(ctx, "doc", "the rust programming language"))
	must(t, before.SetItem(ctx, "x", "rust and go programming"))
	must(t, before.SetItem(ctx, "y", "rust ownership and borrowing"))
	must(t, before.TagByHand(ctx, "x", "t1"))
	must(t, before.TagByHand(ctx, "x", "t2"))
	must(t, before.TagByHand(ctx, "y", "t1"))
	want := suggest(t, before, "doc")
	if len(want) == 0 {
		t.Fatal("the fixture suggests no tag for doc under model a: there is nothing to compare")
	}

	emb := &countingEmbedder{inner: semsearch.FakeEmbedder{}}
	space := mustNew(t, emb, store, Config{Model: "b"}).Scope("1", "10")
	if got := suggest(t, space, "doc"); !slices.Equal(got, want) {
		t.Fatalf("under model b, Suggest = %v, want %v as under model a", got, want)
	}
	tags, _ := store.Tags(ctx, "1", "10")
	if len(tags) != 2 {
		t.Fatalf("tags = %+v, want t1 and t2", tags)
	}
	for _, tag := range tags {
		if tag.Model != "b" || tag.Built != tag.Version {
			t.Errorf("tag %s: Model %q, Built %d, Version %d; want model b, built at its Version", tag.ID, tag.Model, tag.Built, tag.Version)
		}
	}
	for _, id := range []string{"doc", "x", "y"} {
		if item, _, _ := store.Item(ctx, "1", "10", id); item.Model != "b" {
			t.Errorf("item %s: Model %q, want b", id, item.Model)
		}
	}
	if emb.calls != 5 {
		t.Fatalf("embedded %d times, want 5: doc, x, y and the two names, each once", emb.calls)
	}
	must(t, space.Rebuild(ctx))
	if emb.calls != 5 {
		t.Fatalf("Rebuild embedded %d more times, want none: everything is of model b already", emb.calls-5)
	}
}

// errStoreDown is what failingStore answers with.
var errStoreDown = errors.New("store is down")

// failingStore is a MemStore whose SetTagVectors fails while fail is set, as a
// backend's would while its database cannot be reached.
type failingStore struct {
	*MemStore
	fail bool
}

func (f *failingStore) SetTagVectors(ctx context.Context, i, p, id string, v int64, tv TagVectors) (bool, error) {
	if f.fail {
		return false, errStoreDown
	}
	return f.MemStore.SetTagVectors(ctx, i, p, id, v, tv)
}

// TestRebuildFailure_KeepsTheChangeAndHealsOnSuggest: a write whose rebuild
// fails stays written and returns the rebuild's error, naming the tag, and the
// next Suggest rebuilds the tag it left stale.
func TestRebuildFailure_KeepsTheChangeAndHealsOnSuggest(t *testing.T) {
	ctx := context.Background()
	store := &failingStore{MemStore: NewMemStore()}
	space := mustNew(t, semsearch.FakeEmbedder{}, store,
		Config{Model: "fake", NameMinScore: 0.3, MemberMinScore: 0.5}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "cooking / bread"))
	must(t, space.SetItem(ctx, "a", "Baking bread at home: a cooking guide"))

	store.fail = true
	err := space.TagByHand(ctx, "a", "1")
	if !errors.Is(err, errStoreDown) {
		t.Fatalf("TagByHand = %v, want the store's error", err)
	}
	if want := `autotag: rebuild tag "1": `; !strings.HasPrefix(err.Error(), want) {
		t.Errorf("TagByHand = %q, want it to start with %q", err, want)
	}
	ms, _ := store.Members(ctx, "1", "10", "1")
	if len(ms) != 1 || ms[0].ItemID != "a" {
		t.Fatalf("members = %+v, want a: the link stays although its rebuild failed", ms)
	}
	if rec, _, _ := store.Tag(ctx, "1", "10", "1"); rec.Built == rec.Version {
		t.Fatal("the tag is built at its Version although its rebuild failed")
	}

	store.fail = false
	got := suggest(t, space, "a")
	if rec, _, _ := store.Tag(ctx, "1", "10", "1"); rec.Built != rec.Version {
		t.Fatalf("Suggest left the tag stale: Built %d, Version %d", rec.Built, rec.Version)
	}
	if !slices.Contains(got, "1") {
		t.Fatalf("Suggest = %v, want 1 among them", got)
	}
}

package autotag

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
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

// mustTag returns the tag the store holds in the scope ("1", "10"), failing
// the test on an error or a tag that is not there: the zero record of a
// missing tag would pass a check such as Built == Version.
func mustTag(t *testing.T, store Store, id string) TagRecord {
	t.Helper()
	rec, ok, err := store.Tag(context.Background(), "1", "10", id)
	if err != nil {
		t.Fatalf("Tag(%q): %v", id, err)
	}
	if !ok {
		t.Fatalf("Tag(%q) found nothing", id)
	}
	return rec
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
// fillTaughtAndPlain). The text "query" embeds as q and "member" as m. The
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

// fillTaughtAndPlain fills a scope of a Tagger with the taughtAndPlain
// embedder: two tags, "1", named "taught", whose members are the items m1, m2
// and m3 (text "member"), and "2", named "plain", with no members, and the
// item "q" (text "query").
func fillTaughtAndPlain(t *testing.T, space *Scope) {
	t.Helper()
	ctx := context.Background()
	must(t, space.SetTag(ctx, "1", "taught"))
	must(t, space.SetTag(ctx, "2", "plain"))
	for _, id := range []string{"m1", "m2", "m3"} {
		must(t, space.SetItem(ctx, id, "member"))
		must(t, space.TagByHand(ctx, id, "1"))
	}
	must(t, space.SetItem(ctx, "q", "query"))
}

// suggestTaughtAndPlain returns the tags suggested for the item "q" in a
// scope filled by fillTaughtAndPlain, under cfg.
func suggestTaughtAndPlain(t *testing.T, cfg Config) []string {
	t.Helper()
	space := mustNew(t, taughtAndPlain(), NewMemStore(), cfg).Scope("1", "10")
	fillTaughtAndPlain(t, space)
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

// TestNew_RejectsNonFiniteSettings: a minimum score that is not a finite
// number would silently let the tags it weighs in match every item (-Inf) or
// none (NaN, +Inf), a SmoothingK of +Inf would leave every member out of its
// tag's vector, and a NaN one would let no tag with members match, so New
// refuses them, naming the setting. It checks after the defaults are applied:
// a SmoothingK of -Inf is below zero, and takes DefaultSmoothingK like any
// other. rank and storedCutoff still fail closed on a NaN, for a Tagger built
// some other way (see their tests).
func TestNew_RejectsNonFiniteSettings(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	for _, tc := range []struct {
		setting string
		cfg     Config
	}{
		{"NameMinScore", Config{Model: "m", NameMinScore: nan}},
		{"NameMinScore", Config{Model: "m", NameMinScore: inf}},
		{"NameMinScore", Config{Model: "m", NameMinScore: -inf}},
		{"MemberMinScore", Config{Model: "m", MemberMinScore: nan}},
		{"MemberMinScore", Config{Model: "m", MemberMinScore: inf}},
		{"MemberMinScore", Config{Model: "m", MemberMinScore: -inf}},
		{"SmoothingK", Config{Model: "m", SmoothingK: nan}},
		{"SmoothingK", Config{Model: "m", SmoothingK: inf}},
	} {
		_, err := New(semsearch.FakeEmbedder{}, NewMemStore(), tc.cfg)
		if err == nil || !strings.Contains(err.Error(), "Config."+tc.setting) {
			t.Errorf("New with %+v = %v, want an error naming Config.%s", tc.cfg, err, tc.setting)
		}
	}
	tg := mustNew(t, semsearch.FakeEmbedder{}, NewMemStore(), Config{Model: "m", SmoothingK: -inf})
	if tg.cfg.SmoothingK != DefaultSmoothingK {
		t.Errorf("SmoothingK after New with -Inf = %v, want the default %v, as for any value below zero", tg.cfg.SmoothingK, DefaultSmoothingK)
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
		err := space.TagByHand(ctx, tc.itemID, tc.tagID)
		if !errors.Is(err, tc.want) {
			t.Errorf("TagByHand(%q, %q) = %v, want %v", tc.itemID, tc.tagID, err, tc.want)
		}
		if want := fmt.Sprintf("autotag: tag item %q with %q: ", tc.itemID, tc.tagID); err == nil || !strings.HasPrefix(err.Error(), want) {
			t.Errorf("TagByHand(%q, %q) = %v, want it to start with %q, naming both ids", tc.itemID, tc.tagID, err, want)
		}
	}
}

// TestHandTagStoreErrors_NameBothIDs: an error of the store's Link or Unlink
// comes back naming the item and the tag, and still wraps what the store
// returned. The store refuses an empty index.
func TestHandTagStoreErrors_NameBothIDs(t *testing.T) {
	ctx := context.Background()
	space := mustNew(t, semsearch.FakeEmbedder{}, NewMemStore(), Config{Model: "fake"}).Scope("", "10")
	for _, tc := range []struct {
		name string
		err  error
		want string
	}{
		{"TagByHand", space.TagByHand(ctx, "a", "1"), `autotag: tag item "a" with "1": `},
		{"UntagByHand", space.UntagByHand(ctx, "a", "1"), `autotag: untag item "a" from "1": `},
	} {
		if !errors.Is(tc.err, errEmptyIndex) || !strings.HasPrefix(tc.err.Error(), tc.want) {
			t.Errorf("%s in a scope with an empty index = %v, want the store's error, starting with %q", tc.name, tc.err, tc.want)
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

// TestTagByHand_ReusesTheNameVector: a hand tag rebuilds its tag from the name
// vector the tag holds and the vector the item holds, so it embeds nothing.
func TestTagByHand_ReusesTheNameVector(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	emb := &countingEmbedder{inner: semsearch.FakeEmbedder{}}
	space := mustNew(t, emb, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetItem(ctx, "a", "rust ownership and borrowing"))
	calls := emb.calls
	must(t, space.TagByHand(ctx, "a", "1"))
	if emb.calls != calls {
		t.Fatalf("TagByHand embedded %d times, want none", emb.calls-calls)
	}
	if rec := mustTag(t, store, "1"); rec.Built != rec.Version {
		t.Fatalf("TagByHand left the tag stale: Built %d, Version %d", rec.Built, rec.Version)
	}
}

// TestUntagByHand_RebuildsTheTagAtOnce: removing a hand tag rebuilds the tag
// before UntagByHand returns, with no Suggest to do it.
func TestUntagByHand_RebuildsTheTagAtOnce(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetItem(ctx, "a", "rust ownership and borrowing"))
	must(t, space.TagByHand(ctx, "a", "1"))
	must(t, space.UntagByHand(ctx, "a", "1"))
	rec := mustTag(t, store, "1")
	if rec.Built != rec.Version {
		t.Fatalf("UntagByHand left the tag stale: Built %d, Version %d", rec.Built, rec.Version)
	}
	if !slices.Equal(rec.Vec, blend(rec.NameVec, nil, 3)) {
		t.Fatal("the tag's vector is not its name's alone, with its one member untagged")
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

// TestSetItem_UnchangedTextUnderANewModelReembeds: the text an item has, set
// again by a Tagger of another model, is embedded under that model, and the
// tags it is tagged with by hand are rebuilt before SetItem returns.
func TestSetItem_UnchangedTextUnderANewModelReembeds(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	before := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "a"}).Scope("1", "10")
	must(t, before.SetTag(ctx, "1", "rust"))
	must(t, before.SetItem(ctx, "a", "rust ownership"))
	must(t, before.TagByHand(ctx, "a", "1"))

	emb := &countingEmbedder{inner: semsearch.FakeEmbedder{}}
	must(t, mustNew(t, emb, store, Config{Model: "b"}).Scope("1", "10").SetItem(ctx, "a", "rust ownership"))
	if emb.calls != 2 {
		t.Fatalf("embedded %d times, want 2: the item's text, then the tag's name in its rebuild", emb.calls)
	}
	if item, _, _ := store.Item(ctx, "1", "10", "a"); item.Model != "b" {
		t.Fatalf("item a: Model %q, want b", item.Model)
	}
	if rec := mustTag(t, store, "1"); rec.Model != "b" || rec.Built != rec.Version {
		t.Fatalf("tag 1: Model %q, Built %d, Version %d; want model b, built at its Version", rec.Model, rec.Built, rec.Version)
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
	before := mustTag(t, store, "1")
	must(t, space.SetItem(ctx, "a", "cooking bread at home"))
	after := mustTag(t, store, "1")
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
	rec := mustTag(t, store, "1")
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

// brokenEmbedder breaks the Embedder contract on purpose: it says its vectors
// have dims entries, and returns doc for every text embedded as a document (a
// tag's name) and query for every one embedded as a query (an item's text).
type brokenEmbedder struct {
	dims       int
	doc, query []float32
}

func (b brokenEmbedder) Dimensions() int { return b.dims }

func (b brokenEmbedder) Embed(_ context.Context, role semsearch.Role, texts []string) ([][]float32, error) {
	v := b.query
	if role == semsearch.RoleDocument {
		v = b.doc
	}
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = slices.Clone(v)
	}
	return out, nil
}

// TestEmbedderErrors_RefuseAVectorOfTheWrongLength: an embedding whose length
// is not the embedder's Dimensions, or an empty one when the embedder does not
// say its Dimensions, fails the write that needed it: SetItem stores nothing,
// and SetTag fails its rebuild.
func TestEmbedderErrors_RefuseAVectorOfTheWrongLength(t *testing.T) {
	for _, tc := range []struct {
		name string
		dims int
		vec  []float32
		want string
	}{
		{"nil", 4, nil, "embedder returned a vector of 0 dimensions, want 4"},
		{"short", 4, []float32{1, 0}, "embedder returned a vector of 2 dimensions, want 4"},
		{"long", 4, []float32{1, 0, 0, 0, 0}, "embedder returned a vector of 5 dimensions, want 4"},
		{"empty, of unknown dimensions", 0, []float32{}, "embedder returned an empty vector"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			store := NewMemStore()
			space := mustNew(t, brokenEmbedder{dims: tc.dims, doc: tc.vec, query: tc.vec}, store, Config{Model: "broken"}).Scope("1", "10")
			for _, c := range []struct {
				name string
				err  error
			}{
				{"SetItem", space.SetItem(ctx, "a", "some text")},
				{"SetTag", space.SetTag(ctx, "1", "rust")},
			} {
				if c.err == nil || !strings.Contains(c.err.Error(), tc.want) {
					t.Errorf("%s = %v, want an error saying %q", c.name, c.err, tc.want)
				}
			}
			if _, ok, _ := store.Item(ctx, "1", "10", "a"); ok {
				t.Error("SetItem stored the item, with the vector it refused")
			}
		})
	}
}

// TestEmbedderErrors_AnEmptyItemVectorPanicsNothing: an embedder that returns
// a nil vector for an item's text, and a whole one for a tag's name, fails the
// SetItem, so the item is not stored for a later hand tag to blend in: a
// member with fewer entries than the name vector would index out of range.
func TestEmbedderErrors_AnEmptyItemVectorPanicsNothing(t *testing.T) {
	ctx := context.Background()
	emb := brokenEmbedder{dims: 4, doc: unitVec(1, 0, 0, 0), query: nil}
	space := mustNew(t, emb, NewMemStore(), Config{Model: "broken"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	if err := space.SetItem(ctx, "a", "some text"); err == nil {
		t.Fatal("SetItem with an empty vector succeeded, want an error")
	}
	if err := space.TagByHand(ctx, "a", "1"); !errors.Is(err, ErrUnknownItem) {
		t.Fatalf("TagByHand of the item SetItem refused = %v, want ErrUnknownItem", err)
	}
}

// TestRebuild_RefusesAMemberOfAnotherLength: an embedder that does not say
// its Dimensions cannot have a vector's length checked as it is embedded, so a
// rebuild checks that every member has as many entries as the name vector, and
// fails, naming the member, rather than panicking on a shorter one or reading
// a longer one in part. The tag is left stale.
func TestRebuild_RefusesAMemberOfAnotherLength(t *testing.T) {
	for _, query := range [][]float32{unitVec(1, 0), unitVec(1, 0, 0, 0, 0, 0, 0, 0)} {
		t.Run(fmt.Sprintf("%d entries", len(query)), func(t *testing.T) {
			ctx := context.Background()
			store := NewMemStore()
			emb := brokenEmbedder{dims: 0, doc: unitVec(1, 0, 0, 0), query: query}
			space := mustNew(t, emb, store, Config{Model: "broken"}).Scope("1", "10")
			must(t, space.SetTag(ctx, "1", "rust"))
			must(t, space.SetItem(ctx, "a", "some text"))
			err := space.TagByHand(ctx, "a", "1")
			want := fmt.Sprintf(`item "a" has a vector of %d dimensions and the tag's name one of 4`, len(query))
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("TagByHand = %v, want an error saying %q", err, want)
			}
			if rec := mustTag(t, store, "1"); rec.Built == rec.Version {
				t.Fatal("the tag is built at its Version, though its rebuild failed")
			}
		})
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
	rec := mustTag(t, store, "1")
	if rec.Built == rec.Version {
		t.Fatal("the rebuild that missed b must not have been stored")
	}
	if _, err := space.Suggest(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	rec = mustTag(t, store, "1")
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
	rec := mustTag(t, store, "1")
	ms, _ := store.Members(ctx, "1", "10", "1")
	if rec.Built != rec.Version || len(ms) != 2 ||
		!slices.Equal(rec.Vec, blend(rec.NameVec, [][]float32{ms[0].Vec, ms[1].Vec}, 3)) {
		t.Fatalf("TagByHand must build the tag from both members: %+v", rec)
	}
}

// membersCountingStore counts the calls to Members.
type membersCountingStore struct {
	*MemStore
	calls int
}

func (c *membersCountingStore) Members(ctx context.Context, i, p, id string) ([]Member, error) {
	c.calls++
	return c.MemStore.Members(ctx, i, p, id)
}

// TestRebuild_ReadsTheMembersOnceWhenAllAreCurrent: a rebuild whose members
// are all of the Tagger's model builds the tag from its first read of them.
// Every hand tag rebuilds its tag from all of its members, so a bulk import of
// n hand tags of one tag reads n² member rows, and a second read would double
// that.
func TestRebuild_ReadsTheMembersOnceWhenAllAreCurrent(t *testing.T) {
	ctx := context.Background()
	store := &membersCountingStore{MemStore: NewMemStore()}
	space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	for _, id := range []string{"a", "b", "c"} {
		must(t, space.SetItem(ctx, id, "rust ownership, part "+id))
	}
	must(t, space.TagByHand(ctx, "a", "1"))
	must(t, space.TagByHand(ctx, "b", "1"))

	store.calls = 0
	must(t, space.TagByHand(ctx, "c", "1"))
	if store.calls != 1 {
		t.Fatalf("the rebuild of a tag whose members are all current read Members %d times, want once", store.calls)
	}
	if rec := mustTag(t, store, "1"); rec.Built != rec.Version {
		t.Fatalf("TagByHand left the tag stale: Built %d, Version %d", rec.Built, rec.Version)
	}
}

// deleteOnReadStore deletes tag "1" just before the first read of it once
// armed, so the tag goes while a rebuild of it runs.
type deleteOnReadStore struct {
	*MemStore
	armed bool
}

func (d *deleteOnReadStore) Tag(ctx context.Context, i, p, id string) (TagRecord, bool, error) {
	if d.armed && id == "1" {
		d.armed = false
		if err := d.DeleteTag(ctx, i, p, "1"); err != nil {
			return TagRecord{}, false, err
		}
	}
	return d.MemStore.Tag(ctx, i, p, id)
}

// TestRebuild_TagDeletedMeanwhileIsNoError: a tag deleted while it is being
// rebuilt leaves nothing to rebuild or embed, and the write that started the
// rebuild succeeds.
func TestRebuild_TagDeletedMeanwhileIsNoError(t *testing.T) {
	ctx := context.Background()
	store := &deleteOnReadStore{MemStore: NewMemStore()}
	emb := &countingEmbedder{inner: semsearch.FakeEmbedder{}}
	space := mustNew(t, emb, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetItem(ctx, "a", "rust ownership"))
	store.armed = true

	calls := emb.calls
	if err := space.TagByHand(ctx, "a", "1"); err != nil {
		t.Fatalf("TagByHand = %v, want no error: its tag was deleted while it was rebuilt", err)
	}
	if store.armed {
		t.Fatal("the rebuild never read the tag, so it was never deleted")
	}
	if emb.calls != calls {
		t.Fatalf("the rebuild of the deleted tag embedded %d times, want none", emb.calls-calls)
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

// editingStore makes one write to the item editID just before the first
// SetItemVector of that item, as a writer would whose change lands between a
// re-embed's read of the item and its write: it puts edit through PutItem, or
// deletes the item when edit is nil. It clears editID once the write is in.
type editingStore struct {
	*MemStore
	editID string
	edit   *ItemRecord
}

func (e *editingStore) SetItemVector(ctx context.Context, i, p, id, text string, vec []float32, model string) ([]string, bool, error) {
	if e.editID != "" && e.editID == id {
		e.editID = ""
		var err error
		if e.edit == nil {
			_, err = e.DeleteItem(ctx, i, p, id)
		} else {
			_, err = e.PutItem(ctx, i, p, *e.edit)
		}
		if err != nil {
			return nil, false, err
		}
	}
	return e.MemStore.SetItemVector(ctx, i, p, id, text, vec, model)
}

// TestModelSwitch_ReembedKeepsAConcurrentEdit: re-embedding an item of the old
// model never overwrites an edit that lands between the read of its text and
// the write of its new vector, be the item the one Suggest was asked about or
// a member of a tag Suggest rebuilds. The edit's text is what Suggest then
// scores the item by: with the vector its writer stored if that is of the new
// model, and otherwise with one embedded for that call alone. A member the
// edit leaves on the old model keeps its tag stale, never built without it,
// until the next pass re-embeds it.
func TestModelSwitch_ReembedKeepsAConcurrentEdit(t *testing.T) {
	fake := semsearch.FakeEmbedder{}
	edited, err := fake.Embed(context.Background(), semsearch.RoleQuery, []string{"cooking bread"})
	if err != nil {
		t.Fatal(err)
	}
	editOf := func(id, model string) ItemRecord {
		return ItemRecord{ID: id, Text: "cooking bread", Vec: edited[0], Model: model}
	}
	for _, tc := range []struct {
		name       string
		edit, want ItemRecord // the write that lands, and the item after Suggest
		// the embeddings under model b: doc's and m's stored texts and the two
		// tag names, plus the edit's text when the edit is of the old model
		wantCalls int
	}{
		{"the item asked about, edited under the new model", editOf("doc", "b"), editOf("doc", "b"), 4},
		{"the item asked about, edited under the old model", editOf("doc", "a"), editOf("doc", "a"), 5},
		{"a member of a tag rebuilt, edited under the new model", editOf("m", "b"), editOf("m", "b"), 4},
		// the next pass re-embeds the edit's text, under the new model
		{"a member of a tag rebuilt, edited under the old model", editOf("m", "a"), editOf("m", "b"), 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			store := &editingStore{MemStore: NewMemStore()}
			before := mustNew(t, fake, store, Config{Model: "a"}).Scope("1", "10")
			must(t, before.SetTag(ctx, "rust", "rust"))
			must(t, before.SetTag(ctx, "cooking", "cooking"))
			must(t, before.SetItem(ctx, "doc", "rust programming"))
			must(t, before.SetItem(ctx, "m", "rust ownership"))
			must(t, before.TagByHand(ctx, "m", "rust"))
			store.editID, store.edit = tc.edit.ID, &tc.edit

			emb := &countingEmbedder{inner: fake}
			got := suggest(t, mustNew(t, emb, store, Config{Model: "b"}).Scope("1", "10"), "doc")
			if store.editID != "" {
				t.Fatalf("no re-embed wrote item %s, so the edit never landed", tc.edit.ID)
			}
			if item, _, _ := store.Item(ctx, "1", "10", tc.edit.ID); !reflect.DeepEqual(item, tc.want) {
				t.Fatalf("item %s = %+v, want %+v", tc.edit.ID, item, tc.want)
			}
			if tc.edit.ID == "doc" && !slices.Equal(got, []string{"cooking"}) {
				t.Fatalf("Suggest = %v, want [cooking], from the edit's text", got)
			}
			if tc.edit.ID == "m" {
				rec := mustTag(t, store, "rust")
				if rec.Model != "b" || rec.Built != rec.Version || !slices.Equal(rec.Vec, blend(rec.NameVec, [][]float32{tc.want.Vec}, 3)) {
					t.Fatalf("tag rust = %+v, want it built under model b from its name and m's new text", rec)
				}
			}
			if emb.calls != tc.wantCalls {
				t.Fatalf("embedded %d times, want %d", emb.calls, tc.wantCalls)
			}
		})
	}
}

// TestSuggest_ItemDeletedDuringItsReembedIsUnknown: an item deleted between
// Suggest's read of it and the write of its re-embedded vector is
// ErrUnknownItem, as if it had gone before Suggest.
func TestSuggest_ItemDeletedDuringItsReembedIsUnknown(t *testing.T) {
	ctx := context.Background()
	store := &editingStore{MemStore: NewMemStore()}
	must(t, mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "a"}).Scope("1", "10").SetItem(ctx, "doc", "rust programming"))
	store.editID = "doc" // with no edit: a delete

	_, err := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "b"}).Scope("1", "10").Suggest(ctx, "doc")
	if store.editID != "" {
		t.Fatal("no re-embed wrote the item, so the delete never landed")
	}
	if !errors.Is(err, ErrUnknownItem) {
		t.Fatalf("Suggest = %v, want ErrUnknownItem", err)
	}
}

// TestSettingsChange_RebuildsWithoutReembedding: a Tagger whose scoring
// settings differ from those the tags were built with rebuilds them before it
// suggests anything, from the vectors the store holds, so a new value of any
// of the three settings applies at once and nothing is embedded again. Under
// the defaults the item "q" gets "plain" alone (see
// TestSuggest_JudgesEachTagAgainstItsOwnCutoff).
func TestSettingsChange_RebuildsWithoutReembedding(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
		want []string
	}{
		// "plain" scores 0.50, below the new minimum
		{"NameMinScore", Config{Model: "stub", NameMinScore: 0.8}, nil},
		// "taught" clears its cutoff (see TestSuggest_MemberMinScoreSetsTheCutoffOfATaughtTag)
		{"MemberMinScore", Config{Model: "stub", MemberMinScore: 0.5}, []string{"1", "2"}},
		// the members of "taught" weigh 3/13 and no longer 1/2, and its name,
		// above its minimum, carries it over its cutoff
		{"SmoothingK", Config{Model: "stub", SmoothingK: 10}, []string{"1", "2"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := NewMemStore()
			before := mustNew(t, taughtAndPlain(), store, Config{Model: "stub"}).Scope("1", "10")
			fillTaughtAndPlain(t, before)
			if got := suggest(t, before, "q"); !slices.Equal(got, []string{"2"}) {
				t.Fatalf("under the default settings, Suggest = %v, want [2]", got)
			}
			built := map[string]string{"1": mustTag(t, store, "1").Params, "2": mustTag(t, store, "2").Params}

			emb := &countingEmbedder{inner: taughtAndPlain()}
			if got := suggest(t, mustNew(t, emb, store, tc.cfg).Scope("1", "10"), "q"); !slices.Equal(got, tc.want) {
				t.Fatalf("with the new %s, Suggest = %v, want %v", tc.name, got, tc.want)
			}
			for id, params := range built {
				if rec := mustTag(t, store, id); rec.Params == params || rec.Built != rec.Version {
					t.Errorf("tag %s: Params %q (was %q), Built %d, Version %d; want new Params, built at its Version",
						id, rec.Params, params, rec.Built, rec.Version)
				}
			}
			if emb.calls != 0 {
				t.Fatalf("embedded %d times, want none: the names and the items are embedded under this model already", emb.calls)
			}
		})
	}
}

// errStoreDown is what failingStore answers with.
var errStoreDown = errors.New("store is down")

// failingStore is a MemStore whose SetTagVectors fails while fail is set, as a
// backend's would while its database cannot be reached. refused lists the tags
// it failed to store, in the order they came.
type failingStore struct {
	*MemStore
	fail    bool
	refused []string
}

func (f *failingStore) SetTagVectors(ctx context.Context, i, p, id string, v int64, tv TagVectors) (bool, error) {
	if f.fail {
		f.refused = append(f.refused, id)
		return false, errStoreDown
	}
	return f.MemStore.SetTagVectors(ctx, i, p, id, v, tv)
}

// TestRebuildFailure_KeepsTheChangeAndHealsOnSuggest: a write whose rebuild
// fails stays written and returns an error wrapping ErrNotRebuilt and the
// rebuild's, naming the tag, and the next Suggest rebuilds the tag it left
// stale.
func TestRebuildFailure_KeepsTheChangeAndHealsOnSuggest(t *testing.T) {
	ctx := context.Background()
	store := &failingStore{MemStore: NewMemStore()}
	space := mustNew(t, semsearch.FakeEmbedder{}, store,
		Config{Model: "fake", NameMinScore: 0.3, MemberMinScore: 0.5}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "cooking / bread"))
	must(t, space.SetItem(ctx, "a", "Baking bread at home: a cooking guide"))

	store.fail = true
	err := space.TagByHand(ctx, "a", "1")
	if !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, errStoreDown) {
		t.Fatalf("TagByHand = %v, want ErrNotRebuilt and the store's error", err)
	}
	if want := `autotag: tag not rebuilt: tag "1": store is down`; err.Error() != want {
		t.Errorf("TagByHand = %q, want %q", err, want)
	}
	ms, _ := store.Members(ctx, "1", "10", "1")
	if len(ms) != 1 || ms[0].ItemID != "a" {
		t.Fatalf("members = %+v, want a: the link stays although its rebuild failed", ms)
	}
	if rec := mustTag(t, store, "1"); rec.Built == rec.Version {
		t.Fatal("the tag is built at its Version although its rebuild failed")
	}

	store.fail = false
	got := suggest(t, space, "a")
	if rec := mustTag(t, store, "1"); rec.Built != rec.Version {
		t.Fatalf("Suggest left the tag stale: Built %d, Version %d", rec.Built, rec.Version)
	}
	if !slices.Contains(got, "1") {
		t.Fatalf("Suggest = %v, want 1 among them", got)
	}
}

// TestRebuildAll_StopsAtTheFirstError: a write that leaves several tags to
// rebuild stops at the first rebuild that fails and returns its error; the
// tags after it stay stale, for the next Suggest.
func TestRebuildAll_StopsAtTheFirstError(t *testing.T) {
	ctx := context.Background()
	store := &failingStore{MemStore: NewMemStore()}
	space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "rust"))
	must(t, space.SetTag(ctx, "2", "cooking"))
	must(t, space.SetItem(ctx, "a", "rust ownership"))
	must(t, space.TagByHand(ctx, "a", "1"))
	must(t, space.TagByHand(ctx, "a", "2"))

	store.fail = true
	err := space.SetItem(ctx, "a", "cooking with rust")
	if !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, errStoreDown) || err.Error() != `autotag: tag not rebuilt: tag "1": store is down` {
		t.Fatalf("SetItem = %v, want the error of the rebuild of tag 1", err)
	}
	if !slices.Equal(store.refused, []string{"1"}) {
		t.Fatalf("SetItem's rebuilds tried to store tags %v, want 1 alone: they stop at the first error", store.refused)
	}
	if rec := mustTag(t, store, "2"); rec.Built == rec.Version {
		t.Fatal("tag 2 is built at its Version, though no rebuild of it was stored")
	}
}

// TestWrites_ThatCannotRebuildAreStoredAndRetryAsANoOp: every write that
// stores its change and then cannot rebuild a tag it affects returns an error
// wrapping ErrNotRebuilt and the cause. The change is stored, so calling the
// write again changes nothing and succeeds, and a later Rebuild builds the tag.
func TestWrites_ThatCannotRebuildAreStoredAndRetryAsANoOp(t *testing.T) {
	for _, tc := range []struct {
		name  string
		write func(ctx context.Context, space *Scope) error
	}{
		{"SetTag", func(ctx context.Context, s *Scope) error { return s.SetTag(ctx, "1", "rust, renamed") }},
		{"SetItem", func(ctx context.Context, s *Scope) error { return s.SetItem(ctx, "a", "rust ownership, edited") }},
		{"DeleteItem", func(ctx context.Context, s *Scope) error { return s.DeleteItem(ctx, "a") }},
		{"TagByHand", func(ctx context.Context, s *Scope) error { return s.TagByHand(ctx, "b", "1") }},
		{"UntagByHand", func(ctx context.Context, s *Scope) error { return s.UntagByHand(ctx, "a", "1") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			store := &failingStore{MemStore: NewMemStore()}
			space := mustNew(t, semsearch.FakeEmbedder{}, store, Config{Model: "fake"}).Scope("1", "10")
			must(t, space.SetTag(ctx, "1", "rust"))
			must(t, space.SetItem(ctx, "a", "rust ownership"))
			must(t, space.SetItem(ctx, "b", "rust borrowing"))
			must(t, space.TagByHand(ctx, "a", "1"))

			store.fail = true
			if err := tc.write(ctx, space); !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, errStoreDown) {
				t.Fatalf("%s = %v, want ErrNotRebuilt and the store's error", tc.name, err)
			}
			refused := len(store.refused)
			if err := tc.write(ctx, space); err != nil || len(store.refused) != refused {
				t.Fatalf("%s again = %v, after %d more rebuilds; want nil and none: the change is stored",
					tc.name, err, len(store.refused)-refused)
			}
			store.fail = false
			must(t, space.Rebuild(ctx))
			if rec := mustTag(t, store, "1"); rec.Built != rec.Version {
				t.Fatalf("Rebuild left the tag stale: Built %d, Version %d", rec.Built, rec.Version)
			}
		})
	}
}

// rejectingStub is the embedder of the ErrNotRebuilt tests: the name "alpha"
// scores 0.80 against the item "q" (text "query") and "gamma" 0.70, and it has
// no vector for any other text, such as the names "bad" and "bad too", so a
// tag of such a name cannot be built.
func rejectingStub() stubEmbedder {
	const dims = 4
	return stubEmbedder{dims: dims, vecs: map[string][]float32{
		"alpha": scoreVec(dims, 1, 0.80),
		"gamma": scoreVec(dims, 2, 0.70),
		"query": unitVec(1, 0, 0, 0),
	}}
}

// wantNotRebuilt stops the test unless err wraps ErrNotRebuilt, and checks
// that it names each of the tags once.
func wantNotRebuilt(t *testing.T, what string, err error, tagIDs ...string) {
	t.Helper()
	if !errors.Is(err, ErrNotRebuilt) {
		t.Fatalf("%s = %v, want ErrNotRebuilt", what, err)
	}
	for _, id := range tagIDs {
		if n := strings.Count(err.Error(), fmt.Sprintf("tag %q", id)); n != 1 {
			t.Errorf("%s = %q, which names tag %s %d times, want once", what, err, id, n)
		}
	}
}

// TestSetTag_ANameTheEmbedderRejectsIsStoredButNotRebuilt: SetTag stores the
// tag before it embeds the name, so a name the embedder rejects fails the
// rebuild and not the write: the error wraps ErrNotRebuilt and the cause and
// names the tag, the tag is stored, and SetTag again is a no-op. A write that
// fails before it stores anything, as SetItem does on a text the embedder
// rejects, is not ErrNotRebuilt.
func TestSetTag_ANameTheEmbedderRejectsIsStoredButNotRebuilt(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	space := mustNew(t, rejectingStub(), store, Config{Model: "stub"}).Scope("1", "10")

	err := space.SetTag(ctx, "1", "bad")
	if want := `autotag: tag not rebuilt: tag "1": stubEmbedder: no vector for "bad"`; !errors.Is(err, ErrNotRebuilt) || err.Error() != want {
		t.Fatalf("SetTag with a name the embedder rejects = %v, want ErrNotRebuilt: %q", err, want)
	}
	if rec := mustTag(t, store, "1"); rec.Name != "bad" || rec.Built == rec.Version {
		t.Fatalf("tag = %+v, want it stored under its name, and unbuilt", rec)
	}
	if err := space.SetTag(ctx, "1", "bad"); err != nil {
		t.Fatalf("SetTag again = %v, want nil: the change is stored", err)
	}

	err = space.SetItem(ctx, "a", "a text the embedder rejects")
	if err == nil || errors.Is(err, ErrNotRebuilt) {
		t.Fatalf("SetItem with a text the embedder rejects = %v, want an error, and not ErrNotRebuilt: nothing was stored", err)
	}
	if _, ok, _ := store.Item(ctx, "1", "10", "a"); ok {
		t.Fatal("SetItem stored the item whose text it could not embed")
	}
}

// TestSuggest_ATagThatCannotBeRebuiltLeavesTheOthers: Suggest goes on past a
// tag it cannot rebuild, and ranks the others as rank does: a tag never built
// is left out, and one whose last rebuild failed scores with the vectors it
// had. It returns their ids with an error that wraps ErrNotRebuilt and names
// each tag that failed, once, and Rebuild returns the same error.
func TestSuggest_ATagThatCannotBeRebuiltLeavesTheOthers(t *testing.T) {
	ctx := context.Background()
	store := &failingStore{MemStore: NewMemStore()}
	space := mustNew(t, rejectingStub(), store, Config{Model: "stub"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "alpha"))
	wantNotRebuilt(t, "SetTag(2, bad)", space.SetTag(ctx, "2", "bad"), "2")
	must(t, space.SetTag(ctx, "3", "gamma"))
	wantNotRebuilt(t, "SetTag(3, bad too)", space.SetTag(ctx, "3", "bad too"), "3")
	store.fail = true
	wantNotRebuilt(t, "SetTag(4, alpha) while the store is down", space.SetTag(ctx, "4", "alpha"), "4")
	store.fail = false
	must(t, space.SetItem(ctx, "q", "query"))

	ids, err := space.Suggest(ctx, "q")
	// 2 has no vector, 3 keeps the one of its old name, and 4, which comes
	// after the tags that fail, is rebuilt and scores as 1 does
	if want := []string{"1", "4", "3"}; !slices.Equal(ids, want) {
		t.Errorf("Suggest = %v, want %v", ids, want)
	}
	wantNotRebuilt(t, "Suggest", err, "2", "3")
	if strings.Contains(err.Error(), `tag "4"`) {
		t.Errorf("Suggest = %q, which names tag 4, though it rebuilt it", err)
	}
	wantNotRebuilt(t, "Rebuild", space.Rebuild(ctx), "2", "3")
}

// flakyStore is a MemStore whose Item fails while itemDown is set, and whose
// Tags fails from its tagsDownAt-th call on (0: never), as a backend's would
// once its database cannot be reached.
type flakyStore struct {
	*MemStore
	itemDown              bool
	tagsCalls, tagsDownAt int
}

func (f *flakyStore) Item(ctx context.Context, i, p, id string) (ItemRecord, bool, error) {
	if f.itemDown {
		return ItemRecord{}, false, errStoreDown
	}
	return f.MemStore.Item(ctx, i, p, id)
}

func (f *flakyStore) Tags(ctx context.Context, i, p string) ([]TagRecord, error) {
	f.tagsCalls++
	if f.tagsDownAt > 0 && f.tagsCalls >= f.tagsDownAt {
		return nil, errStoreDown
	}
	return f.MemStore.Tags(ctx, i, p)
}

// TestSuggest_OtherErrorsReturnNoIDs: only tags that cannot be rebuilt let
// Suggest return ids with its error. A failed read of the item or of the tags
// returns none, and no ErrNotRebuilt, even when it comes after a tag failed to
// rebuild in the same call, and so does an unknown item.
func TestSuggest_OtherErrorsReturnNoIDs(t *testing.T) {
	ctx := context.Background()
	store := &flakyStore{MemStore: NewMemStore()}
	space := mustNew(t, rejectingStub(), store, Config{Model: "stub"}).Scope("1", "10")
	must(t, space.SetTag(ctx, "1", "alpha"))
	wantNotRebuilt(t, "SetTag(2, bad)", space.SetTag(ctx, "2", "bad"), "2")
	must(t, space.SetItem(ctx, "q", "query"))

	for _, tc := range []struct {
		name       string
		itemDown   bool
		tagsDownAt int
		itemID     string
		want       error
	}{
		{"the read of the item fails", true, 0, "q", errStoreDown},
		{"the first read of the tags fails", false, 1, "q", errStoreDown},
		// the first pass fails to rebuild tag 2, and the second reads the tags again
		{"the read of the tags after a failed rebuild fails", false, 2, "q", errStoreDown},
		{"the item is unknown", false, 0, "missing", ErrUnknownItem},
	} {
		store.itemDown, store.tagsCalls, store.tagsDownAt = tc.itemDown, 0, tc.tagsDownAt
		ids, err := space.Suggest(ctx, tc.itemID)
		if ids != nil || !errors.Is(err, tc.want) || errors.Is(err, ErrNotRebuilt) {
			t.Errorf("Suggest when %s = %v, %v; want no ids and %v, and not ErrNotRebuilt", tc.name, ids, err, tc.want)
		}
	}
}

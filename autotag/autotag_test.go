package autotag

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/go-bumbu/langai/semsearch"
)

// scope is where every test keeps its tags.
var scope = Scope{Index: "1", Partition: "10"}

// nameVecEntry is a cached name embedding and the name hash it is for.
type nameVecEntry struct {
	v    []float32
	hash string
}

// fakeTagStore is an in-memory TagStore. An err* field makes the matching
// method fail with it.
type fakeTagStore struct {
	nameVecs map[string]nameVecEntry
	members  map[string][][]float32
	// memberHashes must change whenever members does, or the blend hash stays constant.
	memberHashes map[string][]string

	errNameVector, errSetNameVector, errMembers error
}

func newFakeTagStore() *fakeTagStore {
	return &fakeTagStore{nameVecs: map[string]nameVecEntry{}, members: map[string][][]float32{}, memberHashes: map[string][]string{}}
}

func (f *fakeTagStore) NameVector(_ context.Context, tagID string) ([]float32, string, error) {
	if f.errNameVector != nil {
		return nil, "", f.errNameVector
	}
	e, ok := f.nameVecs[tagID]
	if !ok {
		return nil, "", nil
	}
	return e.v, e.hash, nil
}

func (f *fakeTagStore) SetNameVector(_ context.Context, tagID string, vec []float32, nameHash string) error {
	if f.errSetNameVector != nil {
		return f.errSetNameVector
	}
	f.nameVecs[tagID] = nameVecEntry{v: vec, hash: nameHash}
	return nil
}

func (f *fakeTagStore) Members(_ context.Context, tagID string) ([][]float32, []string, error) {
	if f.errMembers != nil {
		return nil, nil, f.errMembers
	}
	return f.members[tagID], f.memberHashes[tagID], nil
}

func TestNew_Defaults(t *testing.T) {
	tg := New(semsearch.FakeEmbedder{}, semsearch.NewMemStore(), Config{Model: "fake"})
	if tg.cfg.NameMinScore != 0.46 {
		t.Fatalf("NameMinScore = %v, want default 0.46", tg.cfg.NameMinScore)
	}
	if tg.cfg.MemberMinScore != 0.70 {
		t.Fatalf("MemberMinScore = %v, want default 0.70", tg.cfg.MemberMinScore)
	}
	if tg.cfg.SmoothingK != 3 {
		t.Fatalf("SmoothingK = %v, want default 3", tg.cfg.SmoothingK)
	}
	if tg.cfg.Collection != "tags" {
		t.Fatalf("Collection = %q, want default tags", tg.cfg.Collection)
	}
}

func TestRefresh_ColdStartUsesName(t *testing.T) {
	store := semsearch.NewMemStore()
	fake := semsearch.FakeEmbedder{}
	tg := New(fake, store, Config{Model: "fake", SmoothingK: 3})
	cut, err := tg.Refresh(context.Background(), scope, newFakeTagStore(), Tag{ID: "7", Name: "dev / rust", NameHash: "nh"})
	if err != nil {
		t.Fatal(err)
	}
	// a member-less tag is judged by its name alone, so its cutoff is the default name minimum
	if cut != 0.46 {
		t.Fatalf("cutoff = %v, want the default name minimum 0.46", cut)
	}
	// with zero members, the stored tag vector must equal the name embedding
	want, _ := fake.Embed(context.Background(), semsearch.RoleDocument, []string{"dev / rust"})
	hits, _ := store.Search(context.Background(), "1", "10", "tags", want[0], 5)
	if len(hits) != 1 || hits[0].DocID != "7" || hits[0].Score < 0.999 {
		t.Fatalf("cold-start tag vector should equal the name embedding: %#v", hits)
	}
}

// TestRefresh_ReturnsTheTagsCutoff: once a tag has members its cutoff is the
// mix tagCutoff computes from the name embedding and those members, not the
// name minimum alone. A second refresh that finds nothing changed skips the
// blend, and must report that same cutoff.
func TestRefresh_ReturnsTheTagsCutoff(t *testing.T) {
	ctx := context.Background()
	fake := semsearch.FakeEmbedder{}
	// the members are embedded as queries, the way a document is
	members, err := fake.Embed(ctx, semsearch.RoleQuery,
		[]string{"rust ownership and borrowing", "cargo crates for rust programs"})
	if err != nil {
		t.Fatal(err)
	}
	src := newFakeTagStore()
	src.members["7"] = members
	tg := New(fake, semsearch.NewMemStore(), Config{Model: "fake", SmoothingK: 3})
	tag := Tag{ID: "7", Name: "dev / rust", NameHash: "nh"}

	cut, err := tg.Refresh(ctx, scope, src, tag)
	if err != nil {
		t.Fatal(err)
	}
	// the tag's name is embedded as a document
	docVecs, err := fake.Embed(ctx, semsearch.RoleDocument, []string{"dev / rust"})
	if err != nil {
		t.Fatal(err)
	}
	nameVec := docVecs[0]
	if want := tagCutoff(nameVec, members, 3, 0.46, 0.70); cut != want {
		t.Fatalf("cutoff = %v, want %v", cut, want)
	}
	// guard: were the cutoff just the name minimum, the skipped refresh below
	// could not be told apart from one that reports only the name minimum
	if cut == 0.46 {
		t.Fatalf("cutoff %v is the name minimum: the fixture's members had no effect", cut)
	}

	// nothing changed, so this refresh skips the blend; it must still report the
	// cutoff the members give the tag
	skipped, err := tg.Refresh(ctx, scope, src, tag)
	if err != nil {
		t.Fatal(err)
	}
	if skipped != cut {
		t.Fatalf("skipped refresh returned cutoff %v, want the first call's %v", skipped, cut)
	}
}

// rustCooking is a document that shares several words with tag "rust" and
// one with tag "cooking": under FakeEmbedder it scores ~0.61 against "rust"
// and ~0.20 against "cooking", so a NameMinScore of 0.15 clears both.
const rustCooking = "The Rust Programming Language\nA systems programming language focused on safety and speed\nrust-lang.org\n" +
	"Rust is a modern systems programming language. Rust emphasizes memory safety " +
	"without garbage collection. Learn Rust by building real projects in Rust. " +
	"I also enjoy cooking on weekends."

// TestAssign_ColdStartTagsByName: tags without members are assigned by their
// names alone, and every one whose score clears the name minimum is.
func TestAssign_ColdStartTagsByName(t *testing.T) {
	ctx := context.Background()
	tg := New(semsearch.FakeEmbedder{}, semsearch.NewMemStore(), Config{Model: "fake", NameMinScore: 0.15, SmoothingK: 3})
	vec, err := tg.Embed(ctx, rustCooking)
	if err != nil {
		t.Fatal(err)
	}
	tags := []Tag{{ID: "1", Name: "rust", NameHash: "hash-rust"}, {ID: "2", Name: "cooking", NameHash: "hash-cooking"}}
	got, err := tg.Assign(ctx, scope, newFakeTagStore(), tags, vec)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []string{"1", "2"}) {
		t.Fatalf("Assign = %v, want [1 2]: rust first, it scores higher", got)
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

// TestEmbed_DocumentsAreQueriesAndNamesAreDocuments: a document is searched
// against the tag vectors, so it is embedded as a query, and a tag's name as
// a document. An asymmetric model prefixes the two differently.
func TestEmbed_DocumentsAreQueriesAndNamesAreDocuments(t *testing.T) {
	ctx := context.Background()
	emb := &countingEmbedder{inner: semsearch.FakeEmbedder{}}
	tg := New(emb, semsearch.NewMemStore(), Config{Model: "fake", NameMinScore: 0.15})
	vec, err := tg.Embed(ctx, rustCooking)
	if err != nil {
		t.Fatal(err)
	}
	tags := []Tag{{ID: "1", Name: "rust", NameHash: "hash-rust"}, {ID: "2", Name: "cooking", NameHash: "hash-cooking"}}
	if _, err := tg.Assign(ctx, scope, newFakeTagStore(), tags, vec); err != nil {
		t.Fatal(err)
	}
	want := []semsearch.Role{semsearch.RoleQuery, semsearch.RoleDocument, semsearch.RoleDocument}
	if !slices.Equal(emb.roles, want) {
		t.Fatalf("embed roles = %v, want %v (the document as a query, then one document per tag)", emb.roles, want)
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

// taughtAndPlain is the fixture of the per-tag cutoff tests: a document
// embedded as q and a scope of two tags. "taught" has three members and
// blends to a score of 0.705 against the document: its name scores 0.55 and
// its members' centroid 0.60. "plain" has no members and scores 0.50 by its
// name.
func taughtAndPlain() (stub stubEmbedder, src *fakeTagStore, tags []Tag, q []float32) {
	q = unitVec(1, 0, 0, 0)
	m := unitVec(0.60, 0, 0.80, 0)
	stub = stubEmbedder{dims: 4, vecs: map[string][]float32{
		"taught": unitVec(0.55, math.Sqrt(1-0.55*0.55), 0, 0),
		"plain":  unitVec(0.50, 0, 0, math.Sqrt(0.75)),
	}}
	src = newFakeTagStore()
	src.members["1"] = [][]float32{m, m, m}
	tags = []Tag{{ID: "1", Name: "taught", NameHash: "t"}, {ID: "2", Name: "plain", NameHash: "p"}}
	return stub, src, tags, q
}

// With the default minimums "taught" blends to 0.705, which clears a global
// 0.46 but not its own cutoff (0.711), since its name scores 0.55 against 0.46
// and its members 0.60 against 0.70. "plain" scores 0.50 against 0.46.
// "taught" ranks first, so an implementation that stops at the first refused
// hit never reaches "plain".
func TestAssign_JudgesEachTagAgainstItsOwnCutoff(t *testing.T) {
	stub, src, tags, q := taughtAndPlain()
	tg := New(stub, semsearch.NewMemStore(), Config{Model: "stub"})
	got, err := tg.Assign(context.Background(), scope, src, tags, q)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []string{"2"}) {
		t.Fatalf("Assign = %v, want [plain] only", got)
	}
}

// MemberMinScore reaches the Tagger, and moves the cutoff of a tag with
// members. The default of 0.70 hides it: a Tagger that used 0.70 whatever it
// was given would pass every other test. Lowered to 0.5, the cutoff of "taught"
// (3 members, so a blend weight of 0.5) drops from 0.711 to about 0.589, below
// its score of 0.705, so it is assigned along with "plain".
func TestAssign_MemberMinScoreSetsTheCutoffOfATaughtTag(t *testing.T) {
	stub, src, tags, q := taughtAndPlain()
	tg := New(stub, semsearch.NewMemStore(), Config{Model: "stub", MemberMinScore: 0.5})
	got, err := tg.Assign(context.Background(), scope, src, tags, q)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []string{"1", "2"}) {
		t.Fatalf("Assign = %v, want [taught plain]", got)
	}
}

func TestAssign_AssignsEveryTagThatClearsItsCutoff(t *testing.T) {
	const n = 7 // more than any cap a tagger might be tempted to have
	q := make([]float64, n+1)
	q[0] = 1
	vecs := map[string][]float32{}
	var tags []Tag
	var want []string
	for i := 1; i <= n; i++ {
		v := make([]float64, n+1)
		v[0], v[i] = 0.5, math.Sqrt(0.75) // each name scores 0.50
		name := fmt.Sprintf("tag-%d", i)
		vecs[name] = unitVec(v...)
		tags = append(tags, Tag{ID: fmt.Sprint(i), Name: name, NameHash: name})
		want = append(want, fmt.Sprint(i))
	}
	tg := New(stubEmbedder{dims: n + 1, vecs: vecs}, semsearch.NewMemStore(), Config{Model: "stub"})
	got, err := tg.Assign(context.Background(), scope, newFakeTagStore(), tags, unitVec(q...))
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(got)
	if !slices.Equal(got, want) {
		t.Fatalf("Assign = %v, want all seven", got)
	}
}

// TestAssign_NoTagsSearchesNothing: a scope without tags has nothing to
// assign, so Assign answers without touching the store.
func TestAssign_NoTagsSearchesNothing(t *testing.T) {
	store := &noDeleteStore{MemStore: semsearch.NewMemStore()}
	tg := New(semsearch.FakeEmbedder{}, store, Config{Model: "fake"})
	got, err := tg.Assign(context.Background(), scope, newFakeTagStore(), nil, unitVec(1, 0))
	if err != nil || got != nil {
		t.Fatalf("Assign = %v, %v; want no ids and no error", got, err)
	}
	if store.searches != 0 {
		t.Fatalf("Assign searched %d times with no tags", store.searches)
	}
}

// TestAssign_KeepsToItsScopeAndCollection: the tag vectors live under the
// configured collection in the scope's coordinates, and vectors elsewhere,
// another partition's tags or another collection's documents, are neither
// assigned nor deleted as ghosts.
func TestAssign_KeepsToItsScopeAndCollection(t *testing.T) {
	ctx := context.Background()
	const dims = 4
	store := semsearch.NewMemStore()
	elsewhere := semsearch.Chunk{Vector: scoreVec(dims, 3, 0.90)}
	for _, at := range []struct{ partition, collection string }{{"20", "labels"}, {"10", "docs"}} {
		if err := store.Upsert(ctx, "1", at.partition, at.collection, "999", "h", []semsearch.Chunk{elsewhere}); err != nil {
			t.Fatal(err)
		}
	}
	stub := stubEmbedder{dims: dims, vecs: map[string][]float32{"alpha": scoreVec(dims, 1, 0.80)}}
	tg := New(stub, store, Config{Model: "stub", Collection: "labels"})
	got, err := tg.Assign(ctx, scope, newFakeTagStore(), []Tag{{ID: "1", Name: "alpha", NameHash: "a"}}, unitVec(1, 0, 0, 0))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []string{"1"}) {
		t.Fatalf("Assign = %v, want [1]", got)
	}
	if h, _ := store.ContentHash(ctx, "1", "10", "labels", "1"); h == "" {
		t.Fatal("the tag vector was not stored under the configured collection")
	}
	for _, at := range []struct{ partition, collection string }{{"20", "labels"}, {"10", "docs"}} {
		if h, _ := store.ContentHash(ctx, "1", at.partition, at.collection, "999"); h == "" {
			t.Errorf("the vector in partition %s, collection %s was deleted", at.partition, at.collection)
		}
	}
}

// pickTags judges each hit against its own tag's cutoff: a hit below its cutoff
// does not end the picking, a score equal to the cutoff qualifies, a hit that is
// not a current tag is skipped (it has no cutoff), and nothing limits how many
// tags are picked. The picks come in hit order. It fails closed: a NaN score, or
// a NaN cutoff, picks nothing, though every comparison with a NaN is false.
func TestPickTags_JudgesEachHitAgainstItsOwnTagsCutoff(t *testing.T) {
	cuts := map[string]float64{
		"1": 0.70,
		"2": 0.50,
		"3": 0.46,
		"4": 0.46,
		"5": 0.46,
		"6": 0.46,
		"7": 0.46,
		"8": math.NaN(),
	}
	hit := func(docID string, score float64) semsearch.Match {
		return semsearch.Match{DocID: docID, Score: score}
	}
	for _, tc := range []struct {
		name string
		hits []semsearch.Match
		want []string
	}{
		{"no hits", nil, nil},
		{"a hit below its cutoff does not hide the ones after it",
			[]semsearch.Match{hit("1", 0.65), hit("2", 0.55)}, []string{"2"}},
		{"a score equal to the cutoff qualifies",
			[]semsearch.Match{hit("2", 0.50)}, []string{"2"}},
		{"a hit that is not a current tag is skipped",
			[]semsearch.Match{hit("999", 0.90), hit("2", 0.55)}, []string{"2"}},
		{"a NaN score is not picked, and does not hide the hits after it",
			[]semsearch.Match{hit("2", math.NaN()), hit("3", 0.60)}, []string{"3"}},
		{"no score reaches a NaN cutoff",
			[]semsearch.Match{hit("8", 1), hit("8", math.Inf(1))}, nil},
		{"every tag that clears its cutoff is picked, in hit order",
			[]semsearch.Match{hit("7", 0.9), hit("6", 0.8), hit("5", 0.7), hit("4", 0.6), hit("3", 0.5), hit("2", 0.55), hit("1", 0.75)},
			[]string{"7", "6", "5", "4", "3", "2", "1"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := pickTags(tc.hits, cuts); !slices.Equal(got, tc.want) {
				t.Fatalf("pickTags = %v, want %v", got, tc.want)
			}
		})
	}
}

// A cutoff that is not a number is reached by no score. A caller's
// configuration should refuse one, but a Tagger built any other way must not
// answer it by assigning every tag: with a NaN NameMinScore neither "alpha",
// which scores 0.80 against the document, nor "beta", which scores below
// zero, is assigned.
func TestAssign_NaNCutoffAssignsNoTag(t *testing.T) {
	const dims = 4
	stub := stubEmbedder{dims: dims, vecs: map[string][]float32{
		"alpha": scoreVec(dims, 1, 0.80),
		"beta":  scoreVec(dims, 2, -0.30),
	}}
	tags := []Tag{{ID: "1", Name: "alpha", NameHash: "a"}, {ID: "2", Name: "beta", NameHash: "b"}}
	tg := New(stub, semsearch.NewMemStore(), Config{Model: "stub", NameMinScore: math.NaN()})
	got, err := tg.Assign(context.Background(), scope, newFakeTagStore(), tags, unitVec(1, 0, 0, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("Assign = %v, want none: no score reaches a NaN cutoff", got)
	}
}

// scoreVec returns a unit vector whose cosine with the first axis, where the
// tests put the document, is exactly score. The rest of its length goes on
// axis, so vectors on different axes have nothing in common but the document.
func scoreVec(dims, axis int, score float64) []float32 {
	v := make([]float64, dims)
	v[0], v[axis] = score, math.Sqrt(1-score*score)
	return unitVec(v...)
}

// A ghost is the vector a deleted tag left behind, and it can outscore current
// tags. The search has one slot per current tag, so each ghost it shows takes a
// slot from a tag. In every case the ghosts are stored beforehand, each
// outscoring a current tag so that a search meets it, and the current tags
// (listed by the score their names get against the document) all clear the
// default name minimum of 0.46: one Assign must return every current tag and
// leave no ghost in the store.
func TestAssign_DeletesGhostVectorsAndAssignsEveryQualifyingTag(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tags   []float64
		ghosts []float64
	}{
		{"one ghost above the only tag", []float64{0.5}, []float64{1}},
		{"two ghosts above the only tag", []float64{0.5}, []float64{0.9, 0.7}},
		{"two ghosts, one above and one between two tags", []float64{0.8, 0.5}, []float64{0.9, 0.7}},
		{"six ghosts above two tags", []float64{0.8, 0.5}, []float64{0.99, 0.98, 0.97, 0.96, 0.95, 0.9}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			dims := 1 + len(tc.tags) + len(tc.ghosts)
			doc := make([]float64, dims)
			doc[0] = 1
			vecs := map[string][]float32{}
			store := semsearch.NewMemStore()
			var tags []Tag
			var want []string
			for i, score := range tc.tags {
				id, name := fmt.Sprint(i+1), fmt.Sprintf("tag-%d", i+1)
				vecs[name] = scoreVec(dims, 1+i, score)
				tags = append(tags, Tag{ID: id, Name: name, NameHash: name})
				want = append(want, id)
			}
			var ghostIDs []string
			for i, score := range tc.ghosts {
				id := fmt.Sprint(900 + i)
				ghostIDs = append(ghostIDs, id)
				chunk := semsearch.Chunk{Vector: scoreVec(dims, 1+len(tc.tags)+i, score)}
				if err := store.Upsert(ctx, "1", "10", "tags", id, "ghost", []semsearch.Chunk{chunk}); err != nil {
					t.Fatal(err)
				}
			}
			tg := New(stubEmbedder{dims: dims, vecs: vecs}, store, Config{Model: "stub"})
			got, err := tg.Assign(ctx, scope, newFakeTagStore(), tags, unitVec(doc...))
			if err != nil {
				t.Fatal(err)
			}
			slices.Sort(got)
			if !slices.Equal(got, want) {
				t.Fatalf("Assign = %v, want %v", got, want)
			}
			for _, id := range ghostIDs {
				if h, _ := store.ContentHash(ctx, "1", "10", "tags", id); h != "" {
					t.Errorf("ghost %s was not deleted", id)
				}
			}
		})
	}
}

// maxSearches bounds the searches noDeleteStore answers.
const maxSearches = 20

// noDeleteStore is a MemStore whose Delete takes no effect, as a backend's would
// if its delete silently failed. It records the ids Delete was asked for, and
// fails a search past maxSearches, so a search loop that never ends is an error
// here and not a hung test run.
type noDeleteStore struct {
	*semsearch.MemStore
	deletes  []string
	searches int
}

func (s *noDeleteStore) Search(ctx context.Context, index, partition, collection string, vector []float32, limit int) ([]semsearch.Match, error) {
	s.searches++
	if s.searches > maxSearches {
		return nil, fmt.Errorf("noDeleteStore: more than %d searches, the search loop does not end", maxSearches)
	}
	return s.MemStore.Search(ctx, index, partition, collection, vector, limit)
}

func (s *noDeleteStore) Delete(_ context.Context, _, _, _, docID string) error {
	s.deletes = append(s.deletes, docID)
	return nil
}

// A ghost whose Delete takes no effect stays in the results however often it is
// deleted. The search must still end, after one attempt at deleting it, and the
// ghost, which is not a tag, must never be assigned. The ghost holds one of the
// two slots, so beta, which scores below it, does not make the hits and the
// test says nothing about it; alpha does, and must be assigned.
func TestAssign_NeverAssignsAGhostItCannotDelete(t *testing.T) {
	ctx := context.Background()
	const dims = 4
	store := &noDeleteStore{MemStore: semsearch.NewMemStore()}
	chunk := semsearch.Chunk{Vector: scoreVec(dims, 3, 0.70)}
	if err := store.Upsert(ctx, "1", "10", "tags", "999", "ghost", []semsearch.Chunk{chunk}); err != nil {
		t.Fatal(err)
	}
	stub := stubEmbedder{dims: dims, vecs: map[string][]float32{
		"alpha": scoreVec(dims, 1, 0.80),
		"beta":  scoreVec(dims, 2, 0.50),
	}}
	tags := []Tag{{ID: "1", Name: "alpha", NameHash: "a"}, {ID: "2", Name: "beta", NameHash: "b"}}
	tg := New(stub, store, Config{Model: "stub"})
	got, err := tg.Assign(ctx, scope, newFakeTagStore(), tags, unitVec(1, 0, 0, 0))
	if err != nil {
		t.Fatal(err) // this is also what a search loop that never ends turns into
	}
	if slices.Contains(got, "999") {
		t.Fatalf("Assign = %v: the ghost was assigned", got)
	}
	if !slices.Contains(got, "1") {
		t.Fatalf("Assign = %v, want alpha (1) among them", got)
	}
	if len(store.deletes) != 1 || store.deletes[0] != "999" {
		t.Fatalf("Delete was asked for %v, want one attempt at the ghost (999)", store.deletes)
	}
}

// errStoreDown is what the failing stores below answer with.
var errStoreDown = errors.New("store is down")

// failingSearchStore is a MemStore whose Search fails, as a backend's would when
// its database cannot be reached.
type failingSearchStore struct{ *semsearch.MemStore }

func (failingSearchStore) Search(context.Context, string, string, string, []float32, int) ([]semsearch.Match, error) {
	return nil, errStoreDown
}

// failingDeleteStore is a MemStore whose Delete fails.
type failingDeleteStore struct{ *semsearch.MemStore }

func (failingDeleteStore) Delete(context.Context, string, string, string, string) error {
	return errStoreDown
}

// A vector store that fails while the tags are being searched must fail the
// assignment: answering with no tags instead would read as "this document fits
// no tag", and a caller that replaces a document's tags with the answer would
// drop every tag it holds. The ghost outscores the only current tag, so it
// takes the search's only slot and the search has to delete it.
func TestAssign_VectorStoreErrorsFailTheAssignment(t *testing.T) {
	const dims = 3
	for _, tc := range []struct {
		name  string
		store func(*semsearch.MemStore) semsearch.Store
	}{
		{"Search fails", func(m *semsearch.MemStore) semsearch.Store { return failingSearchStore{m} }},
		{"Delete of a ghost fails", func(m *semsearch.MemStore) semsearch.Store { return failingDeleteStore{m} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			mem := semsearch.NewMemStore()
			ghost := semsearch.Chunk{Vector: scoreVec(dims, 2, 0.90)}
			if err := mem.Upsert(ctx, "1", "10", "tags", "999", "ghost", []semsearch.Chunk{ghost}); err != nil {
				t.Fatal(err)
			}
			stub := stubEmbedder{dims: dims, vecs: map[string][]float32{"alpha": scoreVec(dims, 1, 0.50)}}
			tg := New(stub, tc.store(mem), Config{Model: "stub"})
			got, err := tg.Assign(ctx, scope, newFakeTagStore(), []Tag{{ID: "1", Name: "alpha", NameHash: "a"}}, unitVec(1, 0, 0))
			if !errors.Is(err, errStoreDown) || got != nil {
				t.Fatalf("Assign = %v, %v; want no ids and the store's error", got, err)
			}
		})
	}
}

// TestRefresh_TagStoreErrorsFailTheRefresh: the caller's storage failing at
// any step fails the refresh, and a name vector that could not be cached is
// still not blended into a stored tag vector the next refresh would skip.
func TestRefresh_TagStoreErrorsFailTheRefresh(t *testing.T) {
	for _, tc := range []struct {
		name   string
		break_ func(*fakeTagStore)
	}{
		{"NameVector", func(f *fakeTagStore) { f.errNameVector = errStoreDown }},
		{"SetNameVector", func(f *fakeTagStore) { f.errSetNameVector = errStoreDown }},
		{"Members", func(f *fakeTagStore) { f.errMembers = errStoreDown }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := semsearch.NewMemStore()
			src := newFakeTagStore()
			tc.break_(src)
			tg := New(semsearch.FakeEmbedder{}, store, Config{Model: "fake"})
			if _, err := tg.Refresh(context.Background(), scope, src, Tag{ID: "7", Name: "dev / rust", NameHash: "nh"}); !errors.Is(err, errStoreDown) {
				t.Fatalf("err = %v, want the TagStore's error", err)
			}
			if h, _ := store.ContentHash(context.Background(), "1", "10", "tags", "7"); h != "" {
				t.Fatal("a tag vector was stored although the refresh failed")
			}
		})
	}
}

// vectorlessEmbedder breaks the Embedder contract on purpose: every call
// succeeds without returning a single vector.
type vectorlessEmbedder struct{}

func (vectorlessEmbedder) Dimensions() int { return 64 }

func (vectorlessEmbedder) Embed(context.Context, semsearch.Role, []string) ([][]float32, error) {
	return nil, nil
}

// TestRefresh_VectorlessEmbedderIsAnError: a name embedding that succeeds
// without a vector must fail the refresh, not panic the caller.
func TestRefresh_VectorlessEmbedderIsAnError(t *testing.T) {
	src := newFakeTagStore()
	tg := New(vectorlessEmbedder{}, semsearch.NewMemStore(), Config{Model: "fake"})
	if _, err := tg.Refresh(context.Background(), scope, src, Tag{ID: "7", Name: "dev / rust", NameHash: "nh"}); err == nil {
		t.Fatal("Refresh succeeded without a name vector")
	}
	if _, ok := src.nameVecs["7"]; ok {
		t.Fatal("a name vector was cached although none was embedded")
	}
}

// TestEmbed_VectorlessEmbedderIsAnError: a document embedding that succeeds
// without a vector must be an error, not a nil vector to search with.
func TestEmbed_VectorlessEmbedderIsAnError(t *testing.T) {
	tg := New(vectorlessEmbedder{}, semsearch.NewMemStore(), Config{Model: "fake"})
	if vec, err := tg.Embed(context.Background(), "some text"); err == nil {
		t.Fatalf("Embed = %v, want an error for an embedder that returned no vector", vec)
	}
}

// TestRefresh_SecondCallSkipsUnchangedWork covers the two caches the doc
// comment promises: an unchanged name is not re-embedded, and an unchanged
// blendHash short-circuits before the blend/Upsert.
func TestRefresh_SecondCallSkipsUnchangedWork(t *testing.T) {
	store := semsearch.NewMemStore()
	emb := &countingEmbedder{inner: semsearch.FakeEmbedder{}}
	src := newFakeTagStore()
	tg := New(emb, store, Config{Model: "fake", SmoothingK: 3})

	ctx := context.Background()
	tag := Tag{ID: "7", Name: "dev / rust", NameHash: "nh"}
	first, err := tg.Refresh(ctx, scope, src, tag)
	if err != nil {
		t.Fatal(err)
	}
	if emb.calls != 1 {
		t.Fatalf("first refresh embedded %d times, want 1", emb.calls)
	}
	second, err := tg.Refresh(ctx, scope, src, tag)
	if err != nil {
		t.Fatal(err)
	}
	if emb.calls != 1 {
		t.Fatalf("unchanged name was re-embedded: %d calls, want 1", emb.calls)
	}
	// a skipped refresh still reports the tag's cutoff
	if second != first {
		t.Fatalf("skipped refresh returned cutoff %v, want the first call's %v", second, first)
	}

	// a renamed tag (new hash) must re-embed and re-store
	if _, err := tg.Refresh(ctx, scope, src, Tag{ID: "7", Name: "dev / go", NameHash: "nh2"}); err != nil {
		t.Fatal(err)
	}
	if emb.calls != 2 {
		t.Fatalf("renamed tag embedded %d times, want 2", emb.calls)
	}
	want, _ := emb.inner.Embed(ctx, semsearch.RoleDocument, []string{"dev / go"})
	hits, _ := store.Search(ctx, "1", "10", "tags", want[0], 5)
	if len(hits) != 1 || hits[0].DocID != "7" || hits[0].Score < 0.999 {
		t.Fatalf("renamed tag vector was not re-stored: %#v", hits)
	}
}

// TestRefresh_FollowsTheMemberSet: a tag's vector moves with its members. A
// refresh is skipped when nothing it depends on changed, and the members'
// content hashes are how it tells that they did, so a changed member set must
// be blended in even though the tag's name and the settings stayed the same.
// The name leans along axis 0, the first member along axis 1 and the second
// along axis 2, so a query along axis 2 scores 0 against the first vector and
// about 0.32 against the second.
func TestRefresh_FollowsTheMemberSet(t *testing.T) {
	ctx := context.Background()
	stub := stubEmbedder{dims: 3, vecs: map[string][]float32{"tag": unitVec(1, 0, 0)}}
	store := semsearch.NewMemStore()
	src := newFakeTagStore()
	tg := New(stub, store, Config{Model: "stub"})
	query := unitVec(0, 0, 1)

	// refresh gives the tag one member, and returns the blend hash and the score
	// of the query that the store ends up with
	refresh := func(member []float32, memberHash string) (string, float64) {
		t.Helper()
		src.members["7"], src.memberHashes["7"] = [][]float32{member}, []string{memberHash}
		if _, err := tg.Refresh(ctx, scope, src, Tag{ID: "7", Name: "tag", NameHash: "nh"}); err != nil {
			t.Fatal(err)
		}
		stored, err := store.ContentHash(ctx, "1", "10", "tags", "7")
		if err != nil {
			t.Fatal(err)
		}
		hits, err := store.Search(ctx, "1", "10", "tags", query, 1)
		if err != nil || len(hits) != 1 {
			t.Fatalf("search = %v, %v, want the tag's one vector", hits, err)
		}
		return stored, hits[0].Score
	}
	hashA, scoreA := refresh(unitVec(0, 1, 0), "hash-a")
	hashB, scoreB := refresh(unitVec(0, 0, 1), "hash-b")
	if hashB == hashA {
		t.Fatal("the stored blend hash did not change with the member set")
	}
	if !(scoreB > scoreA) {
		t.Fatalf("the query scores %.4f against the vector of the new member set and %.4f against the old: the stored vector did not move", scoreB, scoreA)
	}
}

// TestBlend_ShiftsFromNamePriorToMemberCentroid pins the smoothing contract:
// with k members' worth of smoothing, one member barely moves the vector off
// the name prior, while many members pull it onto the centroid.
func TestBlend_ShiftsFromNamePriorToMemberCentroid(t *testing.T) {
	const k = 3.0
	name := []float32{1, 0}
	member := []float32{0, 1}

	noMembers := blend(name, nil, k)
	if noMembers[0] != 1 || noMembers[1] != 0 {
		t.Fatalf("with no members the blend must be the name prior, got %v", noMembers)
	}

	// n=1: centroid weight 1/(1+3) = 0.25, so the name still dominates
	one := blend(name, [][]float32{member}, k)
	if one[0] <= one[1] {
		t.Fatalf("one member should stay name-dominated, got %v", one)
	}
	// n=9: centroid weight 9/(9+3) = 0.75, so the centroid dominates
	many := make([][]float32, 9)
	for i := range many {
		many[i] = member
	}
	nine := blend(name, many, k)
	if nine[1] <= nine[0] {
		t.Fatalf("nine members should be centroid-dominated, got %v", nine)
	}
	// more members must move strictly further toward the centroid
	if nine[1] <= one[1] {
		t.Fatalf("centroid weight did not grow with members: one=%v nine=%v", one, nine)
	}
	for _, v := range [][]float32{noMembers, one, nine} {
		if l := length(v); math.Abs(l-1) > 1e-6 {
			t.Fatalf("blend must return a unit vector, got length %f for %v", l, v)
		}
	}
}

func TestTagCutoff_NoMembersIsTheNameMinimum(t *testing.T) {
	if got := tagCutoff(unitVec(1, 2, 3), nil, 3, 0.46, 0.70); got != 0.46 {
		t.Fatalf("cutoff = %v, want 0.46", got)
	}
}

// Comparing a document against the blended vector with tagCutoff must decide
// exactly like judging the name against nameMin and the members' centroid
// against memberMin, mixed by the blend weight.
func TestTagCutoff_MatchesTheTwoScaleRule(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	const dims, k, a, b = 16, 3.0, 0.46, 0.70
	for n := 1; n <= 6; n++ {
		name := randUnit(r, dims)
		members := make([][]float32, n)
		for i := range members {
			members[i] = randUnit(r, dims)
		}
		vec := blend(name, members, k)
		cut := tagCutoff(name, members, k, a, b)
		c := centroid(members, dims)
		w := float64(n) / (float64(n) + k)
		// queries from far away to right on top of the tag, so both
		// outcomes are exercised
		for step := 0; step <= 200; step++ {
			beta := float64(step) / 200
			noise := randUnit(r, dims)
			mix := make([]float64, dims)
			for i := range mix {
				mix[i] = beta*float64(vec[i]) + (1-beta)*float64(noise[i])
			}
			q := unitVec(mix...)
			margin := (1-w)*(dot(q, name)-a) + w*(dot(q, c)-b)
			if math.Abs(margin) < 1e-5 {
				continue // float32 rounding decides the tie
			}
			if got, want := dot(q, vec) >= cut, margin >= 0; got != want {
				t.Fatalf("n=%d: blended %.5f vs cutoff %.5f says %v, two-scale margin %.5f says %v",
					n, dot(q, vec), cut, got, margin, want)
			}
		}
	}
}

func TestTagCutoff_ApproachesTheMemberMinimumWithManyMembers(t *testing.T) {
	m := unitVec(1, 1, 0) // 45° away from the name
	members := make([][]float32, 1000)
	for i := range members {
		members[i] = m
	}
	if got := tagCutoff(unitVec(1, 0, 0), members, 3, 0.46, 0.70); math.Abs(got-0.70) > 0.01 {
		t.Fatalf("cutoff with 1000 members = %.4f, want about 0.70", got)
	}
}

// TestTagCutoff_OpposedNameAndCentroidMatchNothing: a centroid pointing
// exactly away from the name cancels it out at equal weights, and a blend of
// nothing has no direction any document could score against.
func TestTagCutoff_OpposedNameAndCentroidMatchNothing(t *testing.T) {
	members := [][]float32{unitVec(-1, 0), unitVec(-1, 0), unitVec(-1, 0)}
	if got := tagCutoff(unitVec(1, 0), members, 3, 0.46, 0.70); !math.IsInf(got, 1) {
		t.Fatalf("cutoff = %v, want +Inf", got)
	}
}

// TestHashInputs_StableAcrossMemberOrderAndSensitiveToInputs is the guard
// behind Refresh's "unchanged" short-circuit: the fingerprint must ignore the
// store's member ordering, but change when anything the blend actually
// depends on changes. It sorts a copy: the caller's slice, parallel to its
// member embeddings, keeps its order.
func TestHashInputs_StableAcrossMemberOrderAndSensitiveToInputs(t *testing.T) {
	base := hashInputs("dev / rust", []string{"h1", "h2"}, "m", 3)
	reversed := []string{"h2", "h1"}
	if got := hashInputs("dev / rust", reversed, "m", 3); got != base {
		t.Fatal("fingerprint must not depend on member order")
	}
	if !slices.Equal(reversed, []string{"h2", "h1"}) {
		t.Fatalf("hashInputs reordered the caller's member hashes: %v", reversed)
	}
	for _, tc := range []struct {
		name string
		got  string
	}{
		{"name", hashInputs("dev / go", []string{"h1", "h2"}, "m", 3)},
		{"members", hashInputs("dev / rust", []string{"h1"}, "m", 3)},
		{"model", hashInputs("dev / rust", []string{"h1", "h2"}, "m2", 3)},
		{"smoothing", hashInputs("dev / rust", []string{"h1", "h2"}, "m", 4)},
	} {
		if tc.got == base {
			t.Errorf("changing the %s must change the fingerprint", tc.name)
		}
	}
}

// TestHashInputs_IsStable pins the fingerprint of a known input: a tag
// vector stored under it is skipped by every later refresh, so a changed
// derivation would silently recompute every stored tag vector once. The
// value is what the derivation computed before it moved into this package.
func TestHashInputs_IsStable(t *testing.T) {
	const want = "bfbd28f1e6e17731ff94f0e2f2ff0a64fa38ff0b0a95efbaedb5bc8d539ab7cf"
	if got := hashInputs("dev / rust", []string{"h2", "h1"}, "m", 3); got != want {
		t.Fatalf("hashInputs = %s, want %s", got, want)
	}
}

// length returns the Euclidean length of v.
func length(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func unitVec(v ...float64) []float32 {
	var n float64
	for _, x := range v {
		n += x * x
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x / math.Sqrt(n))
	}
	return out
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func randUnit(r *rand.Rand, dims int) []float32 {
	v := make([]float64, dims)
	for i := range v {
		v[i] = r.NormFloat64()
	}
	return unitVec(v...)
}

// verify the fake actually satisfies the interface
var _ TagStore = (*fakeTagStore)(nil)

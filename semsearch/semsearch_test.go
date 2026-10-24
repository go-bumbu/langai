package semsearch

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"
)

func newTestIndex(t *testing.T) (*Index, *MemStore) {
	t.Helper()
	store := NewMemStore()
	ix, err := New(FakeEmbedder{}, store, "t", "p", "c")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return ix, store
}

func TestIndexAndSearch(t *testing.T) {
	ix, _ := newTestIndex(t)
	ctx := context.Background()

	docs := map[string]string{
		"cooking": "# Cooking\n\nRecipes for pasta, pizza and fresh tomato sauce.",
		"golang":  "# Go\n\nNotes about the Go programming language, goroutines and channels.",
		"hiking":  "# Hiking\n\nTrails, mountain routes and camping equipment.",
	}
	for id, text := range docs {
		if err := ix.Index(ctx, id, text); err != nil {
			t.Fatalf("Index(%s): %v", id, err)
		}
	}

	got, err := ix.Search(ctx, "goroutines and channels in the Go language", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("expected results")
	}
	if got[0].DocID != "golang" {
		t.Fatalf("expected golang first, got %q (score %f)", got[0].DocID, got[0].Score)
	}
	if got[0].Score <= 0 {
		t.Errorf("expected positive score, got %f", got[0].Score)
	}
	if got[0].Text == "" {
		t.Error("expected the matching chunk text as snippet")
	}
}

func TestSearchLimitAndDedupe(t *testing.T) {
	ix, _ := newTestIndex(t)
	ctx := context.Background()

	// one doc with many chunks matching the query: results must dedupe to one
	long := "# Dogs\n\n## A\n\ndogs bark loudly\n\n## B\n\ndogs sleep a lot\n\n## C\n\ndogs eat food"
	if err := ix.Index(ctx, "dogs", long); err != nil {
		t.Fatalf("Index: %v", err)
	}
	if err := ix.Index(ctx, "cats", "# Cats\n\ncats purr and dogs bark"); err != nil {
		t.Fatalf("Index: %v", err)
	}

	got, err := ix.Search(ctx, "dogs", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	seen := map[string]int{}
	for _, m := range got {
		seen[m.DocID]++
	}
	if seen["dogs"] != 1 {
		t.Fatalf("expected exactly one match for doc 'dogs', got %d", seen["dogs"])
	}

	got, err = ix.Search(ctx, "dogs", 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected limit to cap results at 1, got %d", len(got))
	}
}

func TestIndexUnchangedContentSkipsEmbedding(t *testing.T) {
	store := NewMemStore()
	emb := &countingEmbedder{}
	ix, err := New(emb, store, "t", "p", "c")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx := context.Background()

	if err := ix.Index(ctx, "doc", "same text"); err != nil {
		t.Fatalf("Index: %v", err)
	}
	first := emb.calls
	if first == 0 {
		t.Fatal("expected the first index to embed")
	}
	if err := ix.Index(ctx, "doc", "same text"); err != nil {
		t.Fatalf("Index: %v", err)
	}
	if emb.calls != first {
		t.Fatalf("expected unchanged content to skip embedding, calls %d -> %d", first, emb.calls)
	}
	if err := ix.Index(ctx, "doc", "different text"); err != nil {
		t.Fatalf("Index: %v", err)
	}
	if emb.calls == first {
		t.Fatal("expected changed content to re-embed")
	}
}

func TestIndexUpdateReplacesChunks(t *testing.T) {
	ix, _ := newTestIndex(t)
	ctx := context.Background()

	if err := ix.Index(ctx, "doc", "bananas are yellow"); err != nil {
		t.Fatalf("Index: %v", err)
	}
	if err := ix.Index(ctx, "doc", "submarines are underwater"); err != nil {
		t.Fatalf("Index: %v", err)
	}

	got, err := ix.Search(ctx, "bananas", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, m := range got {
		if m.DocID == "doc" && m.Score > 0.5 {
			t.Fatalf("old content still strongly matches after update: %#v", m)
		}
	}
}

func TestDeleteRemovesDocument(t *testing.T) {
	ix, _ := newTestIndex(t)
	ctx := context.Background()

	if err := ix.Index(ctx, "doc", "unique zebra content"); err != nil {
		t.Fatalf("Index: %v", err)
	}
	if err := ix.Delete(ctx, "doc"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	got, err := ix.Search(ctx, "zebra", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, m := range got {
		if m.DocID == "doc" {
			t.Fatalf("deleted document still found: %#v", m)
		}
	}
	// deleting again is a no-op
	if err := ix.Delete(ctx, "doc"); err != nil {
		t.Fatalf("Delete (again): %v", err)
	}
}

func TestSearchEmptyQuery(t *testing.T) {
	ix, _ := newTestIndex(t)
	got, err := ix.Search(context.Background(), "   ", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no results for empty query, got %#v", got)
	}
}

// vectorCountEmbedder breaks the Embedder contract on purpose: it answers
// every call with n vectors, however many texts it was given.
type vectorCountEmbedder struct{ n int }

func (v vectorCountEmbedder) Dimensions() int { return 8 }

func (v vectorCountEmbedder) Embed(_ context.Context, _ Role, _ []string) ([][]float32, error) {
	out := make([][]float32, v.n)
	for i := range out {
		out[i] = []float32{1, 0, 0, 0, 0, 0, 0, 0}
	}
	return out, nil
}

// TestSearchRejectsWrongVectorCount: an embedder answering a one-text query
// with no vector, or with several, must surface as an error, not as a panic
// on the missing vector or a silent pick of the first one.
func TestSearchRejectsWrongVectorCount(t *testing.T) {
	for _, n := range []int{0, 2} {
		t.Run(fmt.Sprintf("%d vectors", n), func(t *testing.T) {
			ix, err := New(vectorCountEmbedder{n: n}, NewMemStore(), "t", "p", "c")
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := ix.Search(context.Background(), "query", 5); err == nil {
				t.Fatalf("Search accepted %d vectors for one query", n)
			}
		})
	}
}

func TestIndexEmptyDocID(t *testing.T) {
	ix, _ := newTestIndex(t)
	if err := ix.Index(context.Background(), "", "text"); err == nil {
		t.Fatal("expected error for empty doc id")
	}
	if err := ix.Delete(context.Background(), ""); err == nil {
		t.Fatal("expected error for empty doc id")
	}
}

func TestSearchMinScore(t *testing.T) {
	store := NewMemStore()
	ctx := context.Background()

	docs := map[string]string{
		"cooking": "# Cooking\n\nRecipes for pasta, pizza and fresh tomato sauce.",
		"golang":  "# Go\n\nNotes about the Go programming language, goroutines and channels.",
		"hiking":  "# Hiking\n\nTrails, mountain routes and camping equipment.",
	}
	ix, err := New(FakeEmbedder{}, store, "t", "p", "c")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for id, text := range docs {
		if err := ix.Index(ctx, id, text); err != nil {
			t.Fatalf("Index(%s): %v", id, err)
		}
	}

	// no floor by default: a nearest-neighbour search ranks everything it has,
	// however weak the match
	all, err := ix.Search(ctx, "goroutines and channels in the Go language", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(all) != len(docs) {
		t.Fatalf("expected every document ranked without a floor, got %d of %d", len(all), len(docs))
	}

	// a floor between the first and second hit must cut the list there
	floored, err := newFlooredIndex(t, store, (all[0].Score+all[1].Score)/2)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err := floored.Search(ctx, "goroutines and channels in the Go language", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected the floor to keep only the best hit, got %s", hitsOf(got))
	}
	if got[0].DocID != all[0].DocID {
		t.Fatalf("expected %q to survive the floor, got %q", all[0].DocID, got[0].DocID)
	}

	// the floor is inclusive: a hit scoring exactly the minimum is kept
	floored, err = newFlooredIndex(t, store, all[0].Score)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err = floored.Search(ctx, "goroutines and channels in the Go language", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 || got[0].DocID != all[0].DocID {
		t.Fatalf("expected the best hit to survive its own score as floor, got %s", hitsOf(got))
	}

	// above every score the search legitimately answers with nothing
	floored, err = newFlooredIndex(t, store, all[0].Score+0.01)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err = floored.Search(ctx, "goroutines and channels in the Go language", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no results above the best score, got %s", hitsOf(got))
	}
}

func newFlooredIndex(t *testing.T, store Store, min float64) (*Index, error) {
	t.Helper()
	return New(FakeEmbedder{}, store, "t", "p", "c", WithMinScore(min))
}

func hitsOf(matches []Match) string {
	parts := make([]string, 0, len(matches))
	for _, m := range matches {
		parts = append(parts, fmt.Sprintf("%s(%.4f)", m.DocID, m.Score))
	}
	if len(parts) == 0 {
		return "(nothing)"
	}
	return strings.Join(parts, ", ")
}

// countingEmbedder counts Embed calls (to observe change detection) and
// records the role of each one (to observe what the index asks for).
type countingEmbedder struct {
	calls int
	roles []Role
}

func (c *countingEmbedder) Dimensions() int { return 8 }

func (c *countingEmbedder) Embed(ctx context.Context, role Role, texts []string) ([][]float32, error) {
	c.calls++
	c.roles = append(c.roles, role)
	return FakeEmbedder{Dims: 8}.Embed(ctx, role, texts)
}

// TestIndexPassesRolesToEmbedder: indexed text is a document, a search is a
// query. Asymmetric models prefix the two differently, so the index must say
// which is which.
func TestIndexPassesRolesToEmbedder(t *testing.T) {
	emb := &countingEmbedder{}
	ix, err := New(emb, NewMemStore(), "t", "p", "c")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx := context.Background()
	if err := ix.Index(ctx, "doc", "some text"); err != nil {
		t.Fatalf("Index: %v", err)
	}
	if _, err := ix.Search(ctx, "some", 5); err != nil {
		t.Fatalf("Search: %v", err)
	}
	want := []Role{RoleDocument, RoleQuery}
	if len(emb.roles) != len(want) {
		t.Fatalf("roles = %v, want %v", emb.roles, want)
	}
	for i := range want {
		if emb.roles[i] != want[i] {
			t.Fatalf("roles = %v, want %v", emb.roles, want)
		}
	}
}

// TestNormalize_ScalesToUnitLength: the vector keeps its direction and comes
// back at length 1, so cosine and dot product rank alike.
func TestNormalize_ScalesToUnitLength(t *testing.T) {
	v := []float32{3, 4}
	Normalize(v)
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Fatalf("Normalize([3 4]) = %v, want [0.6 0.8]", v)
	}
}

// TestNormalize_ZeroVectorStaysZero covers the divide-by-zero guard: an
// all-zero vector has no direction to scale, so it must be left alone rather
// than turned into NaNs.
func TestNormalize_ZeroVectorStaysZero(t *testing.T) {
	v := []float32{0, 0, 0}
	Normalize(v)
	for i, x := range v {
		if x != 0 {
			t.Fatalf("zero vector was modified at %d: %v", i, v)
		}
	}
}

func TestRoleString(t *testing.T) {
	if RoleDocument.String() != "document" || RoleQuery.String() != "query" || Role(7).String() != "Role(7)" {
		t.Fatalf("Role strings: %s %s %s", RoleDocument, RoleQuery, Role(7))
	}
}

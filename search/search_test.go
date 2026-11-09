package search

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/vector"
)

func newTestIndex(t *testing.T) (*Index, *vector.MemStore) {
	t.Helper()
	store := vector.NewMemStore()
	ix, err := New(langaitest.FakeEmbedder{}, store, "t", "p", "c")
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
	store := vector.NewMemStore()
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

func (v vectorCountEmbedder) Embed(_ context.Context, _ langai.Role, _ []string) ([][]float32, error) {
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
			ix, err := New(vectorCountEmbedder{n: n}, vector.NewMemStore(), "t", "p", "c")
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
	store := vector.NewMemStore()
	ctx := context.Background()

	docs := map[string]string{
		"cooking": "# Cooking\n\nRecipes for pasta, pizza and fresh tomato sauce.",
		"golang":  "# Go\n\nNotes about the Go programming language, goroutines and channels.",
		"hiking":  "# Hiking\n\nTrails, mountain routes and camping equipment.",
	}
	ix, err := New(langaitest.FakeEmbedder{}, store, "t", "p", "c")
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

func newFlooredIndex(t *testing.T, store vector.Store, min float64) (*Index, error) {
	t.Helper()
	return New(langaitest.FakeEmbedder{}, store, "t", "p", "c", WithMinScore(min))
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
	roles []langai.Role
}

func (c *countingEmbedder) Dimensions() int { return 8 }

func (c *countingEmbedder) Embed(ctx context.Context, role langai.Role, texts []string) ([][]float32, error) {
	c.calls++
	c.roles = append(c.roles, role)
	return langaitest.FakeEmbedder{Dims: 8}.Embed(ctx, role, texts)
}

// TestIndexPassesRolesToEmbedder: indexed text is a document, a search is a
// query. Asymmetric models prefix the two differently, so the index must say
// which is which.
func TestIndexPassesRolesToEmbedder(t *testing.T) {
	emb := &countingEmbedder{}
	ix, err := New(emb, vector.NewMemStore(), "t", "p", "c")
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
	want := []langai.Role{langai.RoleDocument, langai.RoleQuery}
	if len(emb.roles) != len(want) {
		t.Fatalf("roles = %v, want %v", emb.roles, want)
	}
	for i := range want {
		if emb.roles[i] != want[i] {
			t.Fatalf("roles = %v, want %v", emb.roles, want)
		}
	}
}

// TestIndex_SearchIndexedLeavesItselfOut: "more like this" searches with the
// document's stored chunks, runs no model, never answers the document
// itself, and names an id with nothing stored.
func TestIndex_SearchIndexedLeavesItselfOut(t *testing.T) {
	emb := &countingEmbedder{}
	ix, err := New(emb, vector.NewMemStore(), "t", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	for id, text := range map[string]string{
		"golang":  "# Go\n\nNotes about the Go programming language.\n\n## Concurrency\n\ngoroutines and channels",
		"go-copy": "# Go notes\n\nthe Go programming language with goroutines and channels",
		"cooking": "# Cooking\n\nRecipes for pasta, pizza and fresh tomato sauce.",
	} {
		if err := ix.Index(ctx, id, text); err != nil {
			t.Fatal(err)
		}
	}
	calls := emb.calls
	got, err := ix.SearchIndexed(ctx, "golang", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].DocID != "go-copy" || got[0].Text == "" {
		t.Fatalf("SearchIndexed = %s, want go-copy first and golang left out", hitsOf(got))
	}
	for _, m := range got {
		if m.DocID == "golang" {
			t.Errorf("the query document came back: %s", hitsOf(got))
		}
	}
	if emb.calls != calls {
		t.Errorf("SearchIndexed ran the embedder %d times, want none", emb.calls-calls)
	}
	if _, err := ix.SearchIndexed(ctx, "missing", 10); !errors.Is(err, ErrNotIndexed) {
		t.Errorf("unknown id: %v, want ErrNotIndexed", err)
	}
	if _, err := ix.SearchIndexed(ctx, "", 10); !errors.Is(err, ErrEmptyID) {
		t.Errorf("empty id: %v, want ErrEmptyID", err)
	}
}

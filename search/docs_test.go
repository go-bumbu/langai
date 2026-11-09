package search

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/go-bumbu/sense/vector"
)

// idsOf collects the ids Docs lists, failing the test on an error.
func idsOf(t *testing.T, seq func(func(vector.DocInfo, error) bool)) []string {
	t.Helper()
	var ids []string
	for d, err := range seq {
		if err != nil {
			t.Fatal(err)
		}
		if d.Hash == "" {
			t.Errorf("%q listed without its hash", d.ID)
		}
		ids = append(ids, d.ID)
	}
	return ids
}

// TestIndex_DocsAndDelete: Docs lists what is indexed, by id, and Delete
// removes several ids at once; an empty id among them deletes nothing.
func TestIndex_DocsAndDelete(t *testing.T) {
	ix, _ := newTestIndex(t)
	ctx := context.Background()
	if _, err := ix.IndexBatch(ctx, []Doc{{"c", "cooking pasta"}, {"a", "go channels"}, {"b", "hiking trails"}}); err != nil {
		t.Fatal(err)
	}
	if got := idsOf(t, ix.Docs(ctx)); !slices.Equal(got, []string{"a", "b", "c"}) {
		t.Fatalf("Docs = %v, want a b c", got)
	}
	if err := ix.Delete(ctx, "a", "", "c"); !errors.Is(err, ErrEmptyID) {
		t.Errorf("Delete with an empty id: %v, want ErrEmptyID", err)
	}
	if err := ix.Delete(ctx, "a", "c", "never-indexed"); err != nil {
		t.Fatal(err)
	}
	if got := idsOf(t, ix.Docs(ctx)); !slices.Equal(got, []string{"b"}) {
		t.Errorf("Docs after Delete = %v, want b", got)
	}
	if err := ix.Delete(ctx); err != nil {
		t.Errorf("Delete of nothing: %v", err)
	}
}

// TestMediaIndex_DocsAndDelete: the same for the media index.
func TestMediaIndex_DocsAndDelete(t *testing.T) {
	ix, _, _ := newMediaIndex(t)
	ctx := context.Background()
	for _, id := range []string{"y", "x", "z"} {
		if err := ix.IndexImage(ctx, id, bytes.NewReader(encoded(t, pic(16, 16, red, blue)))); err != nil {
			t.Fatal(err)
		}
	}
	if got := idsOf(t, ix.Docs(ctx)); !slices.Equal(got, []string{"x", "y", "z"}) {
		t.Fatalf("Docs = %v, want x y z", got)
	}
	if err := ix.Delete(ctx, "x", "z"); err != nil {
		t.Fatal(err)
	}
	if got := idsOf(t, ix.Docs(ctx)); !slices.Equal(got, []string{"y"}) {
		t.Errorf("Docs after Delete = %v, want y", got)
	}
}

// failingDocs is a store whose Docs fails after the first page.
type failingDocs struct {
	vector.Store
	calls int
}

func (f *failingDocs) Docs(ctx context.Context, c vector.Coord, after string, limit int) ([]vector.DocInfo, error) {
	f.calls++
	if f.calls > 1 {
		return nil, errors.New("connection reset")
	}
	return f.Store.Docs(ctx, c, after, limit)
}

// TestListDocs_PagesStopsAndFails: the listing pages through more than one
// page of the store, stops reading when the loop breaks, and ends with the
// store's error, prefixed.
func TestListDocs_PagesStopsAndFails(t *testing.T) {
	ctx := context.Background()
	store := vector.NewMemStore()
	c := vector.Coord{Index: "t", Partition: "p", Collection: "c"}
	n := 2*docsPage + 1
	docs := make([]vector.Doc, n)
	for i := range docs {
		docs[i] = vector.Doc{ID: fmt.Sprintf("d%05d", i), Hash: "h"}
	}
	if err := store.Upsert(ctx, c, "", docs...); err != nil {
		t.Fatal(err)
	}
	if got := idsOf(t, listDocs(ctx, store, c)); len(got) != n || !slices.IsSorted(got) {
		t.Fatalf("listed %d documents (sorted %v), want %d", len(got), slices.IsSorted(got), n)
	}

	counting := &failingDocs{Store: store}
	for range listDocs(ctx, counting, c) {
		break
	}
	if counting.calls != 1 {
		t.Errorf("a loop that broke at once read %d pages, want 1", counting.calls)
	}

	counting.calls = 0
	var listed int
	var last error
	for _, err := range listDocs(ctx, counting, c) {
		if err != nil {
			last = err
			break
		}
		listed++
	}
	if listed != docsPage || last == nil || last.Error() != "search: list documents: connection reset" {
		t.Errorf("listed %d then %v, want a page then the prefixed store error", listed, last)
	}
}

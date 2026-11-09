package search

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/vector"
)

// countingStore counts the calls IndexBatch makes and the documents of the
// largest Upsert, and can fail the n-th Upsert.
type countingStore struct {
	vector.Store
	hashes, upserts, maxDocs int
	failUpsert               int // fail the n-th Upsert (1-based); 0 never
}

func (s *countingStore) Hashes(ctx context.Context, c vector.Coord, ids ...string) (map[string]string, error) {
	s.hashes++
	return s.Store.Hashes(ctx, c, ids...)
}

func (s *countingStore) Upsert(ctx context.Context, c vector.Coord, model string, docs ...vector.Doc) error {
	s.upserts++
	s.maxDocs = max(s.maxDocs, len(docs))
	if s.upserts == s.failUpsert {
		return errors.New("disk full")
	}
	return s.Store.Upsert(ctx, c, model, docs...)
}

// batchEmbedder records the size of every Embed call and can fail the n-th.
type batchEmbedder struct {
	sensetest.FakeEmbedder
	sizes []int
	fail  int // fail the n-th call (1-based); 0 never
}

func (b *batchEmbedder) Embed(ctx context.Context, role sense.Role, texts []string) ([][]float32, error) {
	b.sizes = append(b.sizes, len(texts))
	if len(b.sizes) == b.fail {
		return nil, errors.New("model down")
	}
	return b.FakeEmbedder.Embed(ctx, role, texts)
}

// corpus is n small documents of one chunk each, about distinct topics.
func corpus(n int) []Doc {
	docs := make([]Doc, n)
	for i := range docs {
		docs[i] = Doc{ID: fmt.Sprintf("doc%03d", i), Text: fmt.Sprintf("note %d about topic%d and word%d", i, i, i*7)}
	}
	return docs
}

func newBatchIndex(t *testing.T) (*Index, *countingStore, *batchEmbedder) {
	t.Helper()
	store := &countingStore{Store: vector.NewMemStore()}
	emb := &batchEmbedder{}
	ix, err := New(emb, store, "t", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	return ix, store, emb
}

// TestIndexBatch_BoundedCalls: a batch reads every hash in one call, never
// hands the embedder more than embedBatch texts nor the store more than
// embedBatch documents at once, and a second run of the same documents
// embeds and writes nothing.
func TestIndexBatch_BoundedCalls(t *testing.T) {
	ix, store, emb := newBatchIndex(t)
	ctx := context.Background()
	docs := corpus(100)
	// one long document among them, of several embed batches by itself
	docs[50].Text = strings.Repeat("a paragraph of some length about something.\n\n", 40*embedBatch)
	res, err := ix.IndexBatch(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Indexed) != 100 || len(res.Unchanged) != 0 || len(res.Failed) != 0 {
		t.Fatalf("result = %d indexed, %d unchanged, %v failed; want 100 indexed", len(res.Indexed), len(res.Unchanged), res.Failed)
	}
	if store.hashes != 1 {
		t.Errorf("%d Hashes calls, want 1", store.hashes)
	}
	if store.upserts < 2 || store.maxDocs > embedBatch {
		t.Errorf("%d Upserts of at most %d documents, want several of at most %d", store.upserts, store.maxDocs, embedBatch)
	}
	for _, n := range emb.sizes {
		if n > embedBatch {
			t.Errorf("an Embed call of %d texts, want at most %d", n, embedBatch)
		}
	}
	got, err := ix.Search(ctx, docs[42].Text, 1)
	if err != nil || len(got) != 1 || got[0].DocID != "doc042" {
		t.Errorf("Search after the batch: %s, %v", hitsOf(got), err)
	}

	calls, upserts := len(emb.sizes), store.upserts
	res, err = ix.IndexBatch(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Unchanged) != 100 || len(res.Indexed) != 0 {
		t.Errorf("second run: %d unchanged, %d indexed; want all unchanged", len(res.Unchanged), len(res.Indexed))
	}
	if len(emb.sizes) != calls || store.upserts != upserts || store.hashes != 2 {
		t.Errorf("second run: %d embed calls, %d upserts, %d hash reads; want none, none, one", len(emb.sizes)-calls, store.upserts-upserts, store.hashes-1)
	}
}

// TestIndexBatch_DocumentFailuresDoNotStopTheBatch: a document refused for
// a reason of its own is reported by id with its error, and the others are
// stored.
func TestIndexBatch_DocumentFailuresDoNotStopTheBatch(t *testing.T) {
	ix, store, _ := newBatchIndex(t)
	ctx := context.Background()
	docs := corpus(5)
	docs[2].Text = "!!! ???" // embeds to the zero vector
	res, err := ix.IndexBatch(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Indexed) != 4 || slices.Contains(res.Indexed, "doc002") {
		t.Errorf("indexed = %v, want the four others", res.Indexed)
	}
	if fe := res.Failed["doc002"]; !errors.Is(fe, vector.ErrInvalidVector) || !strings.Contains(fe.Error(), `"doc002"`) {
		t.Errorf("failed = %v, want doc002 with ErrInvalidVector naming it", res.Failed)
	}
	stored, _ := store.Hashes(ctx, ix.coord, "doc002", "doc003")
	if _, ok := stored["doc002"]; ok || stored["doc003"] == "" {
		t.Errorf("stored hashes = %v, want doc003 and not doc002", stored)
	}
}

// TestIndexBatch_InvalidIDFailsAlone: a document whose id no store can keep
// (a Latin-1 file path, a NUL byte) goes to Failed with vector.ErrInvalidID,
// costs no model time, and the rest of the batch is stored.
func TestIndexBatch_InvalidIDFailsAlone(t *testing.T) {
	ix, store, emb := newBatchIndex(t)
	ctx := context.Background()
	docs := corpus(100)
	docs[17].ID = "caf\xe9.txt"
	docs[60].ID = "a\x00b"
	res, err := ix.IndexBatch(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Indexed) != 98 {
		t.Errorf("indexed %d, want the 98 others", len(res.Indexed))
	}
	for _, bad := range []string{docs[17].ID, docs[60].ID} {
		if fe := res.Failed[bad]; !errors.Is(fe, vector.ErrInvalidID) || !strings.HasPrefix(fe.Error(), "search: ") {
			t.Errorf("Failed[%q] = %v, want a search: error wrapping vector.ErrInvalidID", bad, fe)
		}
	}
	embedded := 0
	for _, n := range emb.sizes {
		embedded += n
	}
	if embedded != 98 {
		t.Errorf("embedded %d texts, want only the 98 valid documents'", embedded)
	}
	if list, err := store.Docs(ctx, ix.coord, "", 0); err != nil || len(list) != 98 {
		t.Errorf("stored %d documents, %v; want 98", len(list), err)
	}
}

// TestIndexBatch_StopsAtABatchWideError: an embedder or store failure stops
// the batch; Indexed is exactly what was stored before it, and running the
// batch again finishes the rest.
func TestIndexBatch_StopsAtABatchWideError(t *testing.T) {
	for _, tc := range []struct {
		name     string
		setup    func(*countingStore, *batchEmbedder)
		wantText string
	}{
		{"embedder", func(_ *countingStore, e *batchEmbedder) { e.fail = 2 }, "model down"},
		{"store", func(s *countingStore, _ *batchEmbedder) { s.failUpsert = 2 }, "disk full"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ix, store, emb := newBatchIndex(t)
			ctx := context.Background()
			tc.setup(store, emb)
			docs := corpus(3 * embedBatch)
			res, err := ix.IndexBatch(ctx, docs)
			if err == nil || !strings.Contains(err.Error(), tc.wantText) || !strings.HasPrefix(err.Error(), "search: ") {
				t.Fatalf("err = %v, want the %s failure", err, tc.name)
			}
			if len(res.Indexed) != embedBatch {
				t.Errorf("indexed %d before the failure, want the first batch of %d", len(res.Indexed), embedBatch)
			}
			stored, err := store.Docs(ctx, ix.coord, "", 0)
			if err != nil {
				t.Fatal(err)
			}
			if len(stored) != len(res.Indexed) {
				t.Errorf("%d documents stored, %d reported indexed", len(stored), len(res.Indexed))
			}
			emb.fail, store.failUpsert = 0, 0
			res, err = ix.IndexBatch(ctx, docs)
			if err != nil || len(res.Unchanged) != embedBatch || len(res.Indexed) != 2*embedBatch {
				t.Errorf("resume: %d unchanged, %d indexed, %v; want the rest indexed", len(res.Unchanged), len(res.Indexed), err)
			}
		})
	}
}

// TestIndexBatch_ArgumentErrors: an empty or repeated id fails the whole
// call before anything is read or written.
func TestIndexBatch_ArgumentErrors(t *testing.T) {
	ix, store, emb := newBatchIndex(t)
	ctx := context.Background()
	if _, err := ix.IndexBatch(ctx, []Doc{{ID: "a", Text: "x"}, {Text: "y"}}); !errors.Is(err, ErrEmptyID) {
		t.Errorf("empty id: %v, want ErrEmptyID", err)
	}
	if _, err := ix.IndexBatch(ctx, []Doc{{ID: "a", Text: "x"}, {ID: "a", Text: "y"}}); err == nil || !strings.Contains(err.Error(), `"a"`) {
		t.Errorf("repeated id: %v, want an error naming it", err)
	}
	if store.hashes != 0 || len(emb.sizes) != 0 || store.upserts != 0 {
		t.Errorf("a refused batch made %d hash reads, %d embed calls, %d writes", store.hashes, len(emb.sizes), store.upserts)
	}
	if res, err := ix.IndexBatch(ctx, nil); err != nil || len(res.Indexed)+len(res.Unchanged)+len(res.Failed) != 0 {
		t.Errorf("no documents: %+v, %v", res, err)
	}
}

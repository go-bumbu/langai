package search

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image/color"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/media"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/vector"
)

// TestIndex_EmptyPartitionIsNotAWildcard: an Index over an empty partition
// sees that partition's documents only, never another tenant's.
func TestIndex_EmptyPartitionIsNotAWildcard(t *testing.T) {
	ctx := context.Background()
	store := vector.NewMemStore()
	emb := sensetest.FakeEmbedder{}
	for _, part := range []string{"", "tenant-a", "tenant-b"} {
		ix, err := New(emb, store, "i", part, "notes")
		if err != nil {
			t.Fatal(err)
		}
		if err := ix.Index(ctx, "doc-"+part, "goroutines and channels"); err != nil {
			t.Fatal(err)
		}
	}
	ix, err := New(emb, store, "i", "", "notes")
	if err != nil {
		t.Fatal(err)
	}
	got, err := ix.Search(ctx, "goroutines", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].DocID != "doc-" {
		t.Fatalf("search over the empty partition = %s, want only its own document", hitsOf(got))
	}
}

// TestIndex_ZeroVectorNamesTheDocument: a text that embeds to the zero
// vector (no letters or digits, for the fake) is refused at indexing, by
// name, instead of breaking every later search of its scope.
func TestIndex_ZeroVectorNamesTheDocument(t *testing.T) {
	ix, _ := newTestIndex(t)
	err := ix.Index(context.Background(), "punctuation", "!!! ???")
	if !errors.Is(err, vector.ErrInvalidVector) || !strings.Contains(err.Error(), `"punctuation"`) {
		t.Fatalf("Index of a zero vector: %v, want ErrInvalidVector naming the document", err)
	}
}

// TestAboveFloor: the floor keeps the leading matches that reach it, and a
// NaN score, which compares false with everything, never passes it.
func TestAboveFloor(t *testing.T) {
	m := func(score float64) vector.Match { return vector.Match{Score: score} }
	nan := math.NaN()
	for _, tc := range []struct {
		name string
		raw  []vector.Match
		min  float64
		want int
	}{
		{"all above", []vector.Match{m(0.9), m(0.5)}, 0.5, 2},
		{"cut", []vector.Match{m(0.9), m(0.4), m(0.3)}, 0.5, 1},
		{"no floor", []vector.Match{m(0.9), m(-1)}, math.Inf(-1), 2},
		{"NaN first", []vector.Match{m(nan), m(0.9)}, 0.5, 0},
		{"NaN without a floor", []vector.Match{m(0.9), m(nan)}, math.Inf(-1), 1},
	} {
		if got := aboveFloor(tc.raw, tc.min); len(got) != tc.want {
			t.Errorf("%s: kept %d, want %d", tc.name, len(got), tc.want)
		}
	}
}

// namedEmbedder is the fake text embedder under another model name.
type namedEmbedder struct {
	sensetest.FakeEmbedder
	model string
}

func (e namedEmbedder) Model() string { return e.model }

var _ sense.Identified = namedEmbedder{}

// TestIndex_RefusesAnotherModel: the embedder's model is recorded with the
// collection's first vectors, so another model of the same dimensions can
// neither write to the collection nor search it.
func TestIndex_RefusesAnotherModel(t *testing.T) {
	ctx := context.Background()
	store := vector.NewMemStore()
	first, err := New(sensetest.FakeEmbedder{}, store, "i", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Index(ctx, "doc", "goroutines and channels"); err != nil {
		t.Fatal(err)
	}
	second, err := New(namedEmbedder{model: "another"}, store, "i", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	if err := second.Index(ctx, "doc2", "pasta and pizza"); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("Index with another model: %v, want ErrModelMismatch", err)
	}
	if _, err := second.Search(ctx, "goroutines", 10); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("Search with another model: %v, want ErrModelMismatch", err)
	}
	// an embedder that names no model is not checked
	anonymous, err := New(sensetest.FakeEmbedder{}, store, "i", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	if got, err := anonymous.Search(ctx, "goroutines", 10); err != nil || len(got) != 1 {
		t.Errorf("Search with the same model: %s, %v", hitsOf(got), err)
	}
}

// renamedText is a counting text embedder under another model name.
type renamedText struct {
	*batchEmbedder
	model string
}

func (e renamedText) Model() string { return e.model }

// renamedImages is a counting image embedder under another model name.
type renamedImages struct {
	*countingImageEmbedder
	model string
}

func (e renamedImages) Model() string { return e.model }

// TestIndex_ModelChangeMigratesThroughReset: after the embedder's model
// changes, indexing fails with ErrModelMismatch before the new model embeds
// anything, and Reset empties the collection so the next run rebuilds it
// with the new model.
func TestIndex_ModelChangeMigratesThroughReset(t *testing.T) {
	ctx := context.Background()
	store := vector.NewMemStore()
	old, err := New(sensetest.FakeEmbedder{}, store, "i", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	docs := corpus(5)
	if _, err := old.IndexBatch(ctx, docs); err != nil {
		t.Fatal(err)
	}
	emb := renamedText{batchEmbedder: &batchEmbedder{}, model: "next"}
	next, err := New(emb, store, "i", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	res, err := next.IndexBatch(ctx, docs)
	if !errors.Is(err, vector.ErrModelMismatch) || !strings.Contains(err.Error(), "Reset") {
		t.Fatalf("IndexBatch with a new model: %+v, %v; want ErrModelMismatch naming Reset", res, err)
	}
	if len(emb.sizes) != 0 {
		t.Fatalf("the new model embedded %v before the store refused it", emb.sizes)
	}
	if err := next.Index(ctx, "doc000", docs[0].Text); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("Index with a new model: %v, want ErrModelMismatch", err)
	}
	if err := next.Reset(ctx); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if res, err := next.IndexBatch(ctx, docs); err != nil || len(res.Indexed) != len(docs) {
		t.Fatalf("IndexBatch after Reset: %+v, %v; want every document indexed", res, err)
	}
	if got, err := next.Search(ctx, docs[3].Text, 1); err != nil || len(got) != 1 || got[0].DocID != "doc003" {
		t.Errorf("Search after the migration = %s, %v", hitsOf(got), err)
	}
	if _, err := old.Search(ctx, docs[3].Text, 1); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("Search with the old model after the migration: %v, want ErrModelMismatch", err)
	}
}

// TestMediaIndex_ModelChangeMigratesThroughReset: as for Index, an image
// model change fails before the new model runs, and Reset lets the next run
// rebuild the collection.
func TestMediaIndex_ModelChangeMigratesThroughReset(t *testing.T) {
	ctx := context.Background()
	old, _, store := newMediaIndex(t, WithSampler(&countingSampler{frames: clipFrames()}))
	img := encoded(t, pic(8, 8, red, blue))
	if err := old.IndexFrames(ctx, "clip", "h", clipFrames()); err != nil {
		t.Fatal(err)
	}
	emb := renamedImages{countingImageEmbedder: &countingImageEmbedder{}, model: "next"}
	next, err := NewMediaIndex(emb, store, "t", "p", "c", WithSampler(&countingSampler{frames: clipFrames()}))
	if err != nil {
		t.Fatal(err)
	}
	video := tempFile(t, "clip.mp4", []byte("not really a video"))
	for name, call := range map[string]func() error{
		"IndexImage":  func() error { return next.IndexImage(ctx, "photo", bytes.NewReader(img)) },
		"IndexVideo":  func() error { return next.IndexVideo(ctx, "video", video) },
		"IndexFrames": func() error { return next.IndexFrames(ctx, "clip", "h", clipFrames()) },
	} {
		if err := call(); !errors.Is(err, vector.ErrModelMismatch) {
			t.Errorf("%s with a new model: %v, want ErrModelMismatch", name, err)
		}
	}
	if n := emb.calls.Load(); n != 0 {
		t.Fatalf("the new model embedded %d batches before the store refused it", n)
	}
	if err := next.Reset(ctx); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if err := next.IndexImage(ctx, "photo", bytes.NewReader(img)); err != nil {
		t.Fatalf("IndexImage after Reset: %v", err)
	}
	for d, err := range next.Docs(ctx) {
		if err != nil || d.ID != "photo" {
			t.Errorf("after Reset and one image, Docs lists %v, %v", d.ID, err)
		}
	}
}

// TestMediaIndex_ManyFramesDoNotCrowdOthers: a near-static video of 64
// frames, all close to the query, still leaves the other items their places
// in the limit.
func TestMediaIndex_ManyFramesDoNotCrowdOthers(t *testing.T) {
	ix, _, _ := newMediaIndex(t)
	ctx := context.Background()
	frames := make([]media.Frame, 64)
	for i := range frames {
		frames[i] = media.Frame{Image: pic(64, 48, red, blue), At: time.Duration(i) * time.Second}
	}
	if err := ix.IndexFrames(ctx, "video", "h", frames); err != nil {
		t.Fatal(err)
	}
	for i := range 12 {
		c := color.RGBA{uint8(190 - 10*i), 30, uint8(40 + 10*i), 255}
		if err := ix.IndexFrames(ctx, fmt.Sprintf("img%02d", i), "h", []media.Frame{{Image: pic(64, 48, c, blue)}}); err != nil {
			t.Fatal(err)
		}
	}
	hits, err := ix.SearchFrames(ctx, []media.Frame{{Image: pic(64, 48, red, blue)}}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 10 || hits[0].DocID != "video" {
		t.Fatalf("hits = %+v, want 10 with the video first", hits)
	}
}

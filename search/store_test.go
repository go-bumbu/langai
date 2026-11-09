package search

import (
	"context"
	"errors"
	"fmt"
	"image/color"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/media"
	"github.com/go-bumbu/langai/vector"
)

// TestIndex_EmptyPartitionIsNotAWildcard: an Index over an empty partition
// sees that partition's documents only, never another tenant's.
func TestIndex_EmptyPartitionIsNotAWildcard(t *testing.T) {
	ctx := context.Background()
	store := vector.NewMemStore()
	emb := langaitest.FakeEmbedder{}
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
	langaitest.FakeEmbedder
	model string
}

func (e namedEmbedder) Model() string { return e.model }

var _ langai.Identified = namedEmbedder{}

// TestIndex_RefusesAnotherModel: the embedder's model is recorded with the
// collection's first vectors, so another model of the same dimensions can
// neither write to the collection nor search it.
func TestIndex_RefusesAnotherModel(t *testing.T) {
	ctx := context.Background()
	store := vector.NewMemStore()
	first, err := New(langaitest.FakeEmbedder{}, store, "i", "p", "c")
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
	anonymous, err := New(langaitest.FakeEmbedder{}, store, "i", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	if got, err := anonymous.Search(ctx, "goroutines", 10); err != nil || len(got) != 1 {
		t.Errorf("Search with the same model: %s, %v", hitsOf(got), err)
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

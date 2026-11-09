package search

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/sense/media"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/vector"
)

func TestSpread(t *testing.T) {
	items := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	for _, tc := range []struct {
		n    int
		want []int
	}{
		{0, items},
		{-1, items},
		{10, items},
		{20, items},
		{1, []int{0}},
		{2, []int{0, 5}},
		{3, []int{0, 3, 6}},
		{4, []int{0, 2, 5, 7}},
	} {
		if got := spread(items, tc.n); !slices.Equal(got, tc.want) {
			t.Errorf("spread(10 items, %d) = %v, want %v", tc.n, got, tc.want)
		}
	}
}

// searchCountingStore records the number of query vectors of every Search.
type searchCountingStore struct {
	vector.Store
	vectors []int
}

func (s *searchCountingStore) Search(ctx context.Context, c vector.Coord, q vector.Query) ([]vector.Match, error) {
	s.vectors = append(s.vectors, len(q.Vectors))
	return s.Store.Search(ctx, c, q)
}

// TestSearchIndexed_OneCappedSearch: a "more like this" search is one store
// search, with every stored chunk up to the cap and an evenly spaced
// selection of them beyond it.
func TestSearchIndexed_OneCappedSearch(t *testing.T) {
	ctx := context.Background()
	var long strings.Builder
	for i := range 40 {
		fmt.Fprintf(&long, "paragraph %d about topic%d\n\n", i, i)
	}
	for _, tc := range []struct {
		name string
		opts []Option
		want int
	}{
		{"default cap", nil, 40},
		{"cap of 4", []Option{WithMaxQueryVectors(4)}, 4},
		{"no cap given", []Option{WithMaxQueryVectors(0)}, 40},
	} {
		store := &searchCountingStore{Store: vector.NewMemStore()}
		opts := append([]Option{WithChunker(&MarkdownChunker{MaxChars: 30})}, tc.opts...)
		ix, err := New(sensetest.FakeEmbedder{}, store, "t", "p", "c", opts...)
		if err != nil {
			t.Fatal(err)
		}
		if err := ix.Index(ctx, "long", long.String()); err != nil {
			t.Fatal(err)
		}
		if err := ix.Index(ctx, "other", "paragraph about topic7"); err != nil {
			t.Fatal(err)
		}
		hits, err := ix.SearchIndexed(ctx, "long", 5)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(store.vectors, []int{tc.want}) {
			t.Errorf("%s: store searches with %v vectors, want one with %d", tc.name, store.vectors, tc.want)
		}
		if len(hits) != 1 || hits[0].DocID != "other" {
			t.Errorf("%s: hits = %s, want other", tc.name, hitsOf(hits))
		}
	}
	if DefaultMaxQueryVectors != 64 {
		t.Errorf("DefaultMaxQueryVectors = %d, the README says 64", DefaultMaxQueryVectors)
	}
}

// imageCounter counts the images it is handed.
type imageCounter struct {
	sensetest.FakeImageEmbedder
	images int
}

func (c *imageCounter) EmbedImages(ctx context.Context, images []image.Image) ([][]float32, error) {
	c.images += len(images)
	return c.FakeImageEmbedder.EmbedImages(ctx, images)
}

// TestSearchFrames_CapsBeforeEmbedding: a query video of more frames than
// the cap embeds only the evenly spaced frames it searches with, and
// QueryAt is the time of one of those.
func TestSearchFrames_CapsBeforeEmbedding(t *testing.T) {
	ctx := context.Background()
	emb := &imageCounter{}
	store := &searchCountingStore{Store: vector.NewMemStore()}
	ix, err := NewMediaIndex(emb, store, "t", "p", "c", WithMediaMaxQueryVectors(3))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.IndexFrames(ctx, "img", "h", []media.Frame{{Image: pic(64, 48, color.RGBA{0, 200, 0, 255}, blue)}}); err != nil {
		t.Fatal(err)
	}
	emb.images = 0
	var frames []media.Frame
	for i := range 9 {
		frames = append(frames, media.Frame{Image: pic(64, 48, color.RGBA{uint8(20 * i), 200, 0, 255}, blue), At: time.Duration(i) * time.Second})
	}
	hits, err := ix.SearchFrames(ctx, frames, 5)
	if err != nil {
		t.Fatal(err)
	}
	if emb.images != 3 || !slices.Equal(store.vectors, []int{3}) {
		t.Errorf("embedded %d frames and searched with %v vectors, want 3 in one search", emb.images, store.vectors)
	}
	if len(hits) != 1 || !slices.Contains([]time.Duration{0, 3 * time.Second, 6 * time.Second}, hits[0].QueryAt) {
		t.Errorf("hits = %+v, want img matched by frame 0, 3 or 6", hits)
	}
}

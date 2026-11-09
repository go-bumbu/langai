package search

import (
	"bytes"
	"context"
	"errors"
	"image"
	"math"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/vector"
)

// countingReader is an endless stream of zeros that counts what was read.
type countingReader struct{ n int64 }

func (c *countingReader) Read(p []byte) (int, error) {
	clear(p)
	c.n += int64(len(p))
	return len(p), nil
}

// TestMediaIndex_RefusesOversizedImages: an image stream longer than the
// limit is refused with ErrTooLarge after reading just past the limit, on
// indexing and on search alike.
func TestMediaIndex_RefusesOversizedImages(t *testing.T) {
	const limit = 4096
	ix, emb, store := newMediaIndex(t, WithMaxImageBytes(limit))
	ctx := context.Background()
	r := &countingReader{}
	if err := ix.IndexImage(ctx, "big", r); !errors.Is(err, ErrTooLarge) || !strings.Contains(err.Error(), `"big"`) {
		t.Errorf("IndexImage: %v, want ErrTooLarge naming the item", err)
	}
	if r.n > limit+bytes.MinRead+1 {
		t.Errorf("read %d bytes for a limit of %d", r.n, limit)
	}
	r = &countingReader{}
	if _, err := ix.SearchImage(ctx, r, 5); !errors.Is(err, ErrTooLarge) {
		t.Errorf("SearchImage: %v, want ErrTooLarge", err)
	}
	if r.n > limit+bytes.MinRead+1 {
		t.Errorf("search read %d bytes for a limit of %d", r.n, limit)
	}
	if emb.calls.Load() != 0 {
		t.Error("an oversized image reached the embedder")
	}
	if h, _ := store.Hashes(ctx, ix.coord, "big"); len(h) != 0 {
		t.Errorf("an oversized image was stored: %v", h)
	}
	// a picture within the limit still indexes
	if err := ix.IndexImage(ctx, "small", bytes.NewReader(encoded(t, pic(16, 16, red, blue)))); err != nil {
		t.Errorf("a small image: %v", err)
	}
	if defaultMaxImageBytes < 32<<20 {
		t.Errorf("the default limit of %d bytes is below a large phone photo", defaultMaxImageBytes)
	}
}

// TestMediaIndex_VideoWithNothingToEmbedIsAnError: a video the sampler
// takes no frames from, or the video embedder no segments, is an error and
// leaves nothing stored, so the next run tries it again; so are no frames
// handed to IndexFrames.
func TestMediaIndex_VideoWithNothingToEmbedIsAnError(t *testing.T) {
	ctx := context.Background()
	path := tempFile(t, "clip.mp4", []byte("clip"))
	sampler := &countingSampler{}
	frames, _, frameStore := newMediaIndex(t, WithSampler(sampler))
	segments, _, segmentStore := newMediaIndex(t, WithVideoEmbedder(&sensetest.FakeVideoEmbedder{Clips: map[string][]image.Image{"clip.mp4": nil}}))
	for name, tc := range map[string]struct {
		call  func() error
		store *vector.MemStore
		id    string
	}{
		"sampler":     {func() error { return frames.IndexVideo(ctx, "clip", path) }, frameStore, "clip"},
		"segments":    {func() error { return segments.IndexVideo(ctx, "clip", path) }, segmentStore, "clip"},
		"IndexFrames": {func() error { return frames.IndexFrames(ctx, "mine", "h", nil) }, frameStore, "mine"},
	} {
		if err := tc.call(); !errors.Is(err, ErrNoFrames) || !strings.Contains(err.Error(), `"`+tc.id+`"`) {
			t.Errorf("%s: %v, want ErrNoFrames naming the item", name, err)
		}
		if h, _ := tc.store.Hashes(ctx, frames.coord, tc.id); len(h) != 0 {
			t.Errorf("%s: stored %v for an item with nothing to embed", name, h)
		}
	}
	// the next run samples again instead of finding it unchanged
	_ = frames.IndexVideo(ctx, "clip", path)
	if sampler.calls != 2 {
		t.Errorf("%d sample calls, want the video tried again", sampler.calls)
	}
}

// TestIndex_SanitizesText: NUL bytes and invalid UTF-8, which a Postgres
// TEXT column refuses, are cleaned before chunking, so every store holds
// the same text, and the hash is over the cleaned text.
func TestIndex_SanitizesText(t *testing.T) {
	ix, store := newTestIndex(t)
	ctx := context.Background()
	if err := ix.Index(ctx, "doc", "nul\x00byte and bad \xff\xfe utf8"); err != nil {
		t.Fatal(err)
	}
	chunks, err := store.Chunks(ctx, ix.coord, "doc")
	if err != nil || len(chunks) != 1 {
		t.Fatalf("chunks = %v, %v", chunks, err)
	}
	if text := chunks[0].Text; strings.ContainsRune(text, 0) || !utf8.ValidString(text) || !strings.Contains(text, "nulbyte") {
		t.Errorf("stored text %q, want it without NUL and valid UTF-8", text)
	}
	before, _ := store.Hashes(ctx, ix.coord, "doc")
	res, err := ix.IndexBatch(ctx, []Doc{{ID: "doc", Text: "nulbyte and bad � utf8"}})
	if err != nil || len(res.Unchanged) != 1 {
		t.Errorf("the cleaned text again: %+v, %v; want it unchanged", res, err)
	}
	if after, _ := store.Hashes(ctx, ix.coord, "doc"); after["doc"] != before["doc"] {
		t.Errorf("hash changed from %q to %q", before["doc"], after["doc"])
	}
}

// zeroEmbedder embeds every text to the zero vector, or to NaN.
type zeroEmbedder struct{ nan bool }

func (zeroEmbedder) Dimensions() int { return 48 }

func (z zeroEmbedder) Embed(_ context.Context, _ sense.Role, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = make([]float32, 48)
		if z.nan {
			out[i][0] = float32(math.NaN())
		}
	}
	return out, nil
}

// TestSearch_ZeroVectorQueryFindsNothing: a query that embeds to the zero
// vector has no direction to rank by, so it finds nothing, as an empty
// query does, rather than surfacing the store's ErrInvalidVector; a NaN is
// the embedder's fault and stays a search: error.
func TestSearch_ZeroVectorQueryFindsNothing(t *testing.T) {
	ctx := context.Background()
	ix, _ := newTestIndex(t)
	if err := ix.Index(ctx, "doc", "goroutines and channels"); err != nil {
		t.Fatal(err)
	}
	if got, err := ix.Search(ctx, "!!! ???", 10); err != nil || len(got) != 0 {
		t.Errorf("zero query: %s, %v; want nothing", hitsOf(got), err)
	}

	mix, _, _ := newMediaIndex(t, WithTextEmbedder(zeroEmbedder{}))
	if err := mix.IndexImage(ctx, "a", bytes.NewReader(encoded(t, pic(16, 16, red, blue)))); err != nil {
		t.Fatal(err)
	}
	if got, err := mix.SearchText(ctx, "anything", 10); err != nil || len(got) != 0 {
		t.Errorf("zero text query: %+v, %v; want nothing", got, err)
	}
	nan, _, _ := newMediaIndex(t, WithTextEmbedder(zeroEmbedder{nan: true}))
	if _, err := nan.SearchText(ctx, "anything", 10); !errors.Is(err, vector.ErrInvalidVector) || !strings.HasPrefix(err.Error(), "search: ") {
		t.Errorf("NaN text query: %v, want a search: error wrapping ErrInvalidVector", err)
	}
}

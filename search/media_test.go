package search

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/media"
	"github.com/go-bumbu/langai/vector"
)

var (
	red   = color.RGBA{200, 30, 30, 255}
	green = color.RGBA{30, 200, 30, 255}
	blue  = color.RGBA{30, 30, 200, 255}
)

// pic draws a w×h picture whose left half is one colour and right half
// another; the same halves look alike at any size.
func pic(w, h int, left, right color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			c := left
			if x >= w/2 {
				c = right
			}
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

// encoded is pic as PNG bytes, what IndexImage and SearchImage take.
func encoded(t *testing.T, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// countingImageEmbedder counts the EmbedImages calls and can fail them.
type countingImageEmbedder struct {
	langaitest.FakeImageEmbedder
	calls atomic.Int32
	err   error
	short bool // answer one vector too few
}

func (c *countingImageEmbedder) EmbedImages(ctx context.Context, images []image.Image) ([][]float32, error) {
	c.calls.Add(1)
	if c.err != nil {
		return nil, c.err
	}
	out, err := c.FakeImageEmbedder.EmbedImages(ctx, images)
	if err == nil && c.short && len(out) > 0 {
		out = out[:len(out)-1]
	}
	return out, err
}

// countingSampler answers every path with frames and counts the calls.
type countingSampler struct {
	frames []media.Frame
	err    error
	calls  int
}

func (s *countingSampler) Sample(_ context.Context, _ string) ([]media.Frame, error) {
	s.calls++
	return s.frames, s.err
}

// tempFile writes content to a file in a temp dir and returns its path.
func tempFile(t *testing.T, name string, content []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func newMediaIndex(t *testing.T, opts ...MediaOption) (*MediaIndex, *countingImageEmbedder, *vector.MemStore) {
	t.Helper()
	emb := &countingImageEmbedder{}
	store := vector.NewMemStore()
	ix, err := NewMediaIndex(emb, store, "t", "p", "c", opts...)
	if err != nil {
		t.Fatal(err)
	}
	return ix, emb, store
}

func TestNewMediaIndex_RejectsNilDependencies(t *testing.T) {
	if _, err := NewMediaIndex(nil, vector.NewMemStore(), "t", "p", "c"); err == nil {
		t.Error("nil embedder: want an error")
	}
	if _, err := NewMediaIndex(langaitest.FakeImageEmbedder{}, nil, "t", "p", "c"); err == nil {
		t.Error("nil store: want an error")
	}
}

func TestMediaIndex_ImageFindsItsCopyFirst(t *testing.T) {
	ix, _, _ := newMediaIndex(t)
	ctx := context.Background()
	for id, img := range map[string]image.Image{
		"a": pic(64, 48, red, blue),
		"b": pic(64, 48, green, red),
		"c": pic(64, 48, blue, blue),
	} {
		if err := ix.IndexImage(ctx, id, bytes.NewReader(encoded(t, img))); err != nil {
			t.Fatalf("IndexImage(%s): %v", id, err)
		}
	}
	// the query is "a" at another size
	hits, err := ix.SearchImage(ctx, bytes.NewReader(encoded(t, pic(640, 480, red, blue))), 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 {
		t.Fatalf("got %d hits, want all 3 (no floor): %+v", len(hits), hits)
	}
	if hits[0].DocID != "a" || hits[0].Score < 0.99 {
		t.Errorf("first hit = %+v, want a with a score near 1", hits[0])
	}
	if hits[0].At != 0 || hits[0].QueryAt != 0 {
		t.Errorf("image hit carries times: %+v", hits[0])
	}
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Errorf("hits out of order: %+v", hits)
		}
	}
}

func TestMediaIndex_UnchangedImageSkipsEmbedding(t *testing.T) {
	ix, emb, _ := newMediaIndex(t)
	ctx := context.Background()
	data := encoded(t, pic(64, 48, red, blue))
	for range 3 {
		if err := ix.IndexImage(ctx, "a", bytes.NewReader(data)); err != nil {
			t.Fatal(err)
		}
	}
	if n := emb.calls.Load(); n != 1 {
		t.Errorf("the same bytes were embedded %d times, want 1", n)
	}
	if err := ix.IndexImage(ctx, "a", bytes.NewReader(encoded(t, pic(64, 48, blue, red)))); err != nil {
		t.Fatal(err)
	}
	if n := emb.calls.Load(); n != 2 {
		t.Errorf("changed bytes: %d embed calls, want 2", n)
	}
}

func TestMediaIndex_UpdateReplacesAndDeleteRemoves(t *testing.T) {
	ix, _, store := newMediaIndex(t)
	ctx := context.Background()
	a := encoded(t, pic(64, 48, red, blue))
	if err := ix.IndexImage(ctx, "x", bytes.NewReader(a)); err != nil {
		t.Fatal(err)
	}
	if err := ix.IndexImage(ctx, "x", bytes.NewReader(encoded(t, pic(64, 48, green, green)))); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.SearchImage(ctx, bytes.NewReader(a), 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Score > 0.99 {
		t.Errorf("after the update, x still matches its old picture: %+v", hits)
	}
	if err := ix.Delete(ctx, "x"); err != nil {
		t.Fatal(err)
	}
	if hits, _ = ix.SearchImage(ctx, bytes.NewReader(a), 10); len(hits) != 0 {
		t.Errorf("after Delete: hits = %+v", hits)
	}
	if h, _ := store.Hashes(ctx, vector.Coord{Index: "t", Partition: "p", Collection: "c"}, "x"); len(h) != 0 {
		t.Errorf("after Delete the hash is still stored: %v", h)
	}
	if err := ix.Delete(ctx, "never-indexed"); err != nil {
		t.Errorf("deleting an unknown id: %v", err)
	}
	if err := ix.Delete(ctx, ""); err == nil {
		t.Error("empty id: want an error")
	}
}

// clipFrames is a video of three shots, each one of the test pictures.
func clipFrames() []media.Frame {
	return []media.Frame{
		{Image: pic(64, 48, red, blue), At: 0},
		{Image: pic(64, 48, green, red), At: 4 * time.Second},
		{Image: pic(64, 48, blue, blue), At: 8 * time.Second},
	}
}

// checkClipChunks checks what IndexVideo stored for clipFrames: one chunk per
// frame, Seq in milliseconds, and a store search that finds the clip once,
// by its frame closest to the query.
func checkClipChunks(t *testing.T, store vector.Store, emb langai.ImageEmbedder) {
	t.Helper()
	ctx := context.Background()
	coord := vector.Coord{Index: "t", Partition: "p", Collection: "c"}
	chunks, err := store.Chunks(ctx, coord, "clip")
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 3 || chunks[0].Seq != 0 || chunks[1].Seq != 4000 || chunks[2].Seq != 8000 || chunks[1].Text != "" {
		t.Errorf("stored chunks = %+v, want 3 at Seq 0, 4000 and 8000", chunks)
	}
	raw, err := store.Search(ctx, coord, vector.Query{Vector: mustEmbed(t, emb, pic(64, 48, green, red)), Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 1 || raw[0].Seq != 4000 {
		t.Errorf("store search = %+v, want the clip once, by its frame at Seq 4000", raw)
	}
}

func TestMediaIndex_ImageQueryFindsTheVideoFrame(t *testing.T) {
	sampler := &countingSampler{frames: clipFrames()}
	ix, emb, store := newMediaIndex(t, WithSampler(sampler))
	ctx := context.Background()
	path := tempFile(t, "clip.mp4", []byte("pretend video bytes"))
	if err := ix.IndexVideo(ctx, "clip", path); err != nil {
		t.Fatal(err)
	}
	checkClipChunks(t, store, emb)
	// the image query is the second shot at another size
	hits, err := ix.SearchImage(ctx, bytes.NewReader(encoded(t, pic(128, 96, green, red))), 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].DocID != "clip" || hits[0].Score < 0.99 || hits[0].At != 4*time.Second || hits[0].QueryAt != 0 {
		t.Errorf("hits = %+v, want clip at 4s with a score near 1", hits)
	}
	// the same file is neither sampled nor embedded again
	embeds := emb.calls.Load()
	if err := ix.IndexVideo(ctx, "clip", path); err != nil {
		t.Fatal(err)
	}
	if sampler.calls != 1 || emb.calls.Load() != embeds {
		t.Errorf("unchanged file: %d sample calls, %d embed calls, want 1 and %d", sampler.calls, emb.calls.Load(), embeds)
	}
	// a changed file is
	if err := os.WriteFile(path, []byte("other bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ix.IndexVideo(ctx, "clip", path); err != nil {
		t.Fatal(err)
	}
	if sampler.calls != 2 {
		t.Errorf("changed file: %d sample calls, want 2", sampler.calls)
	}
}

func TestMediaIndex_VideoQueryFindsImagesAndVideos(t *testing.T) {
	sampler := &countingSampler{frames: clipFrames()}
	ix, _, _ := newMediaIndex(t, WithSampler(sampler))
	ctx := context.Background()
	if err := ix.IndexVideo(ctx, "clip", tempFile(t, "clip.mp4", []byte("clip"))); err != nil {
		t.Fatal(err)
	}
	if err := ix.IndexImage(ctx, "a", bytes.NewReader(encoded(t, pic(64, 48, red, blue)))); err != nil {
		t.Fatal(err)
	}
	if err := ix.IndexImage(ctx, "far", bytes.NewReader(encoded(t, pic(64, 48, green, green)))); err != nil {
		t.Fatal(err)
	}
	// the query video: the third shot exactly, then the first shot in a
	// slightly different red (the fake embeds a resized copy exactly, so a
	// copy would tie with the exact hit)
	query := []media.Frame{
		{Image: pic(64, 48, blue, blue), At: 0},
		{Image: pic(32, 24, color.RGBA{170, 50, 50, 255}, blue), At: 2 * time.Second},
	}
	hits, err := ix.SearchFrames(ctx, query, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 {
		t.Fatalf("hits = %+v, want clip, a and far", hits)
	}
	if hits[0].DocID != "clip" || hits[0].Score < 0.999 || hits[0].At != 8*time.Second || hits[0].QueryAt != 0 {
		t.Errorf("hits[0] = %+v, want clip by its 8s frame against the query's first", hits[0])
	}
	if hits[1].DocID != "a" || hits[1].Score < 0.95 || hits[1].QueryAt != 2*time.Second || hits[1].At != 0 {
		t.Errorf("hits[1] = %+v, want a against the query frame at 2s", hits[1])
	}
	if hits[2].DocID != "far" || hits[2].Score >= hits[1].Score {
		t.Errorf("hits[2] = %+v, want far last", hits[2])
	}

	// SearchVideo goes through the sampler
	sampler.frames = query
	hits, err = ix.SearchVideo(ctx, "any.mp4", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].DocID != "clip" {
		t.Errorf("SearchVideo hits = %+v, want clip alone (limit 1)", hits)
	}
}

func TestMediaIndex_SearchIndexedReusesTheStoredVectors(t *testing.T) {
	sampler := &countingSampler{frames: clipFrames()}
	ix, emb, _ := newMediaIndex(t, WithSampler(sampler))
	ctx := context.Background()
	if err := ix.IndexVideo(ctx, "clip", tempFile(t, "clip.mp4", []byte("clip"))); err != nil {
		t.Fatal(err)
	}
	for id, img := range map[string]image.Image{"a": pic(64, 48, green, red), "far": pic(64, 48, green, green)} {
		if err := ix.IndexImage(ctx, id, bytes.NewReader(encoded(t, img))); err != nil {
			t.Fatal(err)
		}
	}
	calls := emb.calls.Load()

	// an image: the clip first by its frame at 4 s, the image itself left out
	hits, err := ix.SearchIndexed(ctx, "a", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].DocID != "clip" || hits[0].Score < 0.999 {
		t.Fatalf("hits = %+v, want clip first of 2, a left out", hits)
	}
	if hits[0].At != 4*time.Second || hits[0].QueryAt != 0 {
		t.Errorf("hits[0] = %+v, want clip by its 4 s frame", hits[0])
	}

	// a video: every stored frame is a query, QueryAt its time
	hits, err = ix.SearchIndexed(ctx, "clip", 10)
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]MediaMatch{}
	for _, h := range hits {
		byID[h.DocID] = h
	}
	if h, ok := byID["clip"]; ok {
		t.Errorf("clip = %+v, want the query item left out", h)
	}
	if h := byID["a"]; h.Score < 0.999 || h.QueryAt != 4*time.Second {
		t.Errorf("a = %+v, want found by the query frame at 4 s", h)
	}

	if got := emb.calls.Load(); got != calls {
		t.Errorf("SearchIndexed ran the embedder %d times, want none", got-calls)
	}
	if sampler.calls != 1 {
		t.Errorf("sampler ran %d times, want only the indexing", sampler.calls)
	}
	if _, err := ix.SearchIndexed(ctx, "missing", 10); !errors.Is(err, ErrNotIndexed) {
		t.Errorf("unknown id: %v, want ErrNotIndexed", err)
	}
}

func TestMediaIndex_MinScoreLimitAndEmptyQueries(t *testing.T) {
	ix, _, _ := newMediaIndex(t, WithMediaMinScore(0.999))
	ctx := context.Background()
	for id, img := range map[string]image.Image{"a": pic(64, 48, red, blue), "b": pic(64, 48, green, red)} {
		if err := ix.IndexImage(ctx, id, bytes.NewReader(encoded(t, img))); err != nil {
			t.Fatal(err)
		}
	}
	hits, err := ix.SearchImage(ctx, bytes.NewReader(encoded(t, pic(64, 48, red, blue))), 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].DocID != "a" {
		t.Errorf("with a floor of 0.999: hits = %+v, want a alone", hits)
	}
	if hits, err := ix.SearchFrames(ctx, nil, 10); err != nil || hits != nil {
		t.Errorf("no frames: %v, %v; want nothing", hits, err)
	}
}

func TestMediaIndex_IndexFramesHonoursTheCallersHash(t *testing.T) {
	ix, emb, _ := newMediaIndex(t)
	ctx := context.Background()
	frames := clipFrames()[:1]
	for range 2 {
		if err := ix.IndexFrames(ctx, "x", "h1", frames); err != nil {
			t.Fatal(err)
		}
	}
	if emb.calls.Load() != 1 {
		t.Errorf("same hash: %d embed calls, want 1", emb.calls.Load())
	}
	for range 2 {
		if err := ix.IndexFrames(ctx, "x", "", frames); err != nil {
			t.Fatal(err)
		}
	}
	if emb.calls.Load() != 3 {
		t.Errorf("empty hash: %d embed calls, want 3", emb.calls.Load())
	}
}

func TestMediaIndex_ArgumentErrors(t *testing.T) {
	ctx := context.Background()
	ix, _, _ := newMediaIndex(t)
	a := encoded(t, pic(64, 48, red, blue))

	if err := ix.IndexImage(ctx, "", bytes.NewReader(a)); err == nil {
		t.Error("IndexImage with an empty id: want an error")
	}
	if err := ix.IndexVideo(ctx, "", "x.mp4"); err == nil {
		t.Error("IndexVideo with an empty id: want an error")
	}
	if err := ix.IndexFrames(ctx, "", "", nil); err == nil {
		t.Error("IndexFrames with an empty id: want an error")
	}
	if err := ix.IndexImage(ctx, "x", strings.NewReader("not an image")); err == nil || !strings.Contains(err.Error(), "decode") {
		t.Errorf("undecodable image: err = %v", err)
	}
	if _, err := ix.SearchImage(ctx, strings.NewReader("not an image"), 5); err == nil {
		t.Error("undecodable query: want an error")
	}
	if err := ix.IndexVideo(ctx, "v", "x.mp4"); err == nil || !strings.Contains(err.Error(), "sampler") {
		t.Errorf("no sampler: err = %v", err)
	}
	if _, err := ix.SearchVideo(ctx, "x.mp4", 5); err == nil || !strings.Contains(err.Error(), "sampler") {
		t.Errorf("no sampler: err = %v", err)
	}
	if err := ix.IndexFrames(ctx, "x", "", []media.Frame{{}}); err == nil || !strings.Contains(err.Error(), "no image") {
		t.Errorf("frame without an image: err = %v", err)
	}
	if _, err := ix.SearchFrames(ctx, []media.Frame{{}}, 5); err == nil {
		t.Error("query frame without an image: want an error")
	}
}

func TestMediaIndex_DependencyErrors(t *testing.T) {
	ctx := context.Background()
	ix, emb, _ := newMediaIndex(t)
	a := encoded(t, pic(64, 48, red, blue))

	emb.short = true
	if err := ix.IndexFrames(ctx, "x", "", clipFrames()); err == nil || !strings.Contains(err.Error(), "vectors") {
		t.Errorf("short embedder: err = %v", err)
	}
	emb.short = false
	emb.err = errors.New("model down")
	if err := ix.IndexImage(ctx, "x", bytes.NewReader(a)); err == nil || !strings.Contains(err.Error(), "model down") {
		t.Errorf("failing embedder: err = %v", err)
	}
	if _, err := ix.SearchImage(ctx, bytes.NewReader(a), 5); err == nil || !strings.Contains(err.Error(), "model down") {
		t.Errorf("failing embedder on search: err = %v", err)
	}
	emb.err = nil

	withSampler, _, _ := newMediaIndex(t, WithSampler(&countingSampler{err: errors.New("ffmpeg exploded")}))
	if err := withSampler.IndexVideo(ctx, "v", tempFile(t, "v.mp4", []byte("v"))); err == nil || !strings.Contains(err.Error(), "ffmpeg exploded") {
		t.Errorf("failing sampler: err = %v", err)
	}
	if _, err := withSampler.SearchVideo(ctx, "v.mp4", 5); err == nil || !strings.Contains(err.Error(), "ffmpeg exploded") {
		t.Errorf("failing sampler on search: err = %v", err)
	}
	if err := withSampler.IndexVideo(ctx, "v", filepath.Join(t.TempDir(), "missing.mp4")); err == nil || !strings.Contains(err.Error(), "hash video") {
		t.Errorf("missing file: err = %v", err)
	}
}

// hugePNG is a syntactically valid PNG header claiming a 100000×100000
// picture: DecodeConfig believes it, Decode would try to allocate 40 GB.
func hugePNG() []byte {
	var buf bytes.Buffer
	buf.WriteString("\x89PNG\r\n\x1a\n")
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:], 100000)
	binary.BigEndian.PutUint32(ihdr[4:], 100000)
	ihdr[8], ihdr[9] = 8, 6 // 8-bit RGBA
	chunk := append([]byte("IHDR"), ihdr...)
	_ = binary.Write(&buf, binary.BigEndian, uint32(len(ihdr)))
	buf.Write(chunk)
	_ = binary.Write(&buf, binary.BigEndian, crc32.ChecksumIEEE(chunk))
	return buf.Bytes()
}

func TestDecodeImage_RefusesHugePictures(t *testing.T) {
	if _, err := decodeImage(hugePNG()); err == nil || !strings.Contains(err.Error(), "pixels") {
		t.Errorf("err = %v, want a pixel-count refusal", err)
	}
	// a real picture still decodes whole after the config pass
	img, err := decodeImage(encoded(t, pic(300, 200, red, blue)))
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != 300 || b.Dy() != 200 {
		t.Errorf("decoded %v, want 300x200", b)
	}
}

func mustEmbed(t *testing.T, emb langai.ImageEmbedder, img image.Image) []float32 {
	t.Helper()
	vecs, err := emb.EmbedImages(context.Background(), []image.Image{img})
	if err != nil {
		t.Fatal(err)
	}
	return vecs[0]
}

// colourText is a text Embedder in the fake image space: a text names a
// colour, and embeds as a picture of it.
type colourText struct{ emb langaitest.FakeImageEmbedder }

func (c colourText) Dimensions() int { return c.emb.Dimensions() }

func (c colourText) Embed(ctx context.Context, _ langai.Role, texts []string) ([][]float32, error) {
	images := make([]image.Image, len(texts))
	for i, t := range texts {
		col := map[string]color.RGBA{"red": red, "green": green, "blue": blue}[t]
		images[i] = pic(32, 32, col, col)
	}
	return c.emb.EmbedImages(ctx, images)
}

// TestMediaIndex_VideoEmbedderStoresSegments: with a VideoEmbedder a video
// is its segments, found by an image at the segment's start time, and a
// video query says which of its segments matched.
func TestMediaIndex_VideoEmbedderStoresSegments(t *testing.T) {
	ve := &langaitest.FakeVideoEmbedder{Clips: map[string][]image.Image{
		"clip.mp4":  {pic(64, 48, red, red), pic(64, 48, green, green)},
		"query.mp4": {pic(64, 48, blue, blue), pic(64, 48, green, green)},
	}}
	ix, emb, _ := newMediaIndex(t, WithVideoEmbedder(ve), WithTextEmbedder(colourText{}))
	ctx := context.Background()
	clip := tempFile(t, "clip.mp4", []byte("clip"))
	if err := ix.IndexVideo(ctx, "clip", clip); err != nil {
		t.Fatal(err)
	}
	if ve.Calls() != 1 || emb.calls.Load() != 0 {
		t.Errorf("video embedder called %d times, image embedder %d; want 1 and 0", ve.Calls(), emb.calls.Load())
	}
	if err := ix.IndexVideo(ctx, "clip", clip); err != nil || ve.Calls() != 1 {
		t.Errorf("unchanged video was embedded again (%d calls, %v)", ve.Calls(), err)
	}
	if err := ix.IndexImage(ctx, "green", bytes.NewReader(encoded(t, pic(64, 48, green, green)))); err != nil {
		t.Fatal(err)
	}

	hits, err := ix.SearchImage(ctx, bytes.NewReader(encoded(t, pic(32, 24, green, green))), 10)
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]MediaMatch{}
	for _, h := range hits {
		found[h.DocID] = h
	}
	if len(hits) != 2 || found["green"].Score < 0.99 || found["clip"].At != 10*time.Second || found["clip"].Score < 0.99 {
		t.Errorf("image query: %+v, want green and the clip's second segment at 10 s", hits)
	}

	query := tempFile(t, "query.mp4", []byte("query"))
	hits, err = ix.SearchVideo(ctx, query, 10)
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]MediaMatch{}
	for _, h := range hits {
		byID[h.DocID] = h
	}
	if h := byID["clip"]; h.At != 10*time.Second || h.QueryAt != 10*time.Second || h.Score < 0.99 {
		t.Errorf("video query: clip = %+v, want its 10 s segment found by the query's 10 s segment", h)
	}
}

// TestMediaIndex_SearchText: a text in the model's space finds the segment
// that shows it; a blank text finds nothing.
func TestMediaIndex_SearchText(t *testing.T) {
	ve := &langaitest.FakeVideoEmbedder{Clips: map[string][]image.Image{"clip.mp4": {pic(64, 48, red, red), pic(64, 48, green, green)}}}
	ix, _, _ := newMediaIndex(t, WithVideoEmbedder(ve), WithTextEmbedder(colourText{}))
	ctx := context.Background()
	if err := ix.IndexVideo(ctx, "clip", tempFile(t, "clip.mp4", []byte("clip"))); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.SearchText(ctx, "green", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].DocID != "clip" || hits[0].At != 10*time.Second || hits[0].Score < 0.99 {
		t.Errorf("text query: %+v, want the clip's second segment", hits)
	}
	if hits, err := ix.SearchText(ctx, "  ", 10); err != nil || hits != nil {
		t.Errorf("blank text: %v, %v", hits, err)
	}
}

func TestMediaIndex_VideoEmbedderErrors(t *testing.T) {
	ctx := context.Background()
	// dimension mismatches are refused up front
	if _, err := NewMediaIndex(langaitest.FakeImageEmbedder{}, vector.NewMemStore(), "t", "p", "c", WithVideoEmbedder(dims{7})); err == nil || !strings.Contains(err.Error(), "7 dimensions") {
		t.Errorf("video dims: %v", err)
	}
	if _, err := NewMediaIndex(langaitest.FakeImageEmbedder{}, vector.NewMemStore(), "t", "p", "c", WithTextEmbedder(langaitest.FakeEmbedder{Dims: 7})); err == nil || !strings.Contains(err.Error(), "7 dimensions") {
		t.Errorf("text dims: %v", err)
	}
	ix, _, _ := newMediaIndex(t)
	if _, err := ix.SearchText(ctx, "red", 10); err == nil || !strings.Contains(err.Error(), "WithTextEmbedder") {
		t.Errorf("no text embedder: %v", err)
	}
	ve := &langaitest.FakeVideoEmbedder{Err: errors.New("boom")}
	ix, _, _ = newMediaIndex(t, WithVideoEmbedder(ve))
	clip := tempFile(t, "clip.mp4", []byte("clip"))
	if err := ix.IndexVideo(ctx, "clip", clip); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("embed error: %v", err)
	}
	if _, err := ix.SearchVideo(ctx, clip, 10); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("search embed error: %v", err)
	}
	// a segment of the wrong width is refused
	bad := &langaitest.FakeVideoEmbedder{Clips: map[string][]image.Image{"clip.mp4": {pic(8, 8, red, red)}}}
	ix, _, _ = newMediaIndex(t, WithVideoEmbedder(wrongWidth{bad}))
	if err := ix.IndexVideo(ctx, "clip", clip); err == nil || !strings.Contains(err.Error(), "dimensions") {
		t.Errorf("wrong width: %v", err)
	}
}

// dims is a VideoEmbedder of a fixed width that is never called.
type dims struct{ n int }

func (d dims) Dimensions() int { return d.n }
func (dims) EmbedVideo(context.Context, string) ([]langai.Segment, error) {
	return nil, errors.New("unexpected call")
}

// wrongWidth claims the right width but answers short vectors.
type wrongWidth struct{ *langaitest.FakeVideoEmbedder }

func (w wrongWidth) EmbedVideo(ctx context.Context, path string) ([]langai.Segment, error) {
	segs, err := w.FakeVideoEmbedder.EmbedVideo(ctx, path)
	for i := range segs {
		segs[i].Vector = segs[i].Vector[:2]
	}
	return segs, err
}

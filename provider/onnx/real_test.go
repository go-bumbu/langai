package onnx

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/media"
	"github.com/go-bumbu/sense/search"
	"github.com/go-bumbu/sense/vector"
	xdraw "golang.org/x/image/draw"
)

// realConfig is the Config the real-model tests share. On first run they
// download the pinned runtime (~11 MB) and model (~370 MB) into
// SEMSEARCH_ONNX_TEST_DIR, so they only run when SEMSEARCH_MODELS is set:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_ONNX_TEST_DIR=/tmp/semsearch-onnx go test ./provider/onnx/
//
// SEMSEARCH_ONNX_MODEL selects another pinned model (DINOv2Model).
func realConfig(t *testing.T) Config {
	t.Helper()
	if os.Getenv("SEMSEARCH_MODELS") == "" {
		t.Skip("set SEMSEARCH_MODELS=1 to download and test the real model")
	}
	dir := os.Getenv("SEMSEARCH_ONNX_TEST_DIR")
	if dir == "" {
		t.Skip("SEMSEARCH_ONNX_TEST_DIR not set; skipping real-model test")
	}
	return Config{
		Runtime:   Runtime{LibDir: filepath.Join(dir, "onnxruntime", LibVersion), ManagedLibDir: true, Logger: slog.New(slog.NewTextHandler(os.Stderr, nil))},
		ModelsDir: filepath.Join(dir, "models"),
		Model:     os.Getenv("SEMSEARCH_ONNX_MODEL"),
	}
}

// scene renders a picture the model can tell apart from others: a filled
// circle of one colour on a light background, with a dark bar below it,
// offset by dx pixels.
func scene(size int, c color.RGBA, dx int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.Draw(img, img.Bounds(), &image.Uniform{color.RGBA{235, 235, 225, 255}}, image.Point{}, draw.Src)
	cx, cy, r := size/2+dx, size*2/5, size/4
	for y := range size {
		for x := range size {
			if (x-cx)*(x-cx)+(y-cy)*(y-cy) <= r*r {
				img.SetRGBA(x, y, c)
			}
		}
	}
	draw.Draw(img, image.Rect(size/4+dx, size*3/4, size*3/4+dx, size*3/4+size/16), &image.Uniform{color.RGBA{40, 40, 40, 255}}, image.Point{}, draw.Src)
	return img
}

// checker renders a very different picture: a fine black and white grid.
func checker(size int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := range size {
		for x := range size {
			c := color.RGBA{255, 255, 255, 255}
			if (x/8+y/8)%2 == 0 {
				c = color.RGBA{0, 0, 0, 255}
			}
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

func resized(img image.Image, size int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), xdraw.Src, nil)
	return dst
}

func recompressed(t *testing.T, img image.Image, quality int) image.Image {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		t.Fatal(err)
	}
	out, err := jpeg.Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot // vectors are unit length
}

// TestEmbedImagesReal exercises the real model through ONNX Runtime: the
// vectors are unit length, a resized and a recompressed copy of a picture
// score near 1 against it, a shifted copy scores above a different picture,
// and batching does not change the vectors.
func TestEmbedImagesReal(t *testing.T) {
	cfg := realConfig(t)
	ctx := context.Background()
	emb, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = emb.Close() }()
	if emb.Dimensions() != 768 {
		t.Errorf("expected 768 dimensions for the pinned model, got %d", emb.Dimensions())
	}

	red := color.RGBA{200, 40, 40, 255}
	original := scene(512, red, 0)
	images := []image.Image{
		original,
		resized(original, 160),
		recompressed(t, original, 40),
		scene(512, red, 60),                         // the same scene, shifted
		scene(512, color.RGBA{40, 60, 200, 255}, 0), // another colour
		checker(512),                                // something else entirely
	}
	vecs, err := emb.EmbedImages(ctx, images)
	if err != nil {
		t.Fatalf("EmbedImages: %v", err)
	}
	if len(vecs) != len(images) {
		t.Fatalf("got %d vectors for %d images", len(vecs), len(images))
	}
	for i, v := range vecs {
		if len(v) != emb.Dimensions() {
			t.Fatalf("vector %d has %d dims", i, len(v))
		}
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		if math.Abs(sum-1) > 1e-3 {
			t.Errorf("vector %d is not unit length: %f", i, sum)
		}
	}
	scores := make([]float64, len(images))
	for i := range images {
		scores[i] = cosine(vecs[0], vecs[i])
	}
	t.Logf("scores against the original: resized %.3f, recompressed %.3f, shifted %.3f, other colour %.3f, checker %.3f",
		scores[1], scores[2], scores[3], scores[4], scores[5])
	if scores[1] < 0.9 || scores[2] < 0.9 {
		t.Errorf("a resized (%.3f) or recompressed (%.3f) copy must score above 0.9", scores[1], scores[2])
	}
	if scores[3] <= scores[5] || scores[4] <= scores[5] {
		t.Errorf("the shifted (%.3f) and recoloured (%.3f) scenes must score above the checker (%.3f)", scores[3], scores[4], scores[5])
	}
	if scores[3] <= scores[4] {
		t.Errorf("the shifted scene (%.3f) must score above the recoloured one (%.3f)", scores[3], scores[4])
	}

	// one at a time or in batches of two, the vectors are the same
	emb.batch = 2
	batched, err := emb.EmbedImages(ctx, images)
	if err != nil {
		t.Fatalf("EmbedImages (batched): %v", err)
	}
	for i := range images {
		if c := cosine(vecs[i], batched[i]); c < 0.999 {
			t.Errorf("image %d: batched vector differs (cosine %.4f)", i, c)
		}
	}
}

var (
	sceneRed  = color.RGBA{200, 40, 40, 255}
	sceneBlue = color.RGBA{40, 60, 200, 255}
)

// BenchmarkEmbedImagesReal times one picture through the real model, eight
// per batch, with the default threads; it is the number the README quotes.
// It skips unless the real-model tests are enabled (see realConfig).
func BenchmarkEmbedImagesReal(b *testing.B) {
	if os.Getenv("SEMSEARCH_MODELS") == "" || os.Getenv("SEMSEARCH_ONNX_TEST_DIR") == "" {
		b.Skip("set SEMSEARCH_MODELS=1 and SEMSEARCH_ONNX_TEST_DIR to benchmark the real model")
	}
	dir := os.Getenv("SEMSEARCH_ONNX_TEST_DIR")
	emb, err := New(context.Background(), Config{
		Runtime:   Runtime{LibDir: filepath.Join(dir, "onnxruntime", LibVersion), ManagedLibDir: true},
		ModelsDir: filepath.Join(dir, "models"),
		Model:     os.Getenv("SEMSEARCH_ONNX_MODEL"),
	})
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = emb.Close() }()
	images := make([]image.Image, 8)
	for i := range images {
		images[i] = scene(1024, sceneRed, i*10)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := emb.EmbedImages(context.Background(), images); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N*len(images)), "ms/image")
}

// encodePNG is what IndexImage and SearchImage read.
func encodePNG(t *testing.T, img image.Image) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return &buf
}

// realCorpus indexes two pictures and a "video" of three frames (a sampler
// stands in for ffmpeg) and returns the index and the video's path.
func realCorpus(t *testing.T, emb sense.ImageEmbedder) (*search.MediaIndex, string) {
	t.Helper()
	ctx := context.Background()
	clip := media.SamplerFunc(func(_ context.Context, _ string) ([]media.Frame, error) {
		return []media.Frame{
			{Image: checker(256), At: 0},
			{Image: scene(256, sceneBlue, 0), At: 4 * time.Second},
			{Image: scene(256, sceneRed, 30), At: 9 * time.Second},
		}, nil
	})
	ix, err := search.NewMediaIndex(emb, vector.NewMemStore(), "t", "p", "media", search.WithSampler(clip))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.IndexImage(ctx, "red", encodePNG(t, scene(512, sceneRed, 0))); err != nil {
		t.Fatal(err)
	}
	if err := ix.IndexImage(ctx, "checker", encodePNG(t, checker(512))); err != nil {
		t.Fatal(err)
	}
	clipFile := filepath.Join(t.TempDir(), "clip.mp4")
	if err := os.WriteFile(clipFile, []byte("not decoded: the sampler above stands in for ffmpeg"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ix.IndexVideo(ctx, "clip", clipFile); err != nil {
		t.Fatal(err)
	}
	return ix, clipFile
}

// TestMediaIndexReal runs the search end to end with the real model: an
// image query finds its own picture, then the video that contains it, with
// the time of the matching frame; the video as a query finds the pictures
// its frames show.
func TestMediaIndexReal(t *testing.T) {
	cfg := realConfig(t)
	ctx := context.Background()
	emb, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = emb.Close() }()
	ix, clipFile := realCorpus(t, emb)

	hits, err := ix.SearchImage(ctx, encodePNG(t, resized(scene(512, sceneRed, 0), 300)), 10)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("hits: %+v", hits)
	if len(hits) != 3 || hits[0].DocID != "red" || hits[0].Score < 0.9 {
		t.Fatalf("hits = %+v, want the red scene first with a score above 0.9", hits)
	}
	if hits[1].DocID != "clip" || hits[1].At != 9*time.Second {
		t.Errorf("hits[1] = %+v, want the clip by its frame at 9s", hits[1])
	}
	if hits[2].DocID != "checker" {
		t.Errorf("hits[2] = %+v, want the checker last", hits[2])
	}

	hits, err = ix.SearchVideo(ctx, clipFile, 10)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("video hits: %+v", hits)
	if len(hits) != 3 || hits[0].DocID != "clip" {
		t.Fatalf("video hits = %+v, want the clip itself first", hits)
	}
	// each picture is matched by the query frame that shows it; how close
	// the half-size frame comes depends on the model (DINOv2 keeps the fine
	// grid of the checker apart from its half-size copy)
	want := map[string]time.Duration{"red": 9 * time.Second, "checker": 0}
	for _, h := range hits[1:] {
		if h.QueryAt != want[h.DocID] || h.Score < 0.5 {
			t.Errorf("%s matched by the query frame at %v with %.3f, want %v above 0.5", h.DocID, h.QueryAt, h.Score, want[h.DocID])
		}
	}
}

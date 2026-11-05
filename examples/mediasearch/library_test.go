package mediasearch

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/media"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
)

var (
	red    = color.RGBA{220, 30, 30, 255}
	green  = color.RGBA{30, 200, 60, 255}
	blue   = color.RGBA{30, 60, 220, 255}
	red2   = color.RGBA{200, 45, 45, 255}
	green2 = color.RGBA{45, 185, 75, 255}
	blue2  = color.RGBA{45, 75, 200, 255}
)

// pic is a w×h picture split vertically between two colours.
func pic(w, h int, left, right color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			c := left
			if x >= w/2 {
				c = right
			}
			img.Set(x, y, c)
		}
	}
	return img
}

func pngBytes(t *testing.T, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func write(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// fakeSampler stands in for ffmpeg: a "video" is any file whose name is a
// key of frames, and its frames are the pictures listed for it.
type fakeSampler struct {
	frames map[string][]media.Frame
	calls  atomic.Int32
}

func (s *fakeSampler) Sample(_ context.Context, path string) ([]media.Frame, error) {
	s.calls.Add(1)
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	base := filepath.Base(path)
	frames, ok := s.frames[base]
	if !ok && strings.HasPrefix(base, "mediasearch-") { // an uploaded video's temp file
		frames, ok = s.frames["upload"]
	}
	if !ok {
		return nil, errors.New("fake sampler: not a video: " + path)
	}
	return frames, nil
}

// fixture is a media directory with two photos (one in a subdirectory with
// a space in its name), a clip whose frames show the first photo then the
// second, a text file to ignore, and a broken image.
type fixture struct {
	dir     string
	lib     *Library
	sampler *fakeSampler
	emb     *countingEmbedder
}

type countingEmbedder struct {
	langaitest.FakeImageEmbedder
	calls atomic.Int32
}

func (e *countingEmbedder) EmbedImages(ctx context.Context, images []image.Image) ([][]float32, error) {
	e.calls.Add(int32(len(images)))
	return e.FakeImageEmbedder.EmbedImages(ctx, images)
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	dir := t.TempDir()
	write(t, filepath.Join(dir, "red.png"), pngBytes(t, pic(64, 48, red, red)))
	write(t, filepath.Join(dir, "sub dir", "green blue.png"), pngBytes(t, pic(64, 48, green, blue)))
	write(t, filepath.Join(dir, "clip.mp4"), []byte("not really a video"))
	write(t, filepath.Join(dir, "notes.txt"), []byte("ignored"))
	write(t, filepath.Join(dir, "broken.jpg"), []byte("not a jpeg"))
	write(t, filepath.Join(dir, ".hidden", "secret.png"), pngBytes(t, pic(8, 8, red, red)))
	// the clip shows the two photos, slightly re-coloured as a re-encode
	// would, so that a photo scores itself above the clip rather than tie
	sampler := &fakeSampler{frames: map[string][]media.Frame{
		"clip.mp4": {{Image: pic(64, 48, red2, red2), At: 0}, {Image: pic(64, 48, green2, blue2), At: 2 * time.Second}},
	}}
	emb := &countingEmbedder{}
	ix, err := search.NewMediaIndex(emb, vector.NewMemStore(), "examples", "local", "media", search.WithSampler(sampler))
	if err != nil {
		t.Fatal(err)
	}
	cache := t.TempDir()
	lib, err := New(Config{
		Dir: dir, Index: ix, Sampler: sampler, CatalogFile: filepath.Join(cache, "catalog.json"), CacheDir: filepath.Join(cache, "thumbs"),
		Info: Info{Model: "fake", Dimensions: emb.Dimensions(), Provider: "cpu"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &fixture{dir: dir, lib: lib, sampler: sampler, emb: emb}
}

func TestNew_Validates(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Error("empty dir: want error")
	}
	if _, err := New(Config{Dir: "x"}); err == nil {
		t.Error("nil index: want error")
	}
	ix, _ := search.NewMediaIndex(langaitest.FakeImageEmbedder{}, vector.NewMemStore(), "i", "p", "c")
	bad := filepath.Join(t.TempDir(), "catalog.json")
	write(t, bad, []byte("{"))
	if _, err := New(Config{Dir: "x", Index: ix, CatalogFile: bad}); err == nil {
		t.Error("broken catalog: want error")
	}
	lib, err := New(Config{Dir: "x", Index: ix})
	if err != nil {
		t.Fatal(err)
	}
	if lib.Limit() != 12 {
		t.Errorf("default limit = %d, want 12", lib.Limit())
	}
}

// checkCatalog checks what the first scan of the fixture leaves in the catalog.
func checkCatalog(t *testing.T, lib *Library) {
	t.Helper()
	items := lib.Items()
	ids := make([]string, len(items))
	for i, it := range items {
		ids[i] = it.ID
	}
	if want := "broken.jpg clip.mp4 red.png sub dir/green blue.png"; strings.Join(ids, " ") != want {
		t.Errorf("items = %q, want %q", ids, want)
	}
	if it, _ := lib.Item("broken.jpg"); it.Error == "" || it.Kind != KindImage {
		t.Errorf("broken.jpg = %+v, want an error", it)
	}
	if it, _ := lib.Item("clip.mp4"); it.Kind != KindVideo || it.Skipped || it.Error != "" {
		t.Errorf("clip.mp4 = %+v", it)
	}
	if lib.LastScan().At.IsZero() {
		t.Error("LastScan not recorded")
	}
	if _, ok := lib.Path(".hidden/secret.png"); ok {
		t.Error("dot-directory was scanned")
	}
}

func TestScan_IndexesSkipsAndRemoves(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()

	stats, err := f.lib.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Indexed != 3 || stats.Skipped != 0 || stats.Failed != 1 || stats.Removed != 0 {
		t.Errorf("first scan: %+v", stats)
	}
	if got := f.emb.calls.Load(); got != 4 { // two images and two frames
		t.Errorf("embedded %d pictures, want 4", got)
	}
	checkCatalog(t, f.lib)

	// the second scan is all no-ops, the broken file is retried
	f.emb.calls.Store(0)
	stats, err = f.lib.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Indexed != 0 || stats.Skipped != 3 || stats.Failed != 1 {
		t.Errorf("second scan: %+v", stats)
	}
	if got := f.emb.calls.Load(); got != 0 {
		t.Errorf("second scan embedded %d pictures", got)
	}
	if it, _ := f.lib.Item("red.png"); !it.Skipped {
		t.Error("red.png not marked skipped")
	}
}

// TestScan_ChangedAndRemovedFiles: a changed file is re-indexed, a removed
// one deleted, and the catalog file survives a restart.
func TestScan_ChangedAndRemovedFiles(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()
	if _, err := f.lib.Scan(ctx); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(f.dir, "red.png"), pngBytes(t, pic(64, 48, blue, blue)))
	if err := os.Remove(filepath.Join(f.dir, "clip.mp4")); err != nil {
		t.Fatal(err)
	}
	lib2, err := New(f.lib.cfg)
	if err != nil {
		t.Fatal(err)
	}
	stats, err := lib2.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	// lib2 never listed clip.mp4 but its hash was in the catalog: gone
	// hashes are deleted from the index too
	if stats.Indexed != 1 || stats.Skipped != 1 || stats.Removed != 1 {
		t.Errorf("second scan: %+v", stats)
	}
	hits, err := lib2.cfg.Index.SearchImage(ctx, bytes.NewReader(pngBytes(t, pic(64, 48, green, blue))), 10)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.DocID == "clip.mp4" {
			t.Error("clip.mp4 still in the index")
		}
	}
	if _, ok := lib2.Item("clip.mp4"); ok {
		t.Error("clip.mp4 still listed")
	}
	// the first library still lists the clip and removes it on its own scan
	stats, err = f.lib.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Removed != 1 {
		t.Errorf("first library's scan: %+v", stats)
	}
}

func TestScan_OneAtATime(t *testing.T) {
	f := newFixture(t)
	f.lib.scanning.Store(true)
	if _, err := f.lib.Scan(context.Background()); !errors.Is(err, ErrScanning) {
		t.Errorf("err = %v, want ErrScanning", err)
	}
	f.lib.scanning.Store(false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := f.lib.Scan(ctx); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled: err = %v", err)
	}
}

func TestScan_MissingDirectory(t *testing.T) {
	ix, _ := search.NewMediaIndex(langaitest.FakeImageEmbedder{}, vector.NewMemStore(), "i", "p", "c")
	lib, _ := New(Config{Dir: filepath.Join(t.TempDir(), "missing"), Index: ix})
	if _, err := lib.Scan(context.Background()); err == nil {
		t.Error("want error")
	}
}

func TestKindOf(t *testing.T) {
	for name, want := range map[string]string{"a.JPG": KindImage, "b.webp": KindImage, "c.MOV": KindVideo, "d.txt": "", "e": ""} {
		if got := kindOf(name); got != want {
			t.Errorf("kindOf(%q) = %q, want %q", name, got, want)
		}
	}
}

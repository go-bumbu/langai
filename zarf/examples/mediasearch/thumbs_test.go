package mediasearch

import (
	"context"
	"errors"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"
)

func decodeJPEG(t *testing.T, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	img, err := jpeg.Decode(f)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return img
}

func TestThumb_ImageScaledAndCached(t *testing.T) {
	f := scanned(t)
	ctx := context.Background()
	write(t, filepath.Join(f.dir, "big.png"), pngBytes(t, pic(800, 400, red, blue)))
	if _, err := f.lib.Scan(ctx); err != nil {
		t.Fatal(err)
	}
	path, err := f.lib.Thumb(ctx, "big.png")
	if err != nil {
		t.Fatal(err)
	}
	if b := decodeJPEG(t, path).Bounds(); b.Dx() != thumbSide || b.Dy() != thumbSide/2 {
		t.Errorf("thumb size = %v, want %d×%d", b, thumbSide, thumbSide/2)
	}
	small, err := f.lib.Thumb(ctx, "red.png")
	if err != nil {
		t.Fatal(err)
	}
	if b := decodeJPEG(t, small).Bounds(); b.Dx() != 64 || b.Dy() != 48 {
		t.Errorf("small picture was scaled: %v", b)
	}

	// the cache answers the second call even when the source is gone
	if err := os.Remove(filepath.Join(f.dir, "big.png")); err != nil {
		t.Fatal(err)
	}
	again, err := f.lib.Thumb(ctx, "big.png")
	if err != nil || again != path {
		t.Errorf("second call: %q, %v", again, err)
	}

	if _, err := f.lib.Thumb(ctx, "nope.png"); !errors.Is(err, ErrNotFound) {
		t.Errorf("unknown: %v", err)
	}
	if _, err := f.lib.Thumb(ctx, "broken.jpg"); err == nil {
		t.Error("broken: want error")
	}
	if _, err := f.lib.Frames(ctx, "red.png"); !errors.Is(err, ErrNotVideo) {
		t.Errorf("frames of an image: %v", err)
	}
	if _, err := f.lib.Storyboard(ctx, "red.png"); !errors.Is(err, ErrNotVideo) {
		t.Errorf("storyboard of an image: %v", err)
	}
	if _, err := f.lib.Frames(ctx, "nope.mp4"); !errors.Is(err, ErrNotFound) {
		t.Errorf("frames of unknown: %v", err)
	}
}

func TestThumb_VideoSampledOnceForThumbAndStoryboard(t *testing.T) {
	f := scanned(t)
	ctx := context.Background()
	f.sampler.calls.Store(0)
	thumb, err := f.lib.Thumb(ctx, "clip.mp4")
	if err != nil {
		t.Fatal(err)
	}
	if b := decodeJPEG(t, thumb).Bounds(); b.Dx() != 64 || b.Dy() != 48 {
		t.Errorf("video thumb = %v, want the first frame's size", b)
	}
	board, err := f.lib.Frames(ctx, "clip.mp4")
	if err != nil {
		t.Fatal(err)
	}
	if len(board.Times) != 2 || board.Times[1] != 2000 || board.CellH != boardHeight || board.CellW != 160 {
		t.Errorf("board = %+v", board)
	}
	strip, err := f.lib.Storyboard(ctx, "clip.mp4")
	if err != nil {
		t.Fatal(err)
	}
	if b := decodeJPEG(t, strip).Bounds(); b.Dx() != 2*board.CellW || b.Dy() != boardHeight {
		t.Errorf("strip = %v", b)
	}
	if got := f.sampler.calls.Load(); got != 1 {
		t.Errorf("sampled %d times, want 1", got)
	}

	f.lib.cfg.Sampler = nil
	f.lib.cfg.CacheDir = t.TempDir()
	if _, err := f.lib.Thumb(ctx, "clip.mp4"); err == nil {
		t.Error("no sampler: want error")
	}
	f.lib.cfg.CacheDir = ""
	if _, err := f.lib.Thumb(ctx, "red.png"); err == nil {
		t.Error("no cache dir: want error")
	}
}

func TestFit(t *testing.T) {
	if b := fit(pic(10, 5, red, red), 320).Bounds(); b.Dx() != 10 || b.Dy() != 5 {
		t.Errorf("small kept: %v", b)
	}
	if b := fit(pic(300, 900, red, red), 300).Bounds(); b.Dx() != 100 || b.Dy() != 300 {
		t.Errorf("tall scaled: %v", b)
	}
}

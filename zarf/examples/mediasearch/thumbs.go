package mediasearch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"

	"golang.org/x/image/draw"
)

// Sizes of what the page shows.
const (
	thumbSide   = 320 // the longest side of a thumbnail
	boardHeight = 120 // the height of a storyboard cell
	jpegQuality = 80
)

// Board describes a video's storyboard: the sampled frames as one JPEG strip
// of equal cells, with the time of each in milliseconds.
type Board struct {
	CellW int     `json:"cellW"`
	CellH int     `json:"cellH"`
	Times []int64 `json:"times"`
}

// ErrNotVideo is returned by Frames and Storyboard for an image.
var ErrNotVideo = errors.New("mediasearch: not a video")

// Thumb returns the path of the item's thumbnail JPEG, making it when the
// cache has none for the file as it is now (same id, size and mtime). A
// video's thumbnail is its first sampled frame, and the same run makes its
// storyboard.
func (l *Library) Thumb(ctx context.Context, id string) (string, error) {
	it, ok := l.Item(id)
	if !ok {
		return "", ErrNotFound
	}
	return l.cached(ctx, it, ".thumb.jpg")
}

// Storyboard returns the path of a video's storyboard JPEG, see Board.
func (l *Library) Storyboard(ctx context.Context, id string) (string, error) {
	it, ok := l.Item(id)
	if !ok {
		return "", ErrNotFound
	}
	if it.Kind != KindVideo {
		return "", ErrNotVideo
	}
	return l.cached(ctx, it, ".board.jpg")
}

// Frames returns a video's Board.
func (l *Library) Frames(ctx context.Context, id string) (Board, error) {
	it, ok := l.Item(id)
	if !ok {
		return Board{}, ErrNotFound
	}
	if it.Kind != KindVideo {
		return Board{}, ErrNotVideo
	}
	path, err := l.cached(ctx, it, ".board.json")
	if err != nil {
		return Board{}, err
	}
	data, err := os.ReadFile(path) //nolint:gosec // a cache file named after a catalog key
	if err != nil {
		return Board{}, err
	}
	var b Board
	if err := json.Unmarshal(data, &b); err != nil {
		return Board{}, fmt.Errorf("mediasearch: read storyboard %s: %w", path, err)
	}
	return b, nil
}

// cacheKey names the cache files of an item as it is now.
func cacheKey(it Item) string {
	sum := sha256.Sum256(fmt.Appendf(nil, "%s|%d|%d", it.ID, it.Size, it.ModTime.UnixNano()))
	return hex.EncodeToString(sum[:16])
}

// cached returns the cache file with the given suffix for the item, making
// the item's cache files when that one is missing. One item is made at a
// time, which keeps a grid of videos from starting a sampler run each.
func (l *Library) cached(ctx context.Context, it Item, suffix string) (string, error) {
	if l.cfg.CacheDir == "" {
		return "", errors.New("mediasearch: no cache directory")
	}
	path := filepath.Join(l.cfg.CacheDir, cacheKey(it)+suffix)
	l.genMu.Lock()
	defer l.genMu.Unlock()
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	if err := os.MkdirAll(l.cfg.CacheDir, 0o750); err != nil {
		return "", err
	}
	src := filepath.Join(l.cfg.Dir, filepath.FromSlash(it.ID))
	base := filepath.Join(l.cfg.CacheDir, cacheKey(it))
	var err error
	if it.Kind == KindVideo {
		err = l.makeVideoCache(ctx, src, base)
	} else {
		err = makeImageThumb(src, base)
	}
	if err != nil {
		return "", err
	}
	return path, nil
}

func makeImageThumb(src, base string) error {
	f, err := os.Open(src) //nolint:gosec // a file the scan found under the media directory, only decoded
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	img, _, err := image.Decode(f)
	if err != nil {
		return fmt.Errorf("mediasearch: decode %s: %w", src, err)
	}
	return writeJPEG(base+".thumb.jpg", fit(img, thumbSide))
}

func (l *Library) makeVideoCache(ctx context.Context, src, base string) error {
	if l.cfg.Sampler == nil {
		return errors.New("mediasearch: no sampler: videos have no thumbnails")
	}
	frames, err := l.cfg.Sampler.Sample(ctx, src)
	if err != nil {
		return err
	}
	if len(frames) == 0 {
		return fmt.Errorf("mediasearch: %s: no frames", src)
	}
	if err := writeJPEG(base+".thumb.jpg", fit(frames[0].Image, thumbSide)); err != nil {
		return err
	}
	// every cell has the first frame's aspect ratio; a mid-stream size
	// change is squeezed into it
	first := frames[0].Image.Bounds()
	cellW := max(1, (first.Dx()*boardHeight+first.Dy()/2)/max(1, first.Dy()))
	strip := image.NewRGBA(image.Rect(0, 0, cellW*len(frames), boardHeight))
	board := Board{CellW: cellW, CellH: boardHeight, Times: make([]int64, len(frames))}
	for i, fr := range frames {
		cell := image.Rect(i*cellW, 0, (i+1)*cellW, boardHeight)
		draw.CatmullRom.Scale(strip, cell, fr.Image, fr.Image.Bounds(), draw.Src, nil)
		board.Times[i] = fr.At.Milliseconds()
	}
	if err := writeJPEG(base+".board.jpg", strip); err != nil {
		return err
	}
	data, err := json.Marshal(board)
	if err != nil {
		return err
	}
	return os.WriteFile(base+".board.json", data, 0o600)
}

// fit returns src scaled down so that its longest side is at most size,
// keeping the aspect ratio, on white (a JPEG has no transparency).
func fit(src image.Image, size int) *image.RGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if long := max(w, h); long > size {
		w, h = max(1, (w*size+long/2)/long), max(1, (h*size+long/2)/long)
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(dst, dst.Bounds(), image.White, image.Point{}, draw.Src)
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, b, draw.Over, nil)
	return dst
}

// writeJPEG encodes img to path through a temporary file, so a reader never
// sees a half-written picture.
func writeJPEG(path string, img image.Image) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return err
	}
	err = jpeg.Encode(tmp, img, &jpeg.Options{Quality: jpegQuality})
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(tmp.Name(), path)
	}
	if err != nil {
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("mediasearch: write %s: %w", path, err)
	}
	return nil
}

package mediasearch

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/go-bumbu/langai/media"
	"github.com/go-bumbu/langai/search"
)

// Hit is one search result joined with its catalog entry.
type Hit struct {
	Item  Item    `json:"item"`
	Score float64 `json:"score"`
	// AtMs is the time of the matching frame when the hit is a video.
	AtMs int64 `json:"atMs"`
	// QueryAtMs is the time of the query frame that matched when the query
	// is a video.
	QueryAtMs int64 `json:"queryAtMs"`
	// Self marks the hit that is the query item itself, found when the
	// file is embedded again; it is kept and marked rather than dropped,
	// since its score says what "identical" scores like with the model.
	Self bool `json:"self"`
}

// Result is what a search returns.
type Result struct {
	// Query is the catalog item searched with; nil for an uploaded file or
	// a text.
	Query *Item `json:"query,omitempty"`
	// Text is the query text, for a text search.
	Text string `json:"text,omitempty"`
	// Frames is how many query frames were embedded: 1 for an image, 0
	// when Stored.
	Frames int `json:"frames"`
	// Stored says the query was the item's vectors from the index, so
	// nothing was embedded.
	Stored    bool  `json:"stored,omitempty"`
	ElapsedMs int64 `json:"elapsedMs"`
	Hits      []Hit `json:"hits"`
}

// ErrNotFound is returned for an id that is not in the catalog.
var ErrNotFound = errors.New("mediasearch: no such item")

// ErrUnsupported is returned for an uploaded file that is neither an image
// nor a video by its extension.
var ErrUnsupported = errors.New("mediasearch: not an image or a video")

// Similar returns the items that look most like the catalog item id.
// limit <= 0 means the Library's default. By default it searches with the
// vectors the scan stored, so no model runs, and the item itself is left
// out; embed embeds the file again instead, as an upload would be, which
// shows what a query costs with the model and finds the item itself first,
// marked.
func (l *Library) Similar(ctx context.Context, id string, limit int, embed bool) (Result, error) {
	it, ok := l.Item(id)
	if !ok {
		return Result{}, ErrNotFound
	}
	path := filepath.Join(l.cfg.Dir, filepath.FromSlash(id))
	var r Result
	var err error
	switch {
	case !embed:
		start := time.Now()
		var hits []search.MediaMatch
		hits, err = l.cfg.Index.SearchIndexed(ctx, id, l.limit(limit))
		r = l.result(hits, 0, start)
		r.Stored = true
	case it.Kind == KindVideo:
		r, err = l.searchVideo(ctx, path, limit)
	default:
		var f *os.File
		f, err = os.Open(path) //nolint:gosec // the id is a catalog key, so path is a file the scan found under the media directory
		if err != nil {
			return Result{}, err
		}
		defer func() { _ = f.Close() }()
		r, err = l.searchImage(ctx, f, limit)
	}
	if err != nil {
		return Result{}, err
	}
	r.Query = &it
	for i := range r.Hits {
		r.Hits[i].Self = r.Hits[i].Item.ID == id
	}
	return r, nil
}

// Search returns the items that look most like an uploaded file, an image
// or a video by the extension of name. A video is written to a temporary
// file for the sampler and removed afterwards.
func (l *Library) Search(ctx context.Context, name string, r io.Reader, limit int) (Result, error) {
	switch kindOf(name) {
	case KindImage:
		return l.searchImage(ctx, r, limit)
	case KindVideo:
		tmp, err := os.CreateTemp("", "mediasearch-*"+filepath.Ext(name))
		if err != nil {
			return Result{}, err
		}
		defer func() { _ = os.Remove(tmp.Name()) }() //nolint:gosec // the temp file this function just created
		_, err = io.Copy(tmp, r)
		if cerr := tmp.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return Result{}, fmt.Errorf("mediasearch: save upload: %w", err)
		}
		return l.searchVideo(ctx, tmp.Name(), limit)
	}
	return Result{}, ErrUnsupported
}

// SearchText returns the items that look most like what the text describes;
// it needs a model with a text side (see Info.Text).
func (l *Library) SearchText(ctx context.Context, text string, limit int) (Result, error) {
	start := time.Now()
	hits, err := l.cfg.Index.SearchText(ctx, text, l.limit(limit))
	if err != nil {
		return Result{}, err
	}
	r := l.result(hits, 1, start)
	r.Text = text
	return r, nil
}

func (l *Library) searchImage(ctx context.Context, r io.Reader, limit int) (Result, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return Result{}, err
	}
	start := time.Now()
	hits, err := l.cfg.Index.SearchImage(ctx, bytes.NewReader(data), l.limit(limit))
	if err != nil {
		return Result{}, err
	}
	return l.result(hits, 1, start), nil
}

// searchVideo searches with a video the way the index embeds videos: as
// segments through the index's video embedder in the segments mode (Frames
// is then the segment count, from the video's duration), otherwise as the
// sampled frames.
func (l *Library) searchVideo(ctx context.Context, path string, limit int) (Result, error) {
	if l.cfg.Info.Mode == "segments" {
		start := time.Now()
		hits, err := l.cfg.Index.SearchVideo(ctx, path, l.limit(limit))
		if err != nil {
			return Result{}, err
		}
		return l.result(hits, l.segmentCount(ctx, path), start), nil
	}
	if l.cfg.Sampler == nil {
		return Result{}, errors.New("mediasearch: no sampler: videos cannot be searched")
	}
	start := time.Now()
	frames, err := l.cfg.Sampler.Sample(ctx, path)
	if err != nil {
		return Result{}, err
	}
	hits, err := l.cfg.Index.SearchFrames(ctx, frames, l.limit(limit))
	if err != nil {
		return Result{}, err
	}
	return l.result(hits, len(frames), start), nil
}

// segmentCount is how many segments a video is, when the Sampler can
// measure it; 0 when it cannot.
func (l *Library) segmentCount(ctx context.Context, path string) int {
	c, ok := l.cfg.Sampler.(media.Cutter)
	if !ok || l.cfg.Info.SegmentMs <= 0 {
		return 0
	}
	d, err := c.Duration(ctx, path)
	if err != nil || d <= 0 {
		return 0
	}
	segment := time.Duration(l.cfg.Info.SegmentMs) * time.Millisecond
	return int((d + segment - 1) / segment)
}

func (l *Library) limit(limit int) int {
	if limit <= 0 {
		return l.cfg.Limit
	}
	return limit
}

// result joins the matches with the catalog; a match whose item vanished
// since the scan keeps its id only.
func (l *Library) result(matches []search.MediaMatch, frames int, start time.Time) Result {
	r := Result{Frames: frames, ElapsedMs: time.Since(start).Milliseconds(), Hits: make([]Hit, 0, len(matches))}
	for _, m := range matches {
		it, ok := l.Item(m.DocID)
		if !ok {
			it = Item{ID: m.DocID, Name: filepath.Base(m.DocID)}
		}
		r.Hits = append(r.Hits, Hit{Item: it, Score: m.Score, AtMs: m.At.Milliseconds(), QueryAtMs: m.QueryAt.Milliseconds()})
	}
	return r
}

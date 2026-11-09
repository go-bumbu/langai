package mediasearch

import (
	"bytes"
	"context"
	"errors"
	"image"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/sense/search"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/vector"
)

func scanned(t *testing.T) *fixture {
	t.Helper()
	f := newFixture(t)
	if _, err := f.lib.Scan(context.Background()); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestSimilar_ImageFindsItselfAndTheClipFrame(t *testing.T) {
	f := scanned(t)
	res, err := f.lib.Similar(context.Background(), "red.png", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if res.Query == nil || res.Query.ID != "red.png" || res.Frames != 1 || len(res.Hits) < 2 {
		t.Fatalf("result = %+v", res)
	}
	if h := res.Hits[0]; !h.Self || h.Item.ID != "red.png" || h.Score < 0.999 {
		t.Errorf("first hit = %+v, want self with score ~1", h)
	}
	if h := res.Hits[1]; h.Item.ID != "clip.mp4" || h.AtMs != 0 || h.Score < 0.99 || h.Score >= res.Hits[0].Score || h.Self {
		t.Errorf("second hit = %+v, want clip.mp4 at its first frame", h)
	}
	if h := res.Hits[len(res.Hits)-1]; h.Item.ID != "sub dir/green blue.png" || h.Score > 0.99 {
		t.Errorf("last hit = %+v, want the other photo lower", h)
	}
	limited, err := f.lib.Similar(context.Background(), "red.png", 1, true)
	if err != nil || len(limited.Hits) != 1 {
		t.Errorf("limit 1: %d hits, err %v", len(limited.Hits), err)
	}
}

func TestSimilar_VideoFindsItsPictures(t *testing.T) {
	f := scanned(t)
	res, err := f.lib.Similar(context.Background(), "clip.mp4", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if res.Frames != 2 {
		t.Errorf("frames = %d, want 2", res.Frames)
	}
	byID := map[string]Hit{}
	for _, h := range res.Hits {
		byID[h.Item.ID] = h
	}
	if h := byID["clip.mp4"]; !h.Self || h.Score < 0.999 {
		t.Errorf("self = %+v", h)
	}
	if h := byID["red.png"]; h.Score < 0.99 || h.QueryAtMs != 0 || h.AtMs != 0 {
		t.Errorf("red.png = %+v, want found by the frame at 0", h)
	}
	if h := byID["sub dir/green blue.png"]; h.Score < 0.99 || h.QueryAtMs != 2000 {
		t.Errorf("green blue = %+v, want found by the frame at 2 s", h)
	}
}

func TestSimilar_StoredVectorsRunNoModel(t *testing.T) {
	f := scanned(t)
	ctx := context.Background()
	calls := f.emb.calls.Load()
	samples := f.sampler.calls.Load()
	res, err := f.lib.Similar(ctx, "red.png", 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Stored || res.Frames != 0 || res.Query == nil || res.Query.ID != "red.png" || len(res.Hits) < 1 {
		t.Fatalf("result = %+v", res)
	}
	for _, h := range res.Hits {
		if h.Self || h.Item.ID == "red.png" {
			t.Errorf("hit = %+v, want the query item left out", h)
		}
	}
	// a video too, by its stored frames, with no sampler either
	f.lib.cfg.Sampler = nil
	res, err = f.lib.Similar(ctx, "clip.mp4", 0, false)
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]Hit{}
	for _, h := range res.Hits {
		byID[h.Item.ID] = h
	}
	if h := byID["sub dir/green blue.png"]; h.Score < 0.99 || h.QueryAtMs != 2000 {
		t.Errorf("green blue = %+v, want found by the stored frame at 2 s", h)
	}
	if got := f.emb.calls.Load(); got != calls {
		t.Errorf("embedder ran %d times, want none", got-calls)
	}
	if got := f.sampler.calls.Load(); got != samples {
		t.Errorf("sampler ran %d times, want none", got-samples)
	}
	if _, err := f.lib.Similar(ctx, "broken.jpg", 0, false); !errors.Is(err, search.ErrNotIndexed) {
		t.Errorf("unindexed item: %v, want search.ErrNotIndexed", err)
	}
}

func TestSimilar_Errors(t *testing.T) {
	f := scanned(t)
	ctx := context.Background()
	if _, err := f.lib.Similar(ctx, "nope.png", 0, true); !errors.Is(err, ErrNotFound) {
		t.Errorf("unknown id: %v", err)
	}
	if _, err := f.lib.Similar(ctx, "broken.jpg", 0, true); err == nil {
		t.Error("broken image: want error")
	}
	f.lib.cfg.Sampler = nil
	if _, err := f.lib.Similar(ctx, "clip.mp4", 0, true); err == nil || !strings.Contains(err.Error(), "sampler") {
		t.Errorf("no sampler: %v", err)
	}
}

func TestSearch_Upload(t *testing.T) {
	f := scanned(t)
	ctx := context.Background()

	// a copy of a photo at another size is an ordinary hit scoring high
	res, err := f.lib.Search(ctx, "query.png", bytes.NewReader(pngBytes(t, pic(32, 24, red, red))), 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Query != nil || res.Frames != 1 || len(res.Hits) == 0 {
		t.Fatalf("result = %+v", res)
	}
	if h := res.Hits[0]; h.Item.ID != "red.png" || h.Score < 0.99 || h.Self {
		t.Errorf("first hit = %+v", h)
	}

	// a video goes through a temp file, which the fake sampler answers with
	// the clip's frames
	f.sampler.frames["upload"] = f.sampler.frames["clip.mp4"]
	res, err = f.lib.Search(ctx, "upload.mp4", strings.NewReader("video bytes"), 3)
	if err != nil {
		t.Fatal(err)
	}
	if res.Frames != 2 || len(res.Hits) != 3 {
		t.Errorf("video upload = %+v", res)
	}

	if _, err := f.lib.Search(ctx, "notes.txt", strings.NewReader("x"), 0); !errors.Is(err, ErrUnsupported) {
		t.Errorf("text upload: %v", err)
	}
	if _, err := f.lib.Search(ctx, "bad.png", strings.NewReader("x"), 0); err == nil {
		t.Error("undecodable upload: want error")
	}
}

// measuringSampler is the fixture's sampler plus a Cutter's Duration, as the
// ffmpeg Sampler is.
type measuringSampler struct {
	*fakeSampler
	duration time.Duration
}

func (m measuringSampler) Duration(context.Context, string) (time.Duration, error) {
	return m.duration, nil
}

func (measuringSampler) Cut(context.Context, string, time.Duration, time.Duration, string) error {
	return errors.New("not cut in this test")
}

// TestSimilar_SegmentsModeSearchesWholeSegments: in the segments mode a
// video query goes through the index's video embedder, so the clip finds
// itself first and Frames is its segment count.
func TestSimilar_SegmentsModeSearchesWholeSegments(t *testing.T) {
	f := newFixture(t)
	ve := &sensetest.FakeVideoEmbedder{Clips: map[string][]image.Image{"clip.mp4": {pic(64, 48, red, red), pic(64, 48, green, blue)}}}
	ix, err := search.NewMediaIndex(f.emb, vector.NewMemStore(), "examples", "local", "media", search.WithVideoEmbedder(ve), search.WithSampler(f.sampler))
	if err != nil {
		t.Fatal(err)
	}
	f.lib.cfg.Index = ix
	f.lib.cfg.Sampler = measuringSampler{fakeSampler: f.sampler, duration: 15 * time.Second}
	f.lib.cfg.Info.Mode, f.lib.cfg.Info.SegmentMs = "segments", 10000
	ctx := context.Background()
	if _, err := f.lib.Scan(ctx); err != nil {
		t.Fatal(err)
	}
	res, err := f.lib.Similar(ctx, "clip.mp4", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if res.Frames != 2 || len(res.Hits) == 0 || !res.Hits[0].Self || res.Hits[0].Score < 0.999 {
		t.Errorf("segments query: %+v", res)
	}
	byID := map[string]Hit{}
	for _, h := range res.Hits {
		byID[h.Item.ID] = h
	}
	if h := byID["red.png"]; h.QueryAtMs != 0 || h.Score < 0.99 {
		t.Errorf("red.png = %+v, want found by the first segment", h)
	}
	// the sampled frames still make the storyboard
	if _, err := f.lib.Frames(ctx, "clip.mp4"); err != nil {
		t.Errorf("storyboard in segments mode: %v", err)
	}
	// without a measuring sampler the count is unknown
	f.lib.cfg.Sampler = f.sampler
	if res, err := f.lib.Similar(ctx, "clip.mp4", 0, true); err != nil || res.Frames != 0 {
		t.Errorf("unmeasured: frames %d, %v", res.Frames, err)
	}
}

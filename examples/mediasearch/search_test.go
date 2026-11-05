package mediasearch

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
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
	res, err := f.lib.Similar(context.Background(), "red.png", 0)
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
	limited, err := f.lib.Similar(context.Background(), "red.png", 1)
	if err != nil || len(limited.Hits) != 1 {
		t.Errorf("limit 1: %d hits, err %v", len(limited.Hits), err)
	}
}

func TestSimilar_VideoFindsItsPictures(t *testing.T) {
	f := scanned(t)
	res, err := f.lib.Similar(context.Background(), "clip.mp4", 0)
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

func TestSimilar_Errors(t *testing.T) {
	f := scanned(t)
	ctx := context.Background()
	if _, err := f.lib.Similar(ctx, "nope.png", 0); !errors.Is(err, ErrNotFound) {
		t.Errorf("unknown id: %v", err)
	}
	if _, err := f.lib.Similar(ctx, "broken.jpg", 0); err == nil {
		t.Error("broken image: want error")
	}
	f.lib.cfg.Sampler = nil
	if _, err := f.lib.Similar(ctx, "clip.mp4", 0); err == nil || !strings.Contains(err.Error(), "sampler") {
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

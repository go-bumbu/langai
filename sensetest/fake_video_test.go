package sensetest

import (
	"context"
	"errors"
	"image"
	"image/color"
	"slices"
	"testing"
	"time"
)

func TestFakeVideoEmbedder(t *testing.T) {
	red, green := color.RGBA{200, 30, 30, 255}, color.RGBA{30, 200, 30, 255}
	boom := errors.New("boom")
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	clips := map[string][]image.Image{
		"clip.mp4":  {picture(64, 48, red, red), picture(64, 48, green, green)},
		"empty.mp4": nil,
	}
	tests := []struct {
		name     string
		emb      *FakeVideoEmbedder
		ctx      context.Context
		path     string
		segments int
		wantErr  error // nil: any error when fail is set
		fail     bool
	}{
		{name: "listed clip", emb: &FakeVideoEmbedder{Clips: clips}, path: "/videos/clip.mp4", segments: 2},
		{name: "listed without pictures", emb: &FakeVideoEmbedder{Clips: clips}, path: "empty.mp4", segments: 0},
		{name: "not listed", emb: &FakeVideoEmbedder{Clips: clips}, path: "other.mp4", fail: true},
		{name: "zero value", emb: &FakeVideoEmbedder{}, path: "/any/where.mkv", segments: 2},
		{name: "Err", emb: &FakeVideoEmbedder{Err: boom}, path: "clip.mp4", fail: true, wantErr: boom},
		{name: "done context", emb: &FakeVideoEmbedder{}, ctx: cancelled, path: "clip.mp4", fail: true, wantErr: context.Canceled},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := tc.ctx
			if ctx == nil {
				ctx = context.Background()
			}
			segs, err := tc.emb.EmbedVideo(ctx, tc.path)
			if tc.emb.Calls() != 1 {
				t.Errorf("Calls() = %d, want 1", tc.emb.Calls())
			}
			if tc.fail {
				if err == nil || (tc.wantErr != nil && !errors.Is(err, tc.wantErr)) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(segs) != tc.segments {
				t.Fatalf("got %d segments, want %d", len(segs), tc.segments)
			}
			for i, s := range segs {
				if s.Start != time.Duration(i)*FakeVideoSegment || s.End != s.Start+FakeVideoSegment {
					t.Errorf("segment %d spans %v-%v, want %d×%v long ones back to back", i, s.Start, s.End, i, FakeVideoSegment)
				}
				if len(s.Vector) != tc.emb.Dimensions() {
					t.Errorf("segment %d has %d dimensions, want %d", i, len(s.Vector), tc.emb.Dimensions())
				}
			}
		})
	}
}

// TestFakeVideoEmbedder_SharesTheImageSpace: a listed segment is the vector
// FakeImageEmbedder makes of its picture, and the derived videos are
// deterministic per name and differ between names.
func TestFakeVideoEmbedder_SharesTheImageSpace(t *testing.T) {
	ctx := context.Background()
	green := picture(64, 48, color.RGBA{30, 200, 30, 255}, color.RGBA{30, 200, 30, 255})
	segs, err := (&FakeVideoEmbedder{Clips: map[string][]image.Image{"g.mp4": {green}}}).EmbedVideo(ctx, "g.mp4")
	if err != nil {
		t.Fatal(err)
	}
	img, err := FakeImageEmbedder{}.EmbedImages(ctx, []image.Image{green})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(segs[0].Vector, img[0]) {
		t.Error("the segment's vector is not FakeImageEmbedder's of its picture")
	}

	var zero FakeVideoEmbedder
	a1, _ := zero.EmbedVideo(ctx, "/x/a.mp4")
	a2, _ := zero.EmbedVideo(ctx, "/y/a.mp4")
	b, _ := zero.EmbedVideo(ctx, "/x/b.mp4")
	if !slices.Equal(a1[0].Vector, a2[0].Vector) || !slices.Equal(a1[1].Vector, a2[1].Vector) {
		t.Error("the same base name gave two videos")
	}
	if slices.Equal(a1[0].Vector, b[0].Vector) {
		t.Error("two base names gave the same video")
	}
	if zero.Model() != FakeVideoEmbedderModel {
		t.Errorf("Model() = %q", zero.Model())
	}
}

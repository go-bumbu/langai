package langaitest

import (
	"context"
	"errors"
	"image"
	"image/color"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/go-bumbu/langai"
)

// TestFakeEmbedder_FailsOnDemand: the fake fails as a real embedder does, so a
// test of a cancellation or an outage does not pass by accident: with a done
// context, with Err on every call, and with Fail on the calls that hold a text
// it refuses. Every call is counted, the failed ones included.
func TestFakeEmbedder_FailsOnDemand(t *testing.T) {
	boom := errors.New("boom")
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	poison := func(text string) error {
		if strings.Contains(text, "poison") {
			return boom
		}
		return nil
	}
	for _, tc := range []struct {
		name    string
		emb     FakeEmbedder
		ctx     context.Context
		texts   []string
		wantErr error
	}{
		{name: "zero value", texts: []string{"bread", "poison"}},
		{name: "done context", ctx: cancelled, texts: []string{"bread"}, wantErr: context.Canceled},
		{name: "done context before Err", emb: FakeEmbedder{Err: boom}, ctx: cancelled, texts: []string{"bread"}, wantErr: context.Canceled},
		{name: "Err", emb: FakeEmbedder{Err: boom}, texts: []string{"bread"}, wantErr: boom},
		{name: "Err with no texts", emb: FakeEmbedder{Err: boom}, wantErr: boom},
		{name: "Fail on one text of a batch", emb: FakeEmbedder{Fail: poison}, texts: []string{"bread", "a poison text", "butter"}, wantErr: boom},
		{name: "Fail on no text", emb: FakeEmbedder{Fail: poison}, texts: []string{"bread", "butter"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := tc.ctx
			if ctx == nil {
				ctx = context.Background()
			}
			var calls atomic.Int64
			tc.emb.Calls = &calls
			var emb langai.Embedder = tc.emb // still usable as a value
			vecs, err := emb.Embed(ctx, langai.RoleQuery, tc.texts)
			if calls.Load() != 1 {
				t.Errorf("Calls = %d, want 1", calls.Load())
			}
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) || vecs != nil {
					t.Fatalf("Embed = %v, %v; want no vectors and %v", vecs, err, tc.wantErr)
				}
				return
			}
			if err != nil || len(vecs) != len(tc.texts) {
				t.Fatalf("Embed = %d vectors, %v; want %d, nil", len(vecs), err, len(tc.texts))
			}
		})
	}
}

// TestFakeEmbedder_CountsConcurrentCalls: copies of one FakeEmbedder share its
// counter, and it is safe to embed from many goroutines.
func TestFakeEmbedder_CountsConcurrentCalls(t *testing.T) {
	var calls atomic.Int64
	emb := FakeEmbedder{Calls: &calls}
	var wg sync.WaitGroup
	for range 50 {
		wg.Go(func() {
			copied := emb
			if _, err := copied.Embed(context.Background(), langai.RoleQuery, []string{"bread"}); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if calls.Load() != 50 {
		t.Errorf("Calls = %d, want 50", calls.Load())
	}
}

// TestFakeImageEmbedder_FailsOnDemand: like the other fakes it fails with a
// done context's error first and with Err on every call, and counts every
// call, the failed ones included.
func TestFakeImageEmbedder_FailsOnDemand(t *testing.T) {
	boom := errors.New("boom")
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	red := color.RGBA{200, 30, 30, 255}
	for _, tc := range []struct {
		name    string
		emb     FakeImageEmbedder
		ctx     context.Context
		wantErr error
	}{
		{name: "zero value", ctx: context.Background()},
		{name: "done context", ctx: cancelled, wantErr: context.Canceled},
		{name: "done context before Err", emb: FakeImageEmbedder{Err: boom}, ctx: cancelled, wantErr: context.Canceled},
		{name: "Err", emb: FakeImageEmbedder{Err: boom}, ctx: context.Background(), wantErr: boom},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int64
			tc.emb.Calls = &calls
			var emb langai.ImageEmbedder = tc.emb // still usable as a value
			vecs, err := emb.EmbedImages(tc.ctx, []image.Image{picture(8, 8, red, red)})
			if calls.Load() != 1 {
				t.Errorf("Calls = %d, want 1", calls.Load())
			}
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) || vecs != nil {
					t.Fatalf("EmbedImages = %v, %v; want no vectors and %v", vecs, err, tc.wantErr)
				}
				return
			}
			if err != nil || len(vecs) != 1 {
				t.Fatalf("EmbedImages = %d vectors, %v; want 1, nil", len(vecs), err)
			}
		})
	}
}

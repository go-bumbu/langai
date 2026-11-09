package search

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/vector"
)

// TestErrors_EmptyIDIsTheSentinel: every call that takes a document id
// refuses an empty one with ErrEmptyID, before it touches a model or the
// store.
func TestErrors_EmptyIDIsTheSentinel(t *testing.T) {
	ctx := context.Background()
	ix, _ := newTestIndex(t)
	mix, _, _ := newMediaIndex(t, WithSampler(&countingSampler{frames: clipFrames()}))
	img := encoded(t, pic(8, 8, red, blue))
	for name, call := range map[string]func() error{
		"Index":              func() error { return ix.Index(ctx, "", "text") },
		"Delete":             func() error { return ix.Delete(ctx, "") },
		"MediaIndex.Image":   func() error { return mix.IndexImage(ctx, "", bytes.NewReader(img)) },
		"MediaIndex.Video":   func() error { return mix.IndexVideo(ctx, "", "x.mp4") },
		"MediaIndex.Frames":  func() error { return mix.IndexFrames(ctx, "", "", clipFrames()) },
		"MediaIndex.Delete":  func() error { return mix.Delete(ctx, "") },
		"MediaIndex.Indexed": func() error { _, err := mix.SearchIndexed(ctx, "", 5); return err },
	} {
		if err := call(); !errors.Is(err, ErrEmptyID) {
			t.Errorf("%s: %v, want ErrEmptyID", name, err)
		}
	}
}

// TestErrors_CarryThePrefix: every error the package returns says it comes
// from search, whatever produced it underneath.
func TestErrors_CarryThePrefix(t *testing.T) {
	ctx := context.Background()
	ix, _ := newTestIndex(t)
	mix, emb, _ := newMediaIndex(t)
	for name, call := range map[string]func() error{
		"New nil embedder":        func() error { _, err := New(nil, vector.NewMemStore(), "i", "p", "c"); return err },
		"New nil store":           func() error { _, err := New(langaitest.FakeEmbedder{}, nil, "i", "p", "c"); return err },
		"NewMediaIndex nil embed": func() error { _, err := NewMediaIndex(nil, vector.NewMemStore(), "i", "p", "c"); return err },
		"NewMediaIndex nil store": func() error {
			_, err := NewMediaIndex(langaitest.FakeImageEmbedder{}, nil, "i", "p", "c")
			return err
		},
		"Index empty id":      func() error { return ix.Index(ctx, "", "text") },
		"Delete empty id":     func() error { return ix.Delete(ctx, "") },
		"IndexImage empty id": func() error { return mix.IndexImage(ctx, "", strings.NewReader("x")) },
		"IndexImage model down": func() error {
			emb.err = errors.New("model down")
			defer func() { emb.err = nil }()
			return mix.IndexImage(ctx, "x", bytes.NewReader(encoded(t, pic(8, 8, red, blue))))
		},
		"SearchIndexed unknown": func() error { _, err := mix.SearchIndexed(ctx, "missing", 5); return err },
		"store refuses index": func() error {
			bad, err := New(langaitest.FakeEmbedder{}, vector.NewMemStore(), "", "p", "c")
			if err != nil {
				return err
			}
			return bad.Index(ctx, "doc", "goroutines")
		},
	} {
		err := call()
		if err == nil {
			t.Errorf("%s: no error", name)
			continue
		}
		if !strings.HasPrefix(err.Error(), "search: ") {
			t.Errorf("%s: %q lacks the search: prefix", name, err)
		}
	}
}

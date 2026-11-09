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

// TestErrors_InvalidIDs: an id no store can keep (not valid UTF-8, or with
// a NUL byte) is refused by every call that takes one with an error wrapping
// vector.ErrInvalidID, before a model runs; ErrEmptyID is of the same family.
func TestErrors_InvalidIDs(t *testing.T) {
	ctx := context.Background()
	if !errors.Is(ErrEmptyID, vector.ErrInvalidID) {
		t.Error("ErrEmptyID does not wrap vector.ErrInvalidID")
	}
	emb := &batchEmbedder{}
	ix, err := New(emb, vector.NewMemStore(), "t", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	mix, memb, _ := newMediaIndex(t, WithSampler(&countingSampler{frames: clipFrames()}))
	img := encoded(t, pic(8, 8, red, blue))
	for _, bad := range []string{"caf\xe9.txt", "a\x00b"} {
		for name, call := range map[string]func() error{
			"Index":              func() error { return ix.Index(ctx, bad, "text") },
			"Delete":             func() error { return ix.Delete(ctx, "ok", bad) },
			"SearchIndexed":      func() error { _, err := ix.SearchIndexed(ctx, bad, 5); return err },
			"MediaIndex.Image":   func() error { return mix.IndexImage(ctx, bad, bytes.NewReader(img)) },
			"MediaIndex.Video":   func() error { return mix.IndexVideo(ctx, bad, "x.mp4") },
			"MediaIndex.Frames":  func() error { return mix.IndexFrames(ctx, bad, "", clipFrames()) },
			"MediaIndex.Delete":  func() error { return mix.Delete(ctx, bad) },
			"MediaIndex.Indexed": func() error { _, err := mix.SearchIndexed(ctx, bad, 5); return err },
		} {
			if err := call(); !errors.Is(err, vector.ErrInvalidID) || !strings.HasPrefix(err.Error(), "search: ") {
				t.Errorf("%s(%q): %v, want a search: error wrapping vector.ErrInvalidID", name, bad, err)
			}
		}
	}
	if len(emb.sizes) != 0 || memb.calls.Load() != 0 {
		t.Errorf("a model ran for an invalid id: %d text and %d image calls", len(emb.sizes), memb.calls.Load())
	}
}

// TestErrors_CoordsAreCheckedUpFront: New and NewMediaIndex refuse an empty
// index and a partition or collection no store can keep, rather than
// failing on first use.
func TestErrors_CoordsAreCheckedUpFront(t *testing.T) {
	for _, tc := range []struct {
		name                         string
		index, partition, collection string
		want                         error
	}{
		{"empty index", "", "p", "c", vector.ErrEmptyIndex},
		{"bad partition", "i", "caf\xe9", "c", vector.ErrInvalidID},
		{"bad collection", "i", "p", "a\x00b", vector.ErrInvalidID},
	} {
		if _, err := New(langaitest.FakeEmbedder{}, vector.NewMemStore(), tc.index, tc.partition, tc.collection); !errors.Is(err, tc.want) || !strings.HasPrefix(err.Error(), "search: ") {
			t.Errorf("New %s: %v, want a search: error wrapping %v", tc.name, err, tc.want)
		}
		if _, err := NewMediaIndex(langaitest.FakeImageEmbedder{}, vector.NewMemStore(), tc.index, tc.partition, tc.collection); !errors.Is(err, tc.want) || !strings.HasPrefix(err.Error(), "search: ") {
			t.Errorf("NewMediaIndex %s: %v, want a search: error wrapping %v", tc.name, err, tc.want)
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
			_, err := New(langaitest.FakeEmbedder{}, vector.NewMemStore(), "", "p", "c")
			return err
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

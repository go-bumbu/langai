package search

import (
	"bytes"
	"context"
	"image"
	"strings"
	"testing"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/media"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/vector"
)

// lineChunker is a Chunker without a fingerprint: one chunk per non-empty
// line.
type lineChunker struct{}

func (lineChunker) Chunk(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool { return r == '\n' })
}

// wordChunker is another Chunker without a fingerprint: one chunk per word.
type wordChunker struct{}

func (wordChunker) Chunk(text string) []string { return strings.Fields(text) }

// TestIndex_SettingsThatShapeTheVectorsReindex: the same text is indexed
// again after a change of chunker settings, of chunker, or of model, since
// each gives other vectors; with the same settings it stays a no-op.
func TestIndex_SettingsThatShapeTheVectorsReindex(t *testing.T) {
	text := "first paragraph with some words\n\nsecond paragraph with more words"
	ctx := context.Background()
	store := vector.NewMemStore()
	emb := &countingEmbedder{} // names no model
	for _, step := range []struct {
		name    string
		named   bool
		opts    []Option
		reindex bool
	}{
		{"first index, no model name", false, nil, true},
		{"same settings", false, []Option{WithChunker(&MarkdownChunker{})}, false},
		{"default MaxChars spelled out", false, []Option{WithChunker(&MarkdownChunker{MaxChars: defaultMaxChars})}, false},
		{"MaxChars changed", false, []Option{WithChunker(&MarkdownChunker{MaxChars: 40})}, true},
		{"custom chunker", false, []Option{WithChunker(lineChunker{})}, true},
		{"same custom chunker", false, []Option{WithChunker(lineChunker{})}, false},
		{"another custom chunker", false, []Option{WithChunker(wordChunker{})}, true},
		{"a named model now", true, []Option{WithChunker(wordChunker{})}, true},
	} {
		var e sense.Embedder = emb
		if step.named {
			e = namedCounting{emb}
		}
		ix, err := New(e, store, "t", "p", "c", step.opts...)
		if err != nil {
			t.Fatal(err)
		}
		before := emb.calls
		if err := ix.Index(ctx, "doc", text); err != nil {
			t.Fatalf("%s: %v", step.name, err)
		}
		if got := emb.calls > before; got != step.reindex {
			t.Errorf("%s: re-embedded = %v, want %v", step.name, got, step.reindex)
		}
	}
}

// namedCounting is countingEmbedder under the fake's model name.
type namedCounting struct{ *countingEmbedder }

func (namedCounting) Model() string { return sensetest.FakeEmbedderModel }

// TestMarkdownChunker_Fingerprint: the fingerprint names the effective
// MaxChars, so the zero value and the default spelled out agree.
func TestMarkdownChunker_Fingerprint(t *testing.T) {
	if a, b := (&MarkdownChunker{}).Fingerprint(), (&MarkdownChunker{MaxChars: defaultMaxChars}).Fingerprint(); a != b {
		t.Errorf("zero value %q, default %q: want equal", a, b)
	}
	if a, b := (&MarkdownChunker{}).Fingerprint(), (&MarkdownChunker{MaxChars: 40}).Fingerprint(); a == b {
		t.Errorf("MaxChars 1200 and 40 share the fingerprint %q", a)
	}
}

// anonymousImages is the fake image embedder without a model name.
type anonymousImages struct{ *countingImageEmbedder }

func (a anonymousImages) Model() string { return "" }

// describedSampler is a Sampler that describes its settings.
type describedSampler struct {
	*countingSampler
	every string
}

func (d describedSampler) Fingerprint() string { return "every " + d.every }

// TestMediaIndex_SettingsThatShapeTheVectorsReindex: the same bytes are
// embedded again under a newly named model, and the same video sampled
// again when the sampler describes other settings.
func TestMediaIndex_SettingsThatShapeTheVectorsReindex(t *testing.T) {
	ctx := context.Background()
	store := vector.NewMemStore()
	emb := &countingImageEmbedder{}
	img := encoded(t, pic(64, 48, red, blue))

	anonymous, err := NewMediaIndex(anonymousImages{emb}, store, "t", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	named, err := NewMediaIndex(emb, store, "t", "p", "c")
	if err != nil {
		t.Fatal(err)
	}
	for _, ix := range []*MediaIndex{anonymous, anonymous, named, named} {
		if err := ix.IndexImage(ctx, "a", bytes.NewReader(img)); err != nil {
			t.Fatal(err)
		}
	}
	if n := emb.calls.Load(); n != 2 {
		t.Errorf("%d embed calls, want 2: once per model", n)
	}

	sampler := &countingSampler{frames: []media.Frame{{Image: image.Image(pic(64, 48, green, red))}}}
	path := tempFile(t, "clip.mp4", []byte("clip"))
	for i, every := range []string{"5s", "5s", "1s"} {
		ix, err := NewMediaIndex(emb, store, "t", "p", "c", WithSampler(describedSampler{sampler, every}))
		if err != nil {
			t.Fatal(err)
		}
		if err := ix.IndexVideo(ctx, "clip", path); err != nil {
			t.Fatal(err)
		}
		if want := []int{1, 1, 2}[i]; sampler.calls != want {
			t.Errorf("sampler every %s: %d sample calls, want %d", every, sampler.calls, want)
		}
	}
}

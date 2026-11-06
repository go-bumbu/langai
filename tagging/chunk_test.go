package tagging

import (
	"context"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
)

func TestChunk_ShortTextsComeBackWhole(t *testing.T) {
	for _, c := range []struct {
		text string
		max  int
		want []string
	}{
		{"", 10, nil},
		{"   \n ", 10, nil},
		{" short text ", 100, []string{"short text"}},
		{"exactly ten", 11, []string{"exactly ten"}},
		{"no limit at all, however long", 0, []string{"no limit at all, however long"}},
		{"negative limit", -5, []string{"negative limit"}},
	} {
		if got := Chunk(c.text, c.max); !slices.Equal(got, c.want) {
			t.Errorf("Chunk(%q, %d) = %q, want %q", c.text, c.max, got, c.want)
		}
	}
}

func TestChunk_PiecesFitAreEvenAndLoseNoWord(t *testing.T) {
	text := strings.Repeat("alpha beta gamma delta epsilon ", 100) // 3099 bytes once trimmed
	for _, limit := range []int{50, 333, 1000, 2000, 3098} {
		pieces := Chunk(text, limit)
		if len(pieces) < 2 {
			t.Fatalf("limit %d: %d pieces", limit, len(pieces))
		}
		for _, p := range pieces {
			if len(p) > limit {
				t.Errorf("limit %d: a piece of %d bytes", limit, len(p))
			}
		}
		if got, want := strings.Fields(strings.Join(pieces, " ")), strings.Fields(text); !slices.Equal(got, want) {
			t.Errorf("limit %d: words changed", limit)
		}
		// about equal: no piece is under a third of the longest
		longest := 0
		for _, p := range pieces {
			longest = max(longest, len(p))
		}
		for _, p := range pieces {
			if len(p)*3 < longest {
				t.Errorf("limit %d: uneven pieces %d and %d bytes", limit, len(p), longest)
			}
		}
	}
}

func TestChunk_CutsRunesWithoutSpaces(t *testing.T) {
	text := strings.Repeat("ü", 100) + strings.Repeat("日", 50) // 200 + 150 bytes, no whitespace
	pieces := Chunk(text, 64)
	if strings.Join(pieces, "") != text {
		t.Fatalf("pieces do not rejoin into the text: %q", pieces)
	}
	for _, p := range pieces {
		if len(p) > 64 || !utf8.ValidString(p) {
			t.Errorf("piece %q: %d bytes, valid %v", p, len(p), utf8.ValidString(p))
		}
	}
	// a limit under one rune still makes progress, a rune at a time
	if got := Chunk("日日", 2); !slices.Equal(got, []string{"日", "日"}) {
		t.Errorf("limit under a rune: %q", got)
	}
}

// truncating reads only the first n bytes of every text, as a real embedder
// reads only its window of tokens.
type truncating struct {
	langaitest.FakeEmbedder
	n int
}

func (e truncating) Embed(ctx context.Context, role langai.Role, texts []string) ([][]float32, error) {
	cut := make([]string, len(texts))
	for i, s := range texts {
		if len(s) > e.n {
			s = s[:e.n]
		}
		cut[i] = s
	}
	return e.FakeEmbedder.Embed(ctx, role, cut)
}

// longItem is about filler for most of its length and about bread at the end,
// past what the truncating embedder reads.
var longItem = strings.Repeat("lorem ipsum dolor sit amet ", 20) + strings.Repeat("sourdough bread loaf crust ", 6)

func newChunkScope(t *testing.T, chunk int) *Scope {
	t.Helper()
	cat := &fakeCatalog{tags: []CatalogTag{{"bread", "sourdough bread", "1"}}}
	tg, err := NewTagger(truncating{n: 200}, cat, NewMemCache(), TaggerConfig{Model: "fake", NameMinScore: 0.2, MemberMinScore: 0.2, ChunkBytes: chunk})
	if err != nil {
		t.Fatal(err)
	}
	return tg.Scope("ix", "p")
}

func TestScope_ChunkBytesReadsTheWholeItem(t *testing.T) {
	ctx := context.Background()
	for _, c := range []struct {
		chunk int
		want  []string
	}{
		{0, nil},                 // the embedder sees filler only
		{180, []string{"bread"}}, // every piece is read, the last one is about bread
	} {
		got, err := newChunkScope(t, c.chunk).Suggest(ctx, "1", longItem)
		if err != nil || !slices.Equal(got, c.want) {
			t.Errorf("ChunkBytes %d: Suggest = %v, %v; want %v", c.chunk, got, err, c.want)
		}
	}
}

func TestNewTagger_RefusesNegativeChunkBytes(t *testing.T) {
	if _, err := NewTagger(langaitest.FakeEmbedder{}, &fakeCatalog{}, NewMemCache(), TaggerConfig{Model: "m", ChunkBytes: -1}); err == nil {
		t.Error("want an error")
	}
}

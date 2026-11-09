package tagging

import (
	"context"
	"math"
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

func TestItemScore_OnePieceIsItsCosine(t *testing.T) {
	tag := []float32{1, 0}
	if got := mustItemScore(t, [][]float32{{0.6, 0.8}}, tag); math.Abs(got-0.6) > 1e-6 {
		t.Errorf("itemScore = %v, want 0.6", got)
	}
}

// TestItemScore_TwoPiecesTakeTheBestOfThemAndTheirMean: a text of two pieces
// is read by halves and whole, so a topic neither half holds whole still
// scores by the centroid.
func TestItemScore_TwoPiecesTakeTheBestOfThemAndTheirMean(t *testing.T) {
	tag := []float32{1, 1, 0}
	halves := [][]float32{{1, 0, 1}, {0, 1, 1}} // each half has one side of the tag
	whole := cosine(mustCentroid(t, halves), tag)
	if got := mustItemScore(t, halves, tag); math.Abs(got-whole) > 1e-9 || whole <= cosine(halves[0], tag) {
		t.Errorf("itemScore = %v, want the centroid's %v, above a half's %v", got, whole, cosine(halves[0], tag))
	}
	sharp := [][]float32{{1, 1, 0}, {0, 0, 1}} // one half is the tag
	if got := mustItemScore(t, sharp, tag); math.Abs(got-1) > 1e-9 {
		t.Errorf("itemScore = %v, want the matching half's 1", got)
	}
}

// TestItemScore_LongerTextsTakeTheMeanOfTheTwoBest: one paragraph alone does
// not carry a tag on a long text; two pieces about it do, the rest of the text
// not diluting them.
func TestItemScore_LongerTextsTakeTheMeanOfTheTwoBest(t *testing.T) {
	tag := []float32{1, 0}
	off := []float32{0, 1}
	on := []float32{1, 0}
	near := []float32{0.6, 0.8}
	for _, c := range []struct {
		name   string
		pieces [][]float32
		want   float64
	}{
		{"one piece about it", [][]float32{off, on, off, off}, 0.5},
		{"two pieces about it", [][]float32{off, on, off, on, off, off}, 1},
		{"the second best counts", [][]float32{near, off, on}, 0.8},
	} {
		if got := mustItemScore(t, c.pieces, tag); math.Abs(got-c.want) > 1e-6 {
			t.Errorf("%s: itemScore = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestMemberVector_IsThePieceClosestToTheName(t *testing.T) {
	name := []float32{1, 0}
	pieces := [][]float32{{0, 1}, {0.8, 0.6}, {0.6, 0.8}}
	if got := memberVector(pieces, name); !slices.Equal(got, pieces[1]) {
		t.Errorf("memberVector = %v, want %v", got, pieces[1])
	}
}

// longMember is about filler but for one piece about bread.
var longMember = strings.Repeat("lorem ipsum dolor sit amet ", 8) + "sourdough bread loaf crust " + strings.Repeat("lorem ipsum dolor sit amet ", 8)

// TestEngine_LongMemberTeachesWithItsBestPiece: a member of several pieces
// enters its tag's blend as the piece closest to the tag's name, not as the
// mean of its pieces, which sits nearer the filler.
func TestEngine_LongMemberTeachesWithItsBestPiece(t *testing.T) {
	cfg := engineCfg
	cfg.ChunkBytes = 40
	cat := &fakeCatalog{tags: []CatalogTag{{"bread", "sourdough bread", "1"}}, members: map[string][]string{"bread": {longMember}}}
	r := newRig(t, cfg, cat, nil, nil)
	r.scores(t, "1", "bread at home")
	cands, err := r.ix.Candidates(context.Background(), "1", "10", nil)
	if err != nil || len(cands) != 1 {
		t.Fatalf("Candidates = %v, %v", cands, err)
	}
	ctx := context.Background()
	emb := langaitest.FakeEmbedder{}
	nameVecs, err := emb.Embed(ctx, langai.RoleDocument, []string{"sourdough bread"})
	if err != nil {
		t.Fatal(err)
	}
	nameVec := nameVecs[0]
	pieces := Chunk(longMember, cfg.ChunkBytes)
	vecs, err := emb.Embed(ctx, langai.RoleQuery, pieces)
	if err != nil {
		t.Fatal(err)
	}
	want := blend(nameVec, [][]float32{memberVector(vecs, nameVec)}, DefaultSmoothingK)
	if !slices.Equal(cands[0].Vec, want) {
		t.Errorf("the tag's vector is not the blend of its name and the member's best piece")
	}
	if mean := blend(nameVec, [][]float32{mustCentroid(t, vecs)}, DefaultSmoothingK); slices.Equal(want, mean) {
		t.Fatal("test is vacuous: the best piece blends like the mean")
	}
}

// TestEngine_PiecesAreCachedOnTheirOwn: a long text is cached piece by piece
// under the model, so a text that is one of its pieces costs no embedding, and
// neither does the long text again without ChunkBytes' pieces changing.
func TestEngine_PiecesAreCachedOnTheirOwn(t *testing.T) {
	cfg := engineCfg
	cfg.ChunkBytes = 40
	cache := NewMemCache()
	r := newRig(t, cfg, &fakeCatalog{tags: breadAndJapan("1")}, cache, nil)
	r.scores(t, "1", longMember)
	pieces := Chunk(longMember, cfg.ChunkBytes)
	distinct := len(slices.Compact(slices.Sorted(slices.Values(pieces))))
	if n := r.emb.count(langai.RoleQuery); n != distinct {
		t.Fatalf("embedded %d query texts, want its %d distinct pieces", n, distinct)
	}
	plain := newRig(t, engineCfg, &fakeCatalog{tags: breadAndJapan("1")}, cache, nil)
	plain.scores(t, "1", pieces[1])
	if n := plain.emb.count(langai.RoleQuery); n != 0 {
		t.Errorf("a Tagger without ChunkBytes embedded %d texts for a cached piece, want 0", n)
	}
}

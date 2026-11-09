package tagging

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
)

// mustItemScore is itemScore for vectors of one length.
func mustItemScore(t *testing.T, pieces [][]float32, tag []float32) float64 {
	t.Helper()
	s, err := itemScore(pieces, tag)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// mustScoreVectors is scoreVectors for vectors of one length.
func mustScoreVectors(t *testing.T, pieces [][]float32, tags []TagVector, onePath bool) []TagScore {
	t.Helper()
	ss, err := scoreVectors(pieces, tags, onePath)
	if err != nil {
		t.Fatal(err)
	}
	return ss
}

// mustCentroid is centroid for vectors of one length.
func mustCentroid(t *testing.T, vecs [][]float32) []float32 {
	t.Helper()
	c, err := centroid(vecs)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// dimsRig is a rig whose embedder answers with vectors of dims entries, and
// says so unless noDims.
func dimsRig(t *testing.T, cfg TaggerConfig, cat *fakeCatalog, cache EmbeddingCache, dims int, noDims bool) *rig {
	t.Helper()
	r := newRig(t, cfg, cat, cache, nil)
	r.emb.inner, r.emb.noDims = langaitest.FakeEmbedder{Dims: dims}, noDims
	return r
}

// TestEngine_CachedVectorsOfAnotherLengthAreEmbeddedAgain: the cache holds
// what a 16-dimensional embedder made under a Model name, and the application
// now runs an 8-dimensional one under the same name. The cached vectors used
// to be taken as they were: every score came out 0, with no error, and an item
// of two pieces panicked. A cached vector that is not as long as the
// embedder's Dimensions is now a miss, embedded again and overwritten.
func TestEngine_CachedVectorsOfAnotherLengthAreEmbeddedAgain(t *testing.T) {
	const twoPieces = "Croissant dough with cold butter, folded" // two pieces at 20 bytes
	for _, chunkBytes := range []int{0, 20} {
		cfg := engineCfg
		cfg.ChunkBytes = chunkBytes
		cache := NewMemCache()
		dimsRig(t, cfg, breadWithRecipes("1"), cache, 16, false).scores(t, "1", twoPieces)

		r := dimsRig(t, cfg, breadWithRecipes("1"), cache, 8, false)
		got := r.scores(t, "1", twoPieces)
		want := dimsRig(t, cfg, breadWithRecipes("1"), nil, 8, false).scores(t, "1", twoPieces)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("ChunkBytes %d: scores %+v, want those over an empty cache, %+v", chunkBytes, got, want)
		}
		if s := scoreOf(t, got, "bread"); s.Score == 0 {
			t.Errorf("ChunkBytes %d: bread scores 0: vectors of two lengths were compared", chunkBytes)
		}
		for _, piece := range itemPieces(twoPieces, chunkBytes) {
			k := KeyFor("fake", langai.RoleQuery, piece)
			held, err := cache.Get(context.Background(), "1", []CacheKey{k})
			if err != nil || len(held[k]) != 8 {
				t.Errorf("ChunkBytes %d: the cache holds %d dimensions for %q (%v), want the new 8", chunkBytes, len(held[k]), piece, err)
			}
		}
	}
}

// TestEngine_VectorsOfTwoLengthsAreAnErrorNotAScore: an embedder that does not
// say its Dimensions cannot have its cached vectors checked, so a text cached
// at one length can meet tags built at another. The cosine of the two is not
// 0, it is undefined: the call fails, naming the tag and both lengths, where
// it used to score 0 or, for an item of two pieces, panic.
func TestEngine_VectorsOfTwoLengthsAreAnErrorNotAScore(t *testing.T) {
	const twoPieces = "Croissant dough with cold butter, folded"
	for _, chunkBytes := range []int{0, 20} {
		cfg := engineCfg
		cfg.ChunkBytes = chunkBytes
		cache := NewMemCache()
		// the text is cached at 16 dimensions, over a catalog with no tags
		dimsRig(t, cfg, &fakeCatalog{}, cache, 16, true).scores(t, "1", twoPieces)

		r := dimsRig(t, cfg, &fakeCatalog{tags: breadAndJapan("1")}, cache, 8, true)
		ss, err := r.e.scores(context.Background(), "1", "10", "1", twoPieces)
		if err == nil || ss != nil {
			t.Fatalf("ChunkBytes %d: scores %+v, %v; want no scores and an error", chunkBytes, ss, err)
		}
		for _, part := range []string{`"bread"`, "16", "8", "dimensions"} {
			if !strings.Contains(err.Error(), part) {
				t.Errorf("ChunkBytes %d: err = %v, want it to hold %q", chunkBytes, err, part)
			}
		}
	}
}

func TestCentroid_RefusesVectorsOfDifferentLengths(t *testing.T) {
	if c, err := centroid(nil); c != nil || err != nil {
		t.Errorf("centroid of nothing = %v, %v; want nil, nil", c, err)
	}
	c, err := centroid([][]float32{{1, 0}, {0, 1}})
	if err != nil || len(c) != 2 || c[0] != c[1] || c[0] <= 0 {
		t.Errorf("centroid = %v, %v; want the unit mean", c, err)
	}
	for _, ragged := range [][][]float32{{{1, 0}, {1}}, {{1}, {1, 0}}, {{1, 0}, {}}} {
		if c, err := centroid(ragged); err == nil || c != nil {
			t.Errorf("centroid(%v) = %v, %v; want an error", ragged, c, err)
		}
	}
}

func TestItemScore_RefusesVectorsOfDifferentLengths(t *testing.T) {
	tag := []float32{1, 0, 0}
	for name, pieces := range map[string][][]float32{
		"one short piece":       {{1, 0}},
		"one of two is short":   {{1, 0, 0}, {1, 0}},
		"both of two are long":  {{1, 0, 0, 0}, {1, 0, 0, 0}},
		"one of three is short": {{1, 0, 0}, {0, 1, 0}, {1}},
	} {
		if s, err := itemScore(pieces, tag); err == nil {
			t.Errorf("%s: itemScore = %v, want an error", name, s)
		}
	}
	if _, err := queryVector([][]float32{{1, 0}, {1}}); err == nil {
		t.Error("queryVector of two pieces of different lengths: want an error")
	}
}

func TestScoreVectors_NamesTheTagOfAnotherLength(t *testing.T) {
	tags := []TagVector{{ID: "a", Vec: []float32{1, 0}, Cutoff: 0.5}, {ID: "b", Vec: []float32{1, 0, 0}, Cutoff: 0.5}}
	ss, err := scoreVectors([][]float32{{1, 0}}, tags, true)
	if err == nil || ss != nil || !strings.Contains(err.Error(), `"b"`) {
		t.Fatalf("scoreVectors = %+v, %v; want no scores and an error naming tag b", ss, err)
	}
}

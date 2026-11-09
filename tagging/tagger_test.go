package tagging

import (
	"context"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/go-bumbu/langai/langaitest"
)

// newTestTagger returns a Tagger over FakeEmbedder, cat and a MemCache,
// failing the test if NewTagger does.
func newTestTagger(t *testing.T, cat Catalog, cfg TaggerConfig, opts ...TaggerOption) *Tagger {
	t.Helper()
	tg, err := NewTagger(langaitest.FakeEmbedder{}, cat, NewMemCache(), cfg, opts...)
	if err != nil {
		t.Fatal(err)
	}
	return tg
}

func TestNewTagger_RequiresEmbedderCatalogCacheAndModel(t *testing.T) {
	emb, cat, cache := langaitest.FakeEmbedder{}, &fakeCatalog{}, NewMemCache()
	for _, tc := range []struct {
		name string
		err  func() error
	}{
		{"nil embedder", func() error { _, err := NewTagger(nil, cat, cache, TaggerConfig{Model: "m"}); return err }},
		{"nil catalog", func() error { _, err := NewTagger(emb, nil, cache, TaggerConfig{Model: "m"}); return err }},
		{"nil cache", func() error { _, err := NewTagger(emb, cat, nil, TaggerConfig{Model: "m"}); return err }},
		{"no model", func() error { _, err := NewTagger(emb, cat, cache, TaggerConfig{}); return err }},
	} {
		if tc.err() == nil {
			t.Errorf("%s: NewTagger succeeded, want an error", tc.name)
		}
	}
}

func TestNew_Defaults(t *testing.T) {
	tg := newTestTagger(t, &fakeCatalog{}, TaggerConfig{Model: "m"})
	if tg.e.cfg.NameMinScore != 0.42 {
		t.Fatalf("NameMinScore = %v, want default 0.42", tg.e.cfg.NameMinScore)
	}
	if tg.e.cfg.MemberMinScore != 0.58 {
		t.Fatalf("MemberMinScore = %v, want default 0.58", tg.e.cfg.MemberMinScore)
	}
	if tg.e.cfg.AllPathLevels {
		t.Fatal("AllPathLevels is on by default, want one tag per path")
	}
	if tg.e.cfg.SmoothingK != 3 {
		t.Fatalf("SmoothingK = %v, want default 3", tg.e.cfg.SmoothingK)
	}
}

// TestNew_RejectsNonFiniteSettings: a minimum score that is not a finite
// number would silently let the tags it weighs in match every item (-Inf) or
// none (NaN, +Inf), a SmoothingK of +Inf would leave every member out of its
// tag's vector, and a NaN one would let no tag with members match, so New
// refuses them, naming the setting. It checks after the defaults are applied:
// a SmoothingK of -Inf is below zero, and takes DefaultSmoothingK like any
// other. scoreVectors and storedCutoff still fail closed on a NaN, for a
// Tagger built some other way (see their tests).
func TestNew_RejectsNonFiniteSettings(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	for _, tc := range []struct {
		setting string
		cfg     TaggerConfig
	}{
		{"NameMinScore", TaggerConfig{Model: "m", NameMinScore: nan}},
		{"NameMinScore", TaggerConfig{Model: "m", NameMinScore: inf}},
		{"NameMinScore", TaggerConfig{Model: "m", NameMinScore: -inf}},
		{"MemberMinScore", TaggerConfig{Model: "m", MemberMinScore: nan}},
		{"MemberMinScore", TaggerConfig{Model: "m", MemberMinScore: inf}},
		{"MemberMinScore", TaggerConfig{Model: "m", MemberMinScore: -inf}},
		{"SmoothingK", TaggerConfig{Model: "m", SmoothingK: nan}},
		{"SmoothingK", TaggerConfig{Model: "m", SmoothingK: inf}},
	} {
		_, err := NewTagger(langaitest.FakeEmbedder{}, &fakeCatalog{}, NewMemCache(), tc.cfg)
		if err == nil || !strings.Contains(err.Error(), "TaggerConfig."+tc.setting) {
			t.Errorf("NewTagger with %+v = %v, want an error naming TaggerConfig.%s", tc.cfg, err, tc.setting)
		}
	}
	tg := newTestTagger(t, &fakeCatalog{}, TaggerConfig{Model: "m", SmoothingK: -inf})
	if tg.e.cfg.SmoothingK != DefaultSmoothingK {
		t.Errorf("SmoothingK after New with -Inf = %v, want the default %v, as for any value below zero", tg.e.cfg.SmoothingK, DefaultSmoothingK)
	}
}

func TestScope_SuggestIsThePickedScores(t *testing.T) {
	ctx := context.Background()
	cat := &fakeCatalog{tags: []CatalogTag{
		{"bread", "bread baking", "1"}, {"travel", "travel hiking trail", "1"}, {"tax", "income tax", "1"},
	}}
	s := newTestTagger(t, cat, TaggerConfig{Model: "fake", NameMinScore: 0.2, MemberMinScore: 0.2}).Scope(Space{"ix", "p"})
	const text = "baking bread at home, then a hiking trail"
	got, err := s.Scores(ctx, "1", text)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("Scores = %+v, want all three tags", got)
	}
	var picked []string
	for i, sc := range got {
		if i > 0 && sc.Score > got[i-1].Score {
			t.Errorf("not best first: %+v", got)
		}
		if sc.Picked() {
			picked = append(picked, sc.ID)
		}
	}
	want, err := s.Suggest(ctx, "1", text)
	if err != nil || !slices.Equal(picked, want) {
		t.Errorf("picked %v, Suggest %v (%v)", picked, want, err)
	}
	if slices.Contains(picked, "tax") || !slices.Contains(picked, "bread") {
		t.Errorf("picked %v from %+v", picked, got)
	}
	for _, tc := range []struct{ name, revision, text string }{
		{"a blank text", "1", " \n"},
		{"an empty revision", "", text},
	} {
		if _, err := s.Scores(ctx, tc.revision, tc.text); err == nil {
			t.Errorf("Scores of %s: want an error", tc.name)
		}
		if ids, err := s.Suggest(ctx, tc.revision, tc.text); err == nil || ids != nil {
			t.Errorf("Suggest of %s = %v, %v; want no ids and an error", tc.name, ids, err)
		}
	}
	if _, err := s.tg.Scope(Space{"", "p"}).Suggest(ctx, "1", text); err == nil {
		t.Error("Suggest in an empty index: want an error")
	}
}

func TestScope_WarmFillsTheCacheAndTheIndex(t *testing.T) {
	ctx := context.Background()
	emb := &tallyEmbedder{}
	cat := &fakeCatalog{tags: []CatalogTag{{"bread", "cooking / bread", "1"}}}
	tg, err := NewTagger(emb, cat, NewMemCache(), TaggerConfig{Model: "fake", NameMinScore: 0.3, MemberMinScore: 0.4})
	if err != nil {
		t.Fatal(err)
	}
	s := tg.Scope(Space{"ix", "p"})
	const text = "Baking bread at home: a cooking guide"
	if err := s.Warm(ctx, "1", text, "  "); err != nil {
		t.Fatal(err)
	}
	warmed := emb.total()
	got, err := s.Suggest(ctx, "1", text)
	if err != nil || !slices.Equal(got, []string{"bread"}) {
		t.Fatalf("Suggest after Warm = %v, %v; want [bread]", got, err)
	}
	if emb.total() != warmed {
		t.Errorf("Suggest after Warm embedded %d texts, want none", emb.total()-warmed)
	}
	if err := s.Warm(ctx, ""); err == nil {
		t.Error("Warm at an empty revision: want an error")
	}
}

// countingIndex is a TagIndex that counts its Candidates calls.
type countingIndex struct {
	TagIndex
	mu         sync.Mutex
	candidates int
}

func (c *countingIndex) Candidates(ctx context.Context, index, partition string, vec []float32) ([]TagVector, error) {
	c.mu.Lock()
	c.candidates++
	c.mu.Unlock()
	return c.TagIndex.Candidates(ctx, index, partition, vec)
}

func TestWithTagIndex_IsUsed(t *testing.T) {
	ctx := context.Background()
	ix := &countingIndex{TagIndex: NewMemIndex(0)}
	cat := &fakeCatalog{tags: []CatalogTag{{"bread", "cooking / bread", "1"}}}
	s := newTestTagger(t, cat, TaggerConfig{Model: "fake"}, WithTagIndex(ix)).Scope(Space{"ix", "p"})
	for i := range 3 {
		if _, err := s.Suggest(ctx, "1", "Baking bread at home"); err != nil {
			t.Fatal(err)
		}
		if ix.candidates != i+1 {
			t.Fatalf("after %d Suggests, Candidates was called %d times", i+1, ix.candidates)
		}
	}
	// a nil index keeps the default
	if _, err := newTestTagger(t, cat, TaggerConfig{Model: "fake"}, WithTagIndex(nil)).Scope(Space{"ix", "p"}).Suggest(ctx, "1", "bread"); err != nil {
		t.Errorf("Suggest with WithTagIndex(nil): %v", err)
	}
}

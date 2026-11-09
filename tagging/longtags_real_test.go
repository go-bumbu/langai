package tagging

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/provider/yzma"
	"github.com/go-bumbu/langai/zarf/eval"
)

// longCatalog is a space of name-only tags, one per distinct topic label of
// the long golden documents (main, side and mention topics of all of them).
type longCatalog struct{ tags []CatalogTag }

func newLongCatalog(docs []eval.LongDocument) *longCatalog {
	c := &longCatalog{}
	seen := map[string]bool{}
	for _, d := range docs {
		for _, tp := range d.Topics {
			if k := strings.ToLower(tp.Label); !seen[k] {
				seen[k] = true
				c.tags = append(c.tags, CatalogTag{ID: strconv.Itoa(len(c.tags) + 1), Name: tp.Label, Revision: "1"})
			}
		}
	}
	return c
}

func (c *longCatalog) Tags(context.Context, string, string) ([]CatalogTag, error) { return c.tags, nil }

func (c *longCatalog) Members(context.Context, string, string, []string) (map[string][]string, error) {
	return nil, nil
}

// TestTaggerLongTags runs a Tagger with ChunkBytes 2000 on the 24 long
// documents of zarf/eval/testdata/long (title and text, 7 to 31 KB), against a
// catalog of one name-only tag per topic label of all of them: each document
// should get its main and side tags and not its passing mentions, nor the
// tags of the other documents. Its suggestions are scored with
// eval.MatchTopics: a tag counts for a topic when the cosine of their
// RoleDocument embeddings reaches 0.75 (or one contains the other). Gated like
// TestGoldenTagging:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=$HOME/.cache/semsearch-yzma go test ./tagging/ -run TestTaggerLongTags -v
//
// Measured on 2026-11-07 with EmbeddingGemma 300M Q8_0 through llama.cpp
// b11429, Vulkan on the iGPU of an AMD Ryzen AI 9 HX PRO 370, in 17 s: main
// recall 0.917, side recall 0.678, mention rate 0.021 (one of 54), precision
// 0.783, F1 0.755, 8 tags per document; the mean of the pieces, the rule
// before, had side recall 0.583 at precision 0.780, and no ChunkBytes 0.451.
// Main recall is short of 1 because of eval.MatchTopics: a document's main
// label is picked for every one of them. The floors sit 0.05 under this run.
func TestTaggerLongTags(t *testing.T) {
	if os.Getenv("SEMSEARCH_MODELS") == "" {
		t.Skip("set SEMSEARCH_MODELS=1 to run the long tagging test with the real model")
	}
	dir := os.Getenv("SEMSEARCH_YZMA_TEST_DIR")
	if dir == "" {
		t.Skip("SEMSEARCH_YZMA_TEST_DIR not set; skipping real-model test")
	}
	const tau = 0.75
	ctx := context.Background()
	emb, err := yzma.New(ctx, yzma.Config{
		ModelsDir:     filepath.Join(dir, "models"),
		LibDir:        filepath.Join(dir, "llama-cpp", yzma.LibTag),
		ManagedLibDir: true,
	})
	if err != nil {
		t.Fatalf("yzma.New: %v", err)
	}
	docs, err := eval.LoadLong(filepath.Join("..", "zarf", "eval", "testdata", "long"))
	if err != nil {
		t.Fatal(err)
	}
	cat := newLongCatalog(docs)
	names := map[string]string{}
	for _, tg := range cat.tags {
		names[tg.ID] = tg.Name
	}
	tg, err := NewTagger(emb, cat, NewMemCache(), TaggerConfig{Model: "long", ChunkBytes: 2000})
	if err != nil {
		t.Fatal(err)
	}
	space := tg.Scope("long", "1")

	vecs := map[string][]float32{} // RoleDocument vectors of tags and aliases
	embed := func(texts []string) {
		var need []string
		for _, s := range texts {
			if _, ok := vecs[s]; !ok {
				vecs[s] = nil
				need = append(need, s)
			}
		}
		if len(need) == 0 {
			return
		}
		out, err := embedTexts(ctx, emb, langai.RoleDocument, need)
		if err != nil {
			t.Fatalf("embed names: %v", err)
		}
		for i, s := range need {
			vecs[s] = out[i]
		}
	}
	sim := func(tag, alias string) float64 { return cosine(vecs[tag], vecs[alias]) }

	var scores []eval.LongScore
	for _, d := range docs {
		ids, err := space.Suggest(ctx, "1", d.Title+"\n"+d.Text)
		if err != nil {
			t.Fatalf("Suggest(%s): %v", d.ID, err)
		}
		got := make([]string, len(ids))
		for i, id := range ids {
			got[i] = names[id]
		}
		embed(got)
		for _, tp := range d.Topics {
			embed(tp.Aliases)
		}
		s := eval.MatchTopics(got, d.Topics, sim, tau)
		scores = append(scores, s)
		t.Logf("%s (%d bytes): main %.2f side %.2f mention %.2f prec %.2f: %s",
			d.ID, len(d.Text), s.MainRecall, s.SideRecall, s.MentionRate, s.Precision, strings.Join(got, ", "))
	}
	m := eval.MeanLongScores(scores)
	t.Logf("mean over %d documents: recall %.3f main %.3f side %.3f late side %.3f mention %.3f precision %.3f F1 %.3f tags %d",
		len(docs), m.Recall, m.MainRecall, m.SideRecall, m.LateSideRecall, m.MentionRate, m.Precision, m.F1(), m.Tags)

	const (
		minMainRecall  = 0.87
		minSideRecall  = 0.63
		minPrecision   = 0.73
		maxMentionRate = 0.05
	)
	if m.MainRecall < minMainRecall {
		t.Errorf("main recall %.3f, want at least %.2f", m.MainRecall, minMainRecall)
	}
	if m.SideRecall < minSideRecall {
		t.Errorf("side recall %.3f, want at least %.2f", m.SideRecall, minSideRecall)
	}
	if m.Precision < minPrecision {
		t.Errorf("precision %.3f, want at least %.2f", m.Precision, minPrecision)
	}
	if m.MentionRate > maxMentionRate {
		t.Errorf("mention rate %.3f, want at most %.2f", m.MentionRate, maxMentionRate)
	}
}

package yzma

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/zarf/eval"
)

// TestGenerateLongTags runs the default tagging.Generator on the 24 long
// documents of zarf/eval/testdata/long (title and text, 7 to 31 KB) and scores
// the tags it keeps at tagging.DefaultGeneratorMinScore against the labelled
// topics with eval.MatchTopics: a tag counts for a topic when the cosine of
// their RoleDocument embeddings reaches 0.75 (or one contains the other). It
// downloads the models like TestGenerateTagsWithTheDefaultModels, so it only
// runs when SEMSEARCH_MODELS is set, and takes several minutes:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=~/.cache/semsearch-yzma go test ./provider/yzma/ -run TestGenerateLongTags -v -count=1 -timeout 0
//
// Measured on 2026-11-07 with Qwen3 4B Instruct Q4_K_M and EmbeddingGemma
// Q8_0 through llama.cpp b11429, Vulkan on the iGPU of an AMD Ryzen AI 9 HX PRO
// 370: 344 s for the 24 documents, 6 to 28 s each (14.1 s on average), and the
// means at the default cutoff: main recall 1.000, side recall 0.708, late side
// recall 0.708, mention rate 0.000, precision 0.882, F1 0.820, 10 tags per
// document. Without the title at the head of each window side recall was 0.656.
// The floors were set 0.05 under the experiment's numbers and sit 0.02 to 0.05
// under this run's, so a different llama.cpp build or backend can tip side
// recall or precision.
//
//nolint:gocyclo // one linear gated test: generate, score, assert floors
func TestGenerateLongTags(t *testing.T) {
	if os.Getenv("SEMSEARCH_MODELS") == "" {
		t.Skip("set SEMSEARCH_MODELS=1 to download and test the real models")
	}
	dir := os.Getenv("SEMSEARCH_YZMA_TEST_DIR")
	if dir == "" {
		t.Skip("SEMSEARCH_YZMA_TEST_DIR not set; skipping real-model test")
	}
	const tau = 0.75
	ctx := context.Background()
	docs, err := eval.LoadLong(filepath.Join("..", "..", "zarf", "eval", "testdata", "long"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		ModelsDir:     filepath.Join(dir, "models"),
		LibDir:        filepath.Join(dir, "llama-cpp", LibTag),
		ManagedLibDir: true,
		Logger:        slog.New(slog.NewTextHandler(os.Stderr, nil)),
	}
	emb, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	start := time.Now()
	llm, err := NewCompleter(ctx, cfg)
	if err != nil {
		t.Fatalf("NewCompleter: %v", err)
	}
	t.Cleanup(func() { _ = llm.Close() })
	t.Logf("chat model ready in %s", time.Since(start))
	g, err := tagging.NewGenerator(llm, emb, tagging.GeneratorConfig{})
	if err != nil {
		t.Fatal(err)
	}

	vecs := map[string][]float32{} // RoleDocument vectors of tags and aliases
	embed := func(names []string) {
		var need []string
		seen := map[string]bool{}
		for _, s := range names {
			if _, ok := vecs[s]; !ok && !seen[s] {
				seen[s] = true
				need = append(need, s)
			}
		}
		if len(need) == 0 {
			return
		}
		out, err := emb.Embed(ctx, langai.RoleDocument, need)
		if err != nil || len(out) != len(need) {
			t.Fatalf("embed names: %v", err)
		}
		for i, s := range need {
			vecs[s] = out[i]
		}
	}
	sim := func(tag, alias string) float64 { return cosineF32(vecs[tag], vecs[alias]) }

	var scores []eval.LongScore
	var total time.Duration
	for _, d := range docs {
		start := time.Now()
		tags, err := g.Generate(ctx, d.Title+"\n"+d.Text, tagging.DefaultGeneratorMinScore)
		if err != nil {
			t.Fatalf("Generate(%s): %v", d.ID, err)
		}
		took := time.Since(start)
		total += took
		names := make([]string, len(tags))
		var line strings.Builder
		for i, tg := range tags {
			names[i] = tg.Name
			fmt.Fprintf(&line, " %s=%.2f", strings.ReplaceAll(tg.Name, " ", "_"), tg.Score)
		}
		embed(names)
		for _, tp := range d.Topics {
			embed(tp.Aliases)
		}
		s := eval.MatchTopics(names, d.Topics, sim, tau)
		scores = append(scores, s)
		t.Logf("%s (%d bytes) in %.1fs: main %.2f side %.2f late %.2f mention %.2f prec %.2f:%s",
			d.ID, len(d.Text), took.Seconds(), s.MainRecall, s.SideRecall, s.LateSideRecall, s.MentionRate, s.Precision, line.String())
	}
	m := eval.MeanLongScores(scores)
	t.Logf("mean over %d documents, %.1fs each: recall %.3f main %.3f side %.3f late side %.3f mention %.3f precision %.3f F1 %.3f tags %d",
		len(docs), total.Seconds()/float64(len(docs)), m.Recall, m.MainRecall, m.SideRecall, m.LateSideRecall, m.MentionRate, m.Precision, m.F1(), m.Tags)

	const (
		minMainRecall     = 0.95
		minSideRecall     = 0.68
		minLateSideRecall = 0.68
		minPrecision      = 0.86
		maxMentionRate    = 0.05
	)
	if m.MainRecall < minMainRecall {
		t.Errorf("main recall %.3f, want at least %.2f", m.MainRecall, minMainRecall)
	}
	if m.SideRecall < minSideRecall {
		t.Errorf("side recall %.3f, want at least %.2f", m.SideRecall, minSideRecall)
	}
	if m.LateSideRecall < minLateSideRecall {
		t.Errorf("late side recall %.3f, want at least %.2f", m.LateSideRecall, minLateSideRecall)
	}
	if m.Precision < minPrecision {
		t.Errorf("precision %.3f, want at least %.2f", m.Precision, minPrecision)
	}
	if m.MentionRate > maxMentionRate {
		t.Errorf("mention rate %.3f, want at most %.2f", m.MentionRate, maxMentionRate)
	}
}

// cosineF32 is the cosine of two vectors, 0 when either is empty or zero.
func cosineF32(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

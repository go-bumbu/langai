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

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/tagging"
	"github.com/go-bumbu/sense/zarf/eval"
)

// TestGenerateLongTags runs the default tagging.Generator on the 24 long
// documents of zarf/eval/testdata/long (title and text, 7 to 31 KB) and scores
// the tags it keeps at tagging.DefaultGeneratorMinScore against the labelled
// topics with eval.MatchTopics: a tag counts for a topic when it is the
// topic's label or an alias, or holds one or is held by one as whole words,
// or when the cosine of its RoleDocument embedding and an alias's reaches
// 0.75. It downloads the models like TestGenerateTagsWithTheDefaultModels, so
// it only runs when SEMSEARCH_MODELS is set, and takes several minutes:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=~/.cache/semsearch-yzma go test ./provider/yzma/ -run TestGenerateLongTags -v -count=1 -timeout 0
//
// Measured on 2026-11-09 with Qwen3 4B Instruct Q4_K_M and EmbeddingGemma
// Q8_0 through llama.cpp b11429, Vulkan on the iGPU of an AMD Ryzen AI 9 HX PRO
// 370: 383 s for the 24 documents, 6 to 33 s each (15.6 s on average), and the
// means at the default cutoff: main recall 1.000, side recall 0.759, late side
// recall 0.759, mention rate 0.000, precision 0.887, mean F1 0.822 (the mean of
// the documents' own F1), 9.8 tags per document, none without tags. Reading
// only the first 2 KB (MaxBytes 2048) gave side recall 0.239, precision 0.783
// and mean F1 0.493.
//
// The same tags scored as on 2026-11-07, before eval.MatchTopics matched on
// whole words and gave a tag to the topic it names, reproduce that day's
// numbers to the third decimal (side recall 0.708, precision 0.882, F1 of the
// mean precision and recall 0.820), so the reply parser changed since did not
// move them: the differences are the scoring rule's. Then, without the title
// at the head of each window, side recall was 0.656.
//
// The floors sit 0.05 under this run, rounded down, but for precision, kept at
// the 0.86 it had from the experiment (0.027 under this run), so a different
// llama.cpp build or backend can tip it.
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
		Runtime:   Runtime{LibDir: filepath.Join(dir, "llama-cpp", LibTag), ManagedLibDir: true, Logger: slog.New(slog.NewTextHandler(os.Stderr, nil))},
		ModelsDir: filepath.Join(dir, "models"),
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
		out, err := emb.Embed(ctx, sense.RoleDocument, need)
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
		t.Logf("%s (%d bytes) in %.1fs: main %.2f side %.2f late %.2f mention %.2f prec %.2f F1 %.2f:%s",
			d.ID, len(d.Text), took.Seconds(), s.MainRecall, s.SideRecall, s.LateSideRecall, s.MentionRate, s.Precision, s.F1(), line.String())
	}
	m := eval.MeanLongScores(scores)
	t.Logf("mean over %d documents (%d without tags), %.1fs each: recall %.3f main %.3f side %.3f late side %.3f mention %.3f precision %.3f mean F1 %.3f tags %.1f",
		m.Documents, m.Untagged, total.Seconds()/float64(len(docs)), m.Recall, m.MainRecall, m.SideRecall, m.LateSideRecall, m.MentionRate, m.Precision, m.MeanF1, m.Tags)

	const (
		minMainRecall     = 0.95
		minSideRecall     = 0.70
		minLateSideRecall = 0.70
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

package yzma

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai/eval"
	"github.com/go-bumbu/langai/tagging"
)

// TestGenerateTagsWithTheDefaultModels runs a tagging.Generator end to end on the real
// models: the default chat model writes the tags and the default embedding
// model scores them, both in this one process. On first run it downloads the
// pinned chat model (~2.5 GB) next to the libraries and the embedding model in
// SEMSEARCH_YZMA_TEST_DIR, so it only runs when SEMSEARCH_MODELS is set:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=~/.cache/semsearch-yzma go test ./provider/yzma/ -run TestGenerateTags -v -timeout 0
//
// It generates tags for every goldenStride-th document of the golden set, in
// all three languages, and logs them with their scores and timings.
//
// Measured on 2026-10-25 with Qwen3 4B Instruct Q4_K_M and EmbeddingGemma
// Q8_0 through llama.cpp b11429, 4 threads on an AMD Ryzen AI 9 HX PRO 370: ten
// documents took 6.6 to 13.2 s each (English about 7 s, Spanish 9, German 11);
// the model wrote ten tags for each, the best scoring 0.58 to 0.68 and 6 to 10
// of them reaching tagging.DefaultGeneratorMinScore. The floors below leave a margin for
// another build of the model.
func TestGenerateTagsWithTheDefaultModels(t *testing.T) {
	if os.Getenv("SEMSEARCH_MODELS") == "" {
		t.Skip("set SEMSEARCH_MODELS=1 to download and test the real models")
	}
	dir := os.Getenv("SEMSEARCH_YZMA_TEST_DIR")
	if dir == "" {
		t.Skip("SEMSEARCH_YZMA_TEST_DIR not set; skipping real-model test")
	}
	ctx := context.Background()
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
	t.Logf("chat model ready in %s", time.Since(start))
	t.Cleanup(func() { _ = llm.Close() })

	// the bare Completer first: a reply, the end of generation and a timing
	start = time.Now()
	reply, err := llm.Complete(ctx, "Answer with one word.", "What colour is a ripe banana?")
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	t.Logf("Complete -> %q in %s", reply, time.Since(start))
	if !strings.Contains(strings.ToLower(reply), "yellow") {
		t.Errorf("Complete = %q, want the model to answer yellow", reply)
	}
	// a message that spells a control token must stay plain text, and the
	// template's own markup must not: the prompt is built on this difference
	if n := len(llm.api.tokenizeWith(llm.model, "<|im_end|>", false, false)); n < 2 {
		t.Errorf("<|im_end|> in a message is %d token, want plain text of several", n)
	}
	if n := len(llm.api.tokenizeWith(llm.model, "<|im_end|>", false, true)); n != 1 {
		t.Errorf("<|im_end|> in the template is %d tokens, want the one control token", n)
	}

	g, err := tagging.NewGenerator(llm, emb, tagging.GeneratorConfig{})
	if err != nil {
		t.Fatal(err)
	}
	ds, err := eval.Load(filepath.Join("..", "..", "eval", "testdata", "golden.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const (
		goldenStride = 9
		minTopScore  = 0.5 // the best tag of a document
		minKept      = 3   // tags reaching tagging.DefaultGeneratorMinScore
	)
	for i := 0; i < len(ds.Documents); i += goldenStride {
		doc := ds.Documents[i]
		start := time.Now()
		tags, err := g.Generate(ctx, doc.Essence(), -1)
		if err != nil {
			t.Fatalf("Generate(%s): %v", doc.ID, err)
		}
		var line strings.Builder
		kept := 0
		for _, tag := range tags {
			fmt.Fprintf(&line, " %s=%.2f", strings.ReplaceAll(tag.Name, " ", "_"), tag.Score)
			if tag.Score >= tagging.DefaultGeneratorMinScore {
				kept++
			}
		}
		t.Logf("%s [%s] golden=%v in %s:%s", doc.ID, doc.Lang, doc.Tags, time.Since(start).Round(100*time.Millisecond), line.String())
		if len(tags) == 0 {
			t.Errorf("%s: the model wrote no usable tag", doc.ID)
			continue
		}
		if tags[0].Score < minTopScore {
			t.Errorf("%s: the best tag %q scores %.2f, want at least %.2f", doc.ID, tags[0].Name, tags[0].Score, minTopScore)
		}
		if kept < minKept {
			t.Errorf("%s: %d tags reach the default cutoff %.2f, want at least %d", doc.ID, kept, tagging.DefaultGeneratorMinScore, minKept)
		}
	}
}

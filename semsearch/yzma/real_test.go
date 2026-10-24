package yzma

import (
	"context"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/semsearch"
)

// TestEmbedAndRank exercises the real model through llama.cpp. On first run
// it downloads the pinned libraries (~18 MB) and model (~329 MB) into
// SEMSEARCH_YZMA_TEST_DIR, so it only runs when SEMSEARCH_MODELS is set:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=/tmp/semsearch-yzma go test ./semsearch/yzma/
//
// The host needs libffi8, libgomp1 and libstdc++6 (see README).
func TestEmbedAndRank(t *testing.T) {
	if os.Getenv("SEMSEARCH_MODELS") == "" {
		t.Skip("set SEMSEARCH_MODELS=1 to download and test the real model")
	}
	dir := os.Getenv("SEMSEARCH_YZMA_TEST_DIR")
	if dir == "" {
		t.Skip("SEMSEARCH_YZMA_TEST_DIR not set; skipping real-model test")
	}
	ctx := context.Background()
	emb, err := New(ctx, Config{
		ModelsDir:     filepath.Join(dir, "models"),
		LibDir:        filepath.Join(dir, "llama-cpp", LibTag),
		ManagedLibDir: true,
		Logger:        slog.New(slog.NewTextHandler(os.Stderr, nil)),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if emb.Dimensions() != 768 {
		t.Errorf("expected 768 dimensions for the default model, got %d", emb.Dimensions())
	}

	vecs, err := emb.Embed(ctx, semsearch.RoleDocument, []string{
		"How do I cook pasta?",
		"Boil water and add the noodles.",
		"The stock market fell sharply today.",
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 3 {
		t.Fatalf("expected 3 vectors, got %d", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != emb.Dimensions() {
			t.Fatalf("vector %d has %d dims", i, len(v))
		}
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		if math.Abs(sum-1) > 1e-3 {
			t.Errorf("vector %d is not unit length: %f", i, sum)
		}
	}

	// the semantically related pair must be closer than the unrelated one —
	// this is the actual point of a real embedding model
	related := cosine(vecs[0], vecs[1])
	unrelated := cosine(vecs[0], vecs[2])
	if related <= unrelated {
		t.Errorf("expected related texts to score higher: related=%f unrelated=%f", related, unrelated)
	}

	// a text far beyond the 512-token cap must embed (truncated), not fail
	// or abort; it still lands near its topic
	long := strings.Repeat("Cooking pasta at home is simple and rewarding when you salt the water generously. ", 80)
	lv, err := emb.Embed(ctx, semsearch.RoleDocument, []string{long})
	if err != nil {
		t.Fatalf("Embed(long): %v", err)
	}
	if c := cosine(lv[0], vecs[0]); c < 0.3 {
		t.Errorf("truncated long text scored %f against the pasta question, want > 0.3", c)
	}

	// end to end through the Index with the in-memory store
	ix, err := semsearch.New(emb, semsearch.NewMemStore(), "test", "test", "test")
	if err != nil {
		t.Fatalf("semsearch.New: %v", err)
	}
	if err := ix.Index(ctx, "pasta", "# Cooking pasta\n\nBoil salted water, add the noodles, wait ten minutes."); err != nil {
		t.Fatalf("Index: %v", err)
	}
	if err := ix.Index(ctx, "stocks", "# Markets\n\nShares dropped as investors reacted to the earnings report."); err != nil {
		t.Fatalf("Index: %v", err)
	}
	got, err := ix.Search(ctx, "how long to boil spaghetti", 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) == 0 || got[0].DocID != "pasta" {
		t.Fatalf("expected the pasta doc first, got %#v", got)
	}
}

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot // vectors are unit length
}

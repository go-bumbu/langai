package yzma

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai/semsearch/eval"
)

// floor is one asserted minimum for the default model: a task ("search" for
// same-language retrieval, "xling" for cross-lingual retrieval, "tagging"), a
// metric (see lookupMetric for the names) and the lowest value allowed. Each
// minimum is the value measured on the date in the comment minus a margin:
// 0.05 for search (~60 queries: one miss moves recall@5 by ~0.02, a rank
// flip moves MRR by less), 0.15 for xling (~14 queries: one miss moves
// recall@10 by up to ~0.07, so the floor allows two), 0.05 for tagging (~84
// documents). Re-baseline the table in the same commit as a change to the
// model, the pins or the golden set: run the test with -v, take each
// metric's four-decimal line, subtract the margin and round down to two
// decimals (never below 0).
type floor struct {
	task, metric string
	min          float64
}

// Measured on 2026-10-21 with embeddinggemma-300m Q8_0 through llama.cpp
// b11429. f1@floor is tagging at autotag's default NameMinScore: the model's
// scores sit in a narrow band, so a default that drifts from its best cutoff
// shows here even while f1@best holds.
var floors = []floor{
	{"search", "recall@5", 0.95},  // measured 1.0000 on 2026-10-21
	{"search", "mrr", 0.95},       // measured 1.0000 on 2026-10-21
	{"xling", "recall@10", 0.85},  // measured 1.0000 on 2026-10-21
	{"tagging", "f1@best", 0.65},  // measured 0.7071 on 2026-10-21
	{"tagging", "f1@floor", 0.65}, // measured 0.7071 on 2026-10-21
	{"tagging", "top1", 0.84},     // measured 0.8929 on 2026-10-21
}

// lookupMetric reads one metric of one task out of a result. ok is false for
// an unknown task or metric and for a retrieval set the dataset left empty
// (an edit dropped every cross-lingual query, say), so the caller fails
// loudly instead of comparing against a zero.
func lookupMetric(res *eval.Result, task, metric string) (float64, bool) {
	var r eval.Retrieval
	switch task {
	case "search":
		r = res.SameLanguage
	case "xling":
		r = res.CrossLingual
	case "tagging":
		switch metric {
		case "f1@floor":
			return res.Tagging.AtFloor.F1, true
		case "f1@best":
			return res.Tagging.Best.F1, true
		case "top1":
			return res.Tagging.Top1, true
		}
		return 0, false
	default:
		return 0, false
	}
	if r.Queries == 0 {
		return 0, false
	}
	if k, found := strings.CutPrefix(metric, "recall@"); found {
		n, err := strconv.Atoi(k)
		if err != nil {
			return 0, false
		}
		v, has := r.RecallAt[n]
		return v, has
	}
	switch metric {
	case "mrr":
		return r.MRR, true
	case "ndcg@10":
		return r.NDCG10, true
	}
	return 0, false
}

func TestLookupMetric_UnknownOrEmptyIsReported(t *testing.T) {
	res := &eval.Result{
		SameLanguage: eval.Retrieval{Queries: 3, RecallAt: map[int]float64{5: 0.8}, MRR: 0.7, NDCG10: 0.6},
		CrossLingual: eval.Retrieval{RecallAt: map[int]float64{5: 0.4}, MRR: 0.3}, // no queries: the dataset has none
		Tagging:      eval.Tagging{AtFloor: eval.Cut{F1: 0.5}, Best: eval.Cut{F1: 0.55}, Top1: 0.9},
	}
	cases := []struct {
		task, metric string
		want         float64
		ok           bool
	}{
		{"search", "recall@5", 0.8, true}, {"search", "mrr", 0.7, true}, {"search", "ndcg@10", 0.6, true},
		{"tagging", "f1@floor", 0.5, true}, {"tagging", "f1@best", 0.55, true}, {"tagging", "top1", 0.9, true},
		{"xling", "recall@5", 0, false}, {"xling", "mrr", 0, false}, // the set is empty, so its zeros are not scores
		{"search", "recall@7", 0, false}, {"search", "bogus", 0, false}, {"tagging", "mrr", 0, false}, {"de", "recall@5", 0, false},
	}
	for _, c := range cases {
		got, ok := lookupMetric(res, c.task, c.metric)
		if got != c.want || ok != c.ok {
			t.Errorf("lookupMetric(%s, %s) = %f, %v; want %f, %v", c.task, c.metric, got, ok, c.want, c.ok)
		}
	}
}

// TestGoldenSetWithDefaultModel runs the golden set through the default model
// and asserts the floors above. Like TestEmbedAndRank it downloads the
// pinned libraries and model into SEMSEARCH_YZMA_TEST_DIR on first run:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=~/.cache/semsearch-yzma go test ./semsearch/yzma/ -run TestGoldenSet -v
func TestGoldenSetWithDefaultModel(t *testing.T) {
	if os.Getenv("SEMSEARCH_MODELS") == "" {
		t.Skip("set SEMSEARCH_MODELS=1 to download and test the real model")
	}
	dir := os.Getenv("SEMSEARCH_YZMA_TEST_DIR")
	if dir == "" {
		t.Skip("SEMSEARCH_YZMA_TEST_DIR not set; skipping real-model test")
	}
	ctx := context.Background()
	start := time.Now()
	emb, err := New(ctx, Config{
		ModelsDir:     filepath.Join(dir, "models"),
		LibDir:        filepath.Join(dir, "llama-cpp", LibTag),
		ManagedLibDir: true,
		Logger:        slog.New(slog.NewTextHandler(os.Stderr, nil)),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	loadTime := time.Since(start)
	ds, err := eval.Load(filepath.Join("..", "eval", "testdata", "golden.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	res, err := eval.Run(ctx, emb, ds, eval.Options{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	t.Logf("default model, load %s\n%s", loadTime, res)
	if res.Dims != 768 {
		t.Errorf("Dims = %d, want 768", res.Dims)
	}
	for _, f := range floors {
		got, ok := lookupMetric(res, f.task, f.metric)
		if !ok {
			t.Errorf("%s %s: task or metric missing from the result", f.task, f.metric)
			continue
		}
		t.Logf("%s %s = %.4f (floor %.2f)", f.task, f.metric, got, f.min)
		if got < f.min {
			t.Errorf("%s %s = %.4f, below the floor %.4f", f.task, f.metric, got, f.min)
		}
	}
}

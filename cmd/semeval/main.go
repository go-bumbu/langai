// Command semeval compares GGUF embedding models on the semsearch golden set
// (or any dataset in its format): for each entry of the models file it opens
// the model through semsearch/yzma, runs semsearch/eval and prints one row
// per model (same-language search, cross-lingual search, tagging and its
// floor, vector size, speed) with a legend; -json keeps every metric. Models
// and the llama.cpp libraries are cached in -dir, the directory the
// real-model tests use, so the default model is downloaded once for both.
//
//	make semeval                                # every model in cmd/semeval/models.yaml
//	go run ./cmd/semeval -only org/model        # one of them
//	go run ./cmd/semeval -dataset export.yaml   # a private export in the eval format
//
// An entry with a url but no sha256 is downloaded unverified once; the
// digest is printed so it can be pasted into the file. The host needs
// libffi8, libgomp1 and libstdc++6 (see README).
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/go-bumbu/langai/semsearch/eval"
	"github.com/go-bumbu/langai/semsearch/yzma"
)

func main() {
	models := flag.String("models", "cmd/semeval/models.yaml", "models file (see models.yaml for the format)")
	dataset := flag.String("dataset", "semsearch/eval/testdata/golden.yaml", "dataset in the eval format")
	dir := flag.String("dir", defaultDir(), "cache directory for models and the llama.cpp libraries")
	only := flag.String("only", "", "comma-separated model names to run; empty runs every entry")
	threads := flag.Int("threads", 0, "threads per model; 0 picks the yzma default (at most four)")
	floor := flag.Float64("floor", 0.46, "tagging cutoff to report at (autotag's NameMinScore; 0 means 0.46)")
	jsonOut := flag.String("json", "", "also write the full results to this file as JSON")
	flag.Parse()
	os.Exit(run(*models, *dataset, *dir, *only, *threads, *floor, *jsonOut, os.Stdout, os.Stderr))
}

// defaultDir is the cache the real-model tests document: ~/.cache/semsearch-yzma.
func defaultDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "semsearch-yzma")
	}
	return filepath.Join(home, ".cache", "semsearch-yzma")
}

func run(modelsPath, datasetPath, dir, only string, threads int, floor float64, jsonOut string, stdout, stderr io.Writer) int {
	logger := slog.New(slog.NewTextHandler(stderr, nil))
	entries, err := loadModels(modelsPath)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, "semeval:", err)
		return 2
	}
	entries, err = selectEntries(entries, only)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, "semeval:", err)
		return 2
	}
	ds, err := eval.Load(datasetPath)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, "semeval:", err)
		return 2
	}
	rt := yzma.Runtime{LibDir: filepath.Join(dir, "llama-cpp", yzma.LibTag), ManagedLibDir: true, Threads: threads, Logger: logger}
	modelsDir := filepath.Join(dir, "models")
	opts := eval.Options{Floor: floor}
	rows := make([]row, 0, len(entries))
	failed := 0
	for _, e := range entries {
		_, _ = fmt.Fprintf(stderr, "== %s\n", e.Name)
		r := evaluate(context.Background(), e, rt, modelsDir, ds, opts, logger)
		if r.Err != "" {
			failed++
			_, _ = fmt.Fprintf(stderr, "   failed: %s\n", r.Err)
		}
		rows = append(rows, r)
	}
	writeTable(stdout, ds, rows)
	if jsonOut != "" {
		f, err := os.Create(jsonOut) //nolint:gosec // jsonOut is an operator's flag
		if err != nil {
			_, _ = fmt.Fprintln(stderr, "semeval:", err)
			return 2
		}
		defer func() { _ = f.Close() }()
		if err := writeJSON(f, rows); err != nil {
			_, _ = fmt.Fprintln(stderr, "semeval:", err)
			return 2
		}
	}
	if failed > 0 {
		return 1
	}
	return 0
}

// evaluate opens one model (downloading an unpinned one first and reporting
// its digest), runs the dataset through it and closes it again. Any failure
// becomes the row's error so the other models still run.
func evaluate(ctx context.Context, e entry, rt yzma.Runtime, modelsDir string, ds *eval.Dataset, opts eval.Options, logger *slog.Logger) row {
	r := row{Name: e.Name}
	spec := e.spec(modelsDir)
	if spec.URL != "" && spec.SHA256 == "" {
		if _, err := os.Stat(spec.File); err != nil {
			sum, size, err := yzma.Download(ctx, spec.URL, spec.File, logger)
			if err != nil {
				r.Err = err.Error()
				return r
			}
			logger.Warn("unpinned model downloaded; add these to its entry", slog.String("model", e.Name), slog.String("sha256", sum), slog.Int64("size", size))
		}
		spec.URL = "" // open the local file; Open refuses a URL without a digest
	}
	start := time.Now()
	emb, err := yzma.Open(ctx, spec, rt)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	r.Load = time.Since(start)
	defer func() { _ = emb.Close() }()
	res, err := eval.Run(ctx, emb, ds, opts)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	r.Result = res
	return r
}

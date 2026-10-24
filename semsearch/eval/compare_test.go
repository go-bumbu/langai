package eval_test

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai/semsearch/eval"
	"github.com/go-bumbu/langai/semsearch/yzma"
)

// TestCompareModels compares GGUF embedding models on the golden set (or any
// dataset in its format): for each entry of the models file it opens the
// model through semsearch/yzma, runs eval.Run and prints one row per model
// (same-language search, cross-lingual search, tagging and its floor, vector
// size, speed) with a legend; SEMEVAL_JSON keeps every metric. It is a tool
// rather than a check, so it runs only with SEMEVAL=1, and make semeval is
// its entrypoint:
//
//	make semeval                               # every model in testdata/models.yaml
//	make semeval SEMEVAL_ONLY=org/model        # one of them; comma-separate several
//	make semeval SEMEVAL_DATASET=export.yaml   # a private export in the eval format
//
// compareSettings lists every variable. Models and the llama.cpp libraries
// are cached in SEMSEARCH_YZMA_TEST_DIR, the directory the real-model tests
// use, so the default model is downloaded once for both. An entry with a url
// but no sha256 is downloaded unverified once; the digest is logged so it can
// be pasted into the file. A model that fails to load or run fails the test
// once the others have run. The host needs libffi8, libgomp1 and libstdc++6
// (see README).
func TestCompareModels(t *testing.T) {
	if os.Getenv("SEMEVAL") == "" {
		t.Skip("set SEMEVAL=1 (make semeval) to download and compare the models")
	}
	s, err := compareSettingsFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	entries, err := loadModels(s.models)
	if err != nil {
		t.Fatal(err)
	}
	entries, err = selectEntries(entries, s.only)
	if err != nil {
		t.Fatal(err)
	}
	ds, err := eval.Load(s.dataset)
	if err != nil {
		t.Fatal(err)
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	rt := yzma.Runtime{LibDir: filepath.Join(s.dir, "llama-cpp", yzma.LibTag), ManagedLibDir: true, Threads: s.threads, Logger: logger}
	modelsDir := filepath.Join(s.dir, "models")
	opts := eval.Options{Floor: s.floor}
	rows := make([]row, 0, len(entries))
	for _, e := range entries {
		t.Logf("== %s", e.Name)
		r := evaluate(context.Background(), e, rt, modelsDir, ds, opts, logger)
		if r.Err != "" {
			t.Errorf("%s failed: %s", e.Name, r.Err)
		}
		rows = append(rows, r)
	}
	writeTable(t.Output(), ds, rows)
	if s.jsonOut == "" {
		return
	}
	f, err := os.Create(s.jsonOut)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	if err := writeJSON(f, rows); err != nil {
		t.Fatal(err)
	}
}

// compareSettings are TestCompareModels' inputs. make semeval passes them as
// environment variables and turns its paths absolute first: go test runs in
// this package's directory, so a relative path is relative to it. An empty
// variable keeps the default.
type compareSettings struct {
	models  string  // SEMEVAL_MODELS: the models file; testdata/models.yaml
	dataset string  // SEMEVAL_DATASET: the dataset in the eval format; testdata/golden.yaml
	only    string  // SEMEVAL_ONLY: comma-separated model names to run; empty runs every entry
	jsonOut string  // SEMEVAL_JSON: also write the full results to this file as JSON
	dir     string  // SEMSEARCH_YZMA_TEST_DIR: the cache for models and the llama.cpp libraries; required
	threads int     // SEMEVAL_THREADS: threads per model; 0 picks the yzma default (at most four)
	floor   float64 // SEMEVAL_FLOOR: the tagging cutoff to report at (autotag's NameMinScore); 0 means 0.46
}

// compareSettingsFromEnv reads compareSettings. A number that does not parse
// and a missing cache directory are errors.
func compareSettingsFromEnv() (compareSettings, error) {
	s := compareSettings{
		models:  envOr("SEMEVAL_MODELS", "testdata/models.yaml"),
		dataset: envOr("SEMEVAL_DATASET", "testdata/golden.yaml"),
		only:    os.Getenv("SEMEVAL_ONLY"),
		jsonOut: os.Getenv("SEMEVAL_JSON"),
		dir:     os.Getenv("SEMSEARCH_YZMA_TEST_DIR"),
	}
	if s.dir == "" {
		return s, errors.New("SEMSEARCH_YZMA_TEST_DIR is not set; make semeval sets it")
	}
	var err error
	if v := os.Getenv("SEMEVAL_THREADS"); v != "" {
		if s.threads, err = strconv.Atoi(v); err != nil {
			return s, fmt.Errorf("SEMEVAL_THREADS: %w", err)
		}
	}
	if v := os.Getenv("SEMEVAL_FLOOR"); v != "" {
		if s.floor, err = strconv.ParseFloat(v, 64); err != nil {
			return s, fmt.Errorf("SEMEVAL_FLOOR: %w", err)
		}
	}
	return s, nil
}

// envOr is the variable's value, or def when it is unset or empty.
func envOr(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
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

// clearCompareEnv empties every variable compareSettingsFromEnv reads, so a
// value exported in the developer's shell cannot leak into a test.
func clearCompareEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{"SEMEVAL_MODELS", "SEMEVAL_DATASET", "SEMEVAL_ONLY", "SEMEVAL_JSON", "SEMEVAL_THREADS", "SEMEVAL_FLOOR", "SEMSEARCH_YZMA_TEST_DIR"} {
		t.Setenv(name, "")
	}
}

func TestCompareSettingsFromEnv(t *testing.T) {
	clearCompareEnv(t)
	t.Setenv("SEMSEARCH_YZMA_TEST_DIR", "/cache")
	s, err := compareSettingsFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if want := (compareSettings{models: "testdata/models.yaml", dataset: "testdata/golden.yaml", dir: "/cache"}); s != want {
		t.Fatalf("defaults = %+v, want %+v", s, want)
	}
	t.Setenv("SEMEVAL_MODELS", "/m.yaml")
	t.Setenv("SEMEVAL_DATASET", "/d.yaml")
	t.Setenv("SEMEVAL_ONLY", "org/a, org/b")
	t.Setenv("SEMEVAL_JSON", "/out.json")
	t.Setenv("SEMEVAL_THREADS", "2")
	t.Setenv("SEMEVAL_FLOOR", "0.5")
	if s, err = compareSettingsFromEnv(); err != nil {
		t.Fatal(err)
	}
	want := compareSettings{models: "/m.yaml", dataset: "/d.yaml", only: "org/a, org/b", jsonOut: "/out.json", dir: "/cache", threads: 2, floor: 0.5}
	if s != want {
		t.Fatalf("settings = %+v, want %+v", s, want)
	}
}

func TestCompareSettingsFromEnv_RejectsBadValues(t *testing.T) {
	cases := []struct{ name, key, value string }{
		{"no cache directory", "SEMSEARCH_YZMA_TEST_DIR", ""},
		{"threads not a number", "SEMEVAL_THREADS", "four"},
		{"floor not a number", "SEMEVAL_FLOOR", "high"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			clearCompareEnv(t)
			t.Setenv("SEMSEARCH_YZMA_TEST_DIR", "/cache")
			t.Setenv(c.key, c.value)
			if _, err := compareSettingsFromEnv(); err == nil || !strings.Contains(err.Error(), c.key) {
				t.Fatalf("err = %v, want it to name %s", err, c.key)
			}
		})
	}
}

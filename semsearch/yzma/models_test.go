package yzma

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/semsearch"
)

// offline keeps a test from downloading a real model: requests to anything
// but the loopback test servers fail, so a spec that misses its pre-seeded
// file errors out instead of fetching hundreds of megabytes.
func offline(t *testing.T) {
	t.Helper()
	orig := httpClient
	httpClient = &http.Client{Transport: loopbackOnly{}}
	t.Cleanup(func() { httpClient = orig })
}

// loopbackOnly is a RoundTripper that only reaches httptest servers.
type loopbackOnly struct{}

func (loopbackOnly) RoundTrip(r *http.Request) (*http.Response, error) {
	if h := r.URL.Hostname(); h != "127.0.0.1" && h != "::1" && h != "localhost" {
		return nil, fmt.Errorf("test is offline, refusing %s", r.URL)
	}
	return http.DefaultTransport.RoundTrip(r)
}

// TestNew_RunsTheSelectedModel: Config.Model selects one of the pinned
// models, which New loads from its file in ModelsDir (named after its
// download, as make semeval names it, so the two can share the directory)
// and runs with its authors' prefixes.
func TestNew_RunsTheSelectedModel(t *testing.T) {
	tmp := t.TempDir()
	f := newFakeAPI()
	f.buf = make([]float32, 384) // the selected model's size, which its spec checks
	for i := range f.buf {
		f.buf[i] = 1
	}
	rt := fakeRuntime(t, f, tmp)
	offline(t)
	modelsDir := filepath.Join(tmp, "models")
	modelPath := filepath.Join(modelsDir, "multilingual-e5-small-q8_0.gguf")
	if err := os.MkdirAll(modelsDir, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modelPath, []byte("GGUF fake model"), 0o600); err != nil {
		t.Fatal(err)
	}
	e, err := New(context.Background(), Config{
		Model: "intfloat/multilingual-e5-small", ModelsDir: modelsDir,
		LibDir: rt.LibDir, ManagedLibDir: rt.ManagedLibDir, Threads: rt.Threads,
	})
	if err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "loadModel"); args[0] != modelPath {
		t.Errorf("loadModel(%v), want %s", args[0], modelPath)
	}
	f.calls = nil
	if _, err := e.Embed(context.Background(), semsearch.RoleQuery, []string{"x"}); err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "tokenize"); args[1] != "query: x" {
		t.Errorf("tokenize(%v), want the model's query prefix applied", args[1])
	}
}

// TestSpec_DescribesEachModelAsItsAuthorsTrainedIt: a model run with the
// wrong pooling or prefixes still returns vectors, only meaningless ones, and
// a wrong size mis-sizes the store a caller builds from the spec; nothing
// downstream would notice either. The expected values are the ones
// make semeval evaluated each model with.
func TestSpec_DescribesEachModelAsItsAuthorsTrainedIt(t *testing.T) {
	cases := []struct {
		name            string
		pooling         Pooling
		query, document string
		dims            int
	}{
		{"sentence-transformers/all-MiniLM-L6-v2", PoolingMean, "", "", 384},
		{"intfloat/multilingual-e5-small", PoolingMean, "query: ", "passage: ", 384},
		{"BAAI/bge-m3", PoolingCLS, "", "", 1024},
		{"google/embeddinggemma-300m", PoolingMean, "task: search result | query: ", "title: none | text: ", 768},
		{"nomic-ai/nomic-embed-text-v2-moe", PoolingMean, "search_query: ", "search_document: ", 768},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			spec, err := Spec(c.name, "/data/models")
			if err != nil {
				t.Fatal(err)
			}
			if spec.Name != c.name || spec.Pooling != c.pooling || spec.QueryPrefix != c.query || spec.DocumentPrefix != c.document || spec.Dims != c.dims {
				t.Fatalf("spec = %+v, want %s pooling, prefixes %q / %q and %d dims", spec, c.pooling, c.query, c.document, c.dims)
			}
			if spec.URL == "" || spec.SHA256 == "" || filepath.Dir(spec.File) != "/data/models" {
				t.Fatalf("spec = %+v, want a pinned download into /data/models", spec)
			}
		})
	}
}

// TestSpec_EmptyNameIsTheDefaultModel: Config.Model's zero value runs the
// default model, EmbeddingGemma, from the default directory.
func TestSpec_EmptyNameIsTheDefaultModel(t *testing.T) {
	spec, err := Spec("", "")
	if err != nil {
		t.Fatal(err)
	}
	if spec.Name != "google/embeddinggemma-300m" || spec.File != filepath.Join("models", "embeddinggemma-300M-Q8_0.gguf") || spec.Dims != 768 {
		t.Fatalf("spec = %+v, want google/embeddinggemma-300m (768 dims) from the models directory", spec)
	}
}

// TestSpec_UnknownModelListsTheAvailableOnes: a misspelt model is refused
// with the names that would have worked.
func TestSpec_UnknownModelListsTheAvailableOnes(t *testing.T) {
	_, err := Spec("org/unknown", "")
	if err == nil || !strings.Contains(err.Error(), DefaultModel) || !strings.Contains(err.Error(), "BAAI/bge-m3") {
		t.Fatalf("err = %v, want a refusal listing the available models", err)
	}
}

// TestOpen_TwoModelsSideBySide: a migration runs the current model and the
// next one in the same process, so a second model opens next to the first
// (the libraries load once) and each keeps its own file and prefixes.
func TestOpen_TwoModelsSideBySide(t *testing.T) {
	tmp := t.TempDir()
	f := newFakeAPI()
	rt := fakeRuntime(t, f, tmp)
	currentPath := filepath.Join(tmp, "current.gguf")
	nextPath := filepath.Join(tmp, "next.gguf")
	for _, p := range []string{currentPath, nextPath} {
		if err := os.WriteFile(p, []byte("GGUF fake model"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	current, err := Open(context.Background(), ModelSpec{Name: "org/current", File: currentPath}, rt)
	if err != nil {
		t.Fatal(err)
	}
	next, err := Open(context.Background(), ModelSpec{Name: "org/next", File: nextPath, QueryPrefix: "query: "}, rt)
	if err != nil {
		t.Fatalf("second model: %v", err)
	}
	var loaded []any
	for _, c := range f.calls {
		if c.name == "loadModel" {
			loaded = append(loaded, c.args[0])
		}
	}
	if len(loaded) != 2 || loaded[0] != currentPath || loaded[1] != nextPath {
		t.Fatalf("loaded %v, want %s then %s", loaded, currentPath, nextPath)
	}
	if n := f.count("load"); n != 1 {
		t.Fatalf("the libraries were loaded %d times, want once", n)
	}
	f.calls = nil
	if _, err := current.Embed(context.Background(), semsearch.RoleQuery, []string{"x"}); err != nil {
		t.Fatal(err)
	}
	if _, err := next.Embed(context.Background(), semsearch.RoleQuery, []string{"x"}); err != nil {
		t.Fatal(err)
	}
	var seen []any
	for _, c := range f.calls {
		if c.name == "tokenize" {
			seen = append(seen, c.args[1])
		}
	}
	if len(seen) != 2 || seen[0] != "x" || seen[1] != "query: x" {
		t.Fatalf("tokenized %v, want x for the current model and query: x for the next", seen)
	}
}

package textsearch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The eval golden texts the samples come from.
var (
	goldenFile = filepath.Join("..", "..", "eval", "testdata", "golden.yaml")
	longDir    = filepath.Join("..", "..", "eval", "testdata", "long")
)

func TestWriteSamples_WritesBothSetsOnce(t *testing.T) {
	dir := t.TempDir()
	n, err := WriteSamples(dir, goldenFile, longDir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 84+24 {
		t.Errorf("wrote %d files, want the 84 golden and 24 long documents", n)
	}
	short, err := os.ReadFile(filepath.Join(dir, "golden", "en-go-modules.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(short), "# Managing dependencies with Go modules\n\nHow go.mod and go.sum pin") || !strings.Contains(string(short), "A Go module is a tree of packages") {
		t.Errorf("a golden document: %.120q", short)
	}
	long, err := os.ReadFile(filepath.Join(dir, "long", "en-honey-bee.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(long), "# The honey bee\n\n") || !strings.Contains(string(long), "\n## Anatomy and life cycle\n") {
		t.Errorf("a long document: %.120q", long)
	}

	// a file that is there is kept, the user's edits with it
	edited := filepath.Join(dir, "golden", "en-go-modules.md")
	write(t, edited, "mine")
	if n, err = WriteSamples(dir, goldenFile, longDir); err != nil || n != 0 {
		t.Errorf("a second run wrote %d files, %v", n, err)
	}
	if data, _ := os.ReadFile(edited); string(data) != "mine" {
		t.Errorf("an existing file was overwritten: %q", data)
	}

	// one set alone, and none
	if n, err = WriteSamples(t.TempDir(), "", longDir); err != nil || n != 24 {
		t.Errorf("the long set alone: %d, %v", n, err)
	}
	if n, err = WriteSamples(t.TempDir(), "", ""); err != nil || n != 0 {
		t.Errorf("no set: %d, %v", n, err)
	}
}

func TestWriteSamples_Errors(t *testing.T) {
	if _, err := WriteSamples(t.TempDir(), filepath.Join(t.TempDir(), "missing.yaml"), ""); err == nil {
		t.Error("a missing golden file was accepted")
	}
	if _, err := WriteSamples(t.TempDir(), "", t.TempDir()); err == nil {
		t.Error("a directory without long documents was accepted")
	}
	// the target is a file, not a directory
	file := filepath.Join(t.TempDir(), "file")
	write(t, file, "x")
	if _, err := WriteSamples(file, goldenFile, ""); err == nil {
		t.Error("a file as the target directory was accepted")
	}
	if _, err := WriteSamples(file, "", longDir); err == nil {
		t.Error("a file as the target directory was accepted for the long set")
	}
}

func TestSampleQueries_CrossALanguage(t *testing.T) {
	qs := SampleQueries(goldenFile, 4)
	if len(qs) != 4 {
		t.Fatalf("got %d queries, want 4: %q", len(qs), qs)
	}
	seen := map[string]bool{}
	for _, q := range qs {
		if q == "" || seen[q] {
			t.Errorf("an empty or repeated query in %q", qs)
		}
		seen[q] = true
	}
	if all := SampleQueries(goldenFile, 1000); len(all) < 4 || len(all) >= 1000 {
		t.Errorf("asked for more than there are: %d", len(all))
	}
	if qs := SampleQueries(filepath.Join(t.TempDir(), "missing.yaml"), 4); qs != nil {
		t.Errorf("queries of a missing file: %q", qs)
	}
}

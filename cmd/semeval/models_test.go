package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/semsearch/yzma"
)

func writeModels(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "models.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadModels_ParsesEntries(t *testing.T) {
	path := writeModels(t, `
- name: org/a
  url: https://example.org/a.gguf
  sha256: `+strings.Repeat("a", 64)+`
  pooling: cls
  query_prefix: "query: "
  document_prefix: "passage: "
  max_tokens: 256
- name: local
  file: /models/local.gguf
`)
	entries, err := loadModels(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 || entries[0].Name != "org/a" || entries[0].Pooling != "cls" || entries[0].QueryPrefix != "query: " || entries[0].MaxTokens != 256 {
		t.Fatalf("entries = %+v", entries)
	}
	spec := entries[0].spec("/cache/models")
	if spec.File != "/cache/models/a.gguf" || spec.Pooling != yzma.PoolingCLS || spec.URL != "https://example.org/a.gguf" || spec.DocumentPrefix != "passage: " || spec.MaxTokens != 256 {
		t.Fatalf("spec = %+v", spec)
	}
	if local := entries[1].spec("/cache/models"); local.File != "/models/local.gguf" || local.URL != "" || local.Pooling != yzma.PoolingMean {
		t.Fatalf("local spec = %+v, want the given file and mean pooling", local)
	}
}

func TestLoadModels_RejectsBadEntries(t *testing.T) {
	cases := []struct{ name, body, want string }{
		{"no name", "- url: https://x/a.gguf\n", "entries[0]: name is required"},
		{"neither file nor url", "- name: a\n", `"a": file or url is required`},
		{"unknown pooling", "- name: a\n  file: /a.gguf\n  pooling: average\n", "mean, cls, last"},
		{"unknown key", "- name: a\n  file: /a.gguf\n  querry_prefix: x\n", "querry_prefix"},
		{"duplicate name", "- name: a\n  file: /a.gguf\n- name: a\n  file: /b.gguf\n", `duplicate name "a"`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := loadModels(writeModels(t, c.body))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want it to contain %q", err, c.want)
			}
		})
	}
}

func TestSelectEntries_UnknownNameIsAnError(t *testing.T) {
	entries := []entry{{Name: "a"}, {Name: "b"}, {Name: "c"}}
	got, err := selectEntries(entries, "c, a")
	if err != nil || len(got) != 2 || got[0].Name != "a" || got[1].Name != "c" {
		t.Fatalf("selectEntries = %+v, %v; want a and c in file order", got, err)
	}
	if all, err := selectEntries(entries, ""); err != nil || len(all) != 3 {
		t.Fatalf("empty filter = %+v, %v; want everything", all, err)
	}
	_, err = selectEntries(entries, "a,zzz")
	if err == nil || !strings.Contains(err.Error(), `"zzz"`) || !strings.Contains(err.Error(), "a, b, c") {
		t.Fatalf("err = %v, want the unknown name and the known ones", err)
	}
}

// TestModelsFile_IsValid keeps the checked-in candidates loadable and pinned.
func TestModelsFile_IsValid(t *testing.T) {
	entries, err := loadModels("models.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) < 2 {
		t.Fatalf("models.yaml lists %d models, want the baseline and at least one candidate", len(entries))
	}
	if entries[0].Name != yzma.DefaultModel {
		t.Errorf("first entry is %q, want the default model as baseline", entries[0].Name)
	}
	for _, e := range entries {
		if e.URL != "" && len(e.SHA256) != 64 {
			t.Errorf("%s: unpinned (sha256 %q); run the command once and paste the digest it reports", e.Name, e.SHA256)
		}
	}
}

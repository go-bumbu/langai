package eval_test

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"

	"github.com/go-bumbu/langai/semsearch/yzma"
)

// entry is one row of the models file: a GGUF to evaluate and how to run it.
// Either url (+ sha256, once pinned) or file names the model.
type entry struct {
	Name           string `yaml:"name" json:"name"`
	URL            string `yaml:"url" json:"url,omitempty"`
	SHA256         string `yaml:"sha256" json:"sha256,omitempty"`
	Size           int64  `yaml:"size" json:"size,omitempty"`
	File           string `yaml:"file" json:"file,omitempty"`
	Pooling        string `yaml:"pooling" json:"pooling,omitempty"`
	QueryPrefix    string `yaml:"query_prefix" json:"query_prefix,omitempty"`
	DocumentPrefix string `yaml:"document_prefix" json:"document_prefix,omitempty"`
	MaxTokens      int    `yaml:"max_tokens" json:"max_tokens,omitempty"`
}

// poolings are the accepted pooling names, for error messages.
const poolings = "mean, cls, last"

// loadModels reads and validates the models file. Unknown keys are an error
// so a misspelt prefix cannot silently evaluate a model without it.
func loadModels(path string) ([]entry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)
	var entries []entry
	if err := dec.Decode(&entries); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	seen := make(map[string]bool, len(entries))
	for i, e := range entries {
		switch {
		case e.Name == "":
			return nil, fmt.Errorf("%s: entries[%d]: name is required", path, i)
		case seen[e.Name]:
			return nil, fmt.Errorf("%s: duplicate name %q", path, e.Name)
		case e.File == "" && e.URL == "":
			return nil, fmt.Errorf("%s: %q: file or url is required", path, e.Name)
		}
		switch yzma.Pooling(e.Pooling) {
		case "", yzma.PoolingMean, yzma.PoolingCLS, yzma.PoolingLast:
		default:
			return nil, fmt.Errorf("%s: %q: unknown pooling %q, use one of %s", path, e.Name, e.Pooling, poolings)
		}
		seen[e.Name] = true
	}
	return entries, nil
}

// selectEntries keeps the entries named in only (comma-separated), in file
// order; an empty only keeps everything. An unknown name is an error that
// lists the known ones.
func selectEntries(entries []entry, only string) ([]entry, error) {
	if strings.TrimSpace(only) == "" {
		return entries, nil
	}
	known := make([]string, 0, len(entries))
	byName := make(map[string]bool, len(entries))
	for _, e := range entries {
		known = append(known, e.Name)
		byName[e.Name] = true
	}
	wanted := map[string]bool{}
	for _, name := range strings.Split(only, ",") {
		name = strings.TrimSpace(name)
		if !byName[name] {
			return nil, fmt.Errorf("unknown model %q; the models file lists: %s", name, strings.Join(known, ", "))
		}
		wanted[name] = true
	}
	out := make([]entry, 0, len(wanted))
	for _, e := range entries {
		if wanted[e.Name] {
			out = append(out, e)
		}
	}
	return out, nil
}

// spec turns an entry into a ModelSpec; a downloaded model lives under
// modelsDir named after its URL's last path element, and an entry without a
// pooling gets mean.
func (e entry) spec(modelsDir string) yzma.ModelSpec {
	file := e.File
	if file == "" {
		file = filepath.Join(modelsDir, path.Base(e.URL))
	}
	pooling := yzma.Pooling(e.Pooling)
	if pooling == "" {
		pooling = yzma.PoolingMean
	}
	return yzma.ModelSpec{
		Name:           e.Name,
		File:           file,
		URL:            e.URL,
		SHA256:         e.SHA256,
		Size:           e.Size,
		Pooling:        pooling,
		QueryPrefix:    e.QueryPrefix,
		DocumentPrefix: e.DocumentPrefix,
		MaxTokens:      e.MaxTokens,
	}
}

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
	entries, err := loadModels("testdata/models.yaml")
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
			t.Errorf("%s: unpinned (sha256 %q); run make semeval once and paste the digest it logs", e.Name, e.SHA256)
		}
	}
}

package main

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

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
	f, err := os.Open(path) //nolint:gosec // path is an operator's flag, read once
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

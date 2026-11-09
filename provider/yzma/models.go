package yzma

import (
	"fmt"
	"path/filepath"
	"strings"
)

// pinnedModel is one embedding model New can run by name: its pinned
// download, the file it is kept in, and the way its authors trained it to be
// run.
type pinnedModel struct {
	name           string
	file           string
	asset          asset
	pooling        Pooling
	queryPrefix    string
	documentPrefix string
	dims           int
}

// pinnedModels lists the embedding models Config.Model selects from in New,
// DefaultModel first. Each is run with the settings make semeval evaluates
// it with (see zarf/eval/testdata/models.yaml), and its digest pins the
// GGUF that was evaluated. A model's file name in ModelsDir is fixed here
// rather than taken from its URL, since operators pre-seed it for offline
// installs; it is the name make semeval downloads the model under, so the
// two can share the directory. The list is built per call so it reads defaultModelAsset as it
// is (tests point it at a local server).
func pinnedModels() []pinnedModel {
	return []pinnedModel{
		{
			name:           DefaultModel,
			file:           defaultModelFile,
			asset:          defaultModelAsset,
			pooling:        PoolingMean,
			queryPrefix:    "task: search result | query: ",
			documentPrefix: "title: none | text: ",
			dims:           768,
		},
		{
			// the F16 conversion (Apache-2.0); the same bytes are published by
			// leliuga/ and second-state/ on Hugging Face
			name: "sentence-transformers/all-MiniLM-L6-v2",
			file: "all-MiniLM-L6-v2.F16.gguf",
			asset: asset{
				URL:    "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.F16.gguf",
				SHA256: "797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662",
				Size:   45949216,
			},
			pooling: PoolingMean,
			dims:    384,
		},
		{
			name: "intfloat/multilingual-e5-small",
			file: "multilingual-e5-small-q8_0.gguf",
			asset: asset{
				URL:    "https://huggingface.co/TwinSunsLLC/multilingual-e5-small-gguf/resolve/main/multilingual-e5-small-q8_0.gguf",
				SHA256: "e011debc1208e31bf7b6aebee2d9fc8bd2ca11694a77ed66ac9d0c9d0a877c93",
				Size:   132439008,
			},
			pooling:        PoolingMean,
			queryPrefix:    "query: ",
			documentPrefix: "passage: ",
			dims:           384,
		},
		{
			name: "BAAI/bge-m3",
			file: "bge-m3-Q8_0.gguf",
			asset: asset{
				URL:    "https://huggingface.co/smarttasks/bge-m3-GGUF/resolve/main/bge-m3-Q8_0.gguf",
				SHA256: "aa473d51f451a22f0fcf39ba3330c14bed38a385712b1113440f69df4047a173",
				Size:   634553760,
			},
			pooling: PoolingCLS,
			dims:    1024,
		},
		{
			name: "nomic-ai/nomic-embed-text-v2-moe",
			file: "nomic-embed-text-v2-moe-q8_0.gguf",
			asset: asset{
				URL:    "https://huggingface.co/ggml-org/Nomic-Embed-Text-V2-GGUF/resolve/main/nomic-embed-text-v2-moe-q8_0.gguf",
				SHA256: "36c5817bc25f379e62021f49efde05b10ed3b0c93ab8059c43173a7a5de73565",
				Size:   512225120,
			},
			pooling:        PoolingMean,
			queryPrefix:    "search_query: ",
			documentPrefix: "search_document: ",
			dims:           768,
		},
	}
}

// pinnedCompleter is one chat model NewCompleter can run by name: its pinned
// download, the file it is kept in, and the chat template its authors trained
// it with.
type pinnedCompleter struct {
	name     string
	file     string
	asset    asset
	template string
}

// pinnedCompleters lists the chat models Config.Model selects from in
// NewCompleter, DefaultCompleterModel first. As with pinnedModels, the list is
// built per call so it reads defaultCompleterAsset as it is (tests point it at
// a local server).
func pinnedCompleters() []pinnedCompleter {
	return []pinnedCompleter{
		{
			name:     DefaultCompleterModel,
			file:     defaultCompleterFile,
			asset:    defaultCompleterAsset,
			template: "chatml",
		},
	}
}

// CompleterModels lists the names of the pinned chat models NewCompleter
// selects from, DefaultCompleterModel first.
func CompleterModels() []string {
	return names(pinnedCompleters(), func(m pinnedCompleter) string { return m.name })
}

// LookupCompleterSpec returns the pinned spec of the named chat model, its
// file inside modelsDir ("" means "models"). name "" means
// DefaultCompleterModel; any other name must be one of CompleterModels, and
// the error for one that is not lists them.
func LookupCompleterSpec(name, modelsDir string) (CompleterSpec, error) {
	if name == "" {
		name = DefaultCompleterModel
	}
	for _, m := range pinnedCompleters() {
		if m.name == name {
			return CompleterSpec{
				Name:     m.name,
				File:     filepath.Join(modelsOr(modelsDir), m.file),
				URL:      m.asset.URL,
				SHA256:   m.asset.SHA256,
				Size:     m.asset.Size,
				Template: m.template,
			}, nil
		}
	}
	return CompleterSpec{}, fmt.Errorf("yzma: unsupported chat model %q: use one of %s", name, strings.Join(CompleterModels(), ", "))
}

// DefaultCompleterSpec is LookupCompleterSpec(DefaultCompleterModel,
// modelsDir): the pinned Qwen3 4B Instruct Q4_K_M GGUF inside modelsDir ("" means
// "models") and its ChatML template.
func DefaultCompleterSpec(modelsDir string) CompleterSpec {
	spec, _ := LookupCompleterSpec(DefaultCompleterModel, modelsDir) // DefaultCompleterModel is always pinned
	return spec
}

// Models lists the names of the pinned embedding models New selects from,
// DefaultModel first.
func Models() []string {
	return names(pinnedModels(), func(m pinnedModel) string { return m.name })
}

// LookupSpec returns the pinned spec of the named embedding model, its file
// inside modelsDir ("" means "models"). name "" means DefaultModel; any
// other name must be one of Models, and the error for one that is not lists
// them. Its Dims lets a caller size a vector store before the model is
// loaded.
func LookupSpec(name, modelsDir string) (ModelSpec, error) {
	if name == "" {
		name = DefaultModel
	}
	for _, m := range pinnedModels() {
		if m.name == name {
			return ModelSpec{
				Name:           m.name,
				File:           filepath.Join(modelsOr(modelsDir), m.file),
				URL:            m.asset.URL,
				SHA256:         m.asset.SHA256,
				Size:           m.asset.Size,
				Pooling:        m.pooling,
				QueryPrefix:    m.queryPrefix,
				DocumentPrefix: m.documentPrefix,
				Dims:           m.dims,
			}, nil
		}
	}
	return ModelSpec{}, fmt.Errorf("yzma: unsupported embedding model %q: use one of %s", name, strings.Join(Models(), ", "))
}

// modelsOr is dir, or "models" when it is empty.
func modelsOr(dir string) string {
	if dir == "" {
		return "models"
	}
	return dir
}

// names maps the pinned entries to their names, in order.
func names[T any](pinned []T, name func(T) string) []string {
	out := make([]string, len(pinned))
	for i, p := range pinned {
		out[i] = name(p)
	}
	return out
}

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

// pinnedModels lists the embedding models Config.Model selects from,
// DefaultModel first. Each is run with the settings make semeval evaluates
// it with (see semsearch/eval/testdata/models.yaml), and its digest pins the
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
			},
			pooling:        PoolingMean,
			queryPrefix:    "search_query: ",
			documentPrefix: "search_document: ",
			dims:           768,
		},
	}
}

// Spec returns the pinned spec of the named model, its file inside modelsDir
// ("" means "models"). name "" means DefaultModel; any other name must be one
// of the pinned models, and the error for one that is not lists them. Its
// Dims lets a caller size a vector store before the model is loaded.
func Spec(name, modelsDir string) (ModelSpec, error) {
	if name == "" {
		name = DefaultModel
	}
	if modelsDir == "" {
		modelsDir = "models"
	}
	models := pinnedModels()
	names := make([]string, 0, len(models))
	for _, m := range models {
		if m.name == name {
			return ModelSpec{
				Name:           m.name,
				File:           filepath.Join(modelsDir, m.file),
				URL:            m.asset.URL,
				SHA256:         m.asset.SHA256,
				Size:           m.asset.Size,
				Pooling:        m.pooling,
				QueryPrefix:    m.queryPrefix,
				DocumentPrefix: m.documentPrefix,
				Dims:           m.dims,
			}, nil
		}
		names = append(names, m.name)
	}
	return ModelSpec{}, fmt.Errorf("unsupported embedding model %q: use one of %s", name, strings.Join(names, ", "))
}

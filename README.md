# sense

sense is a set of Go libraries for semantic search, image and video similarity, and
automatic tagging, with the models running inside your process through llama.cpp and
ONNX Runtime. Nothing leaves the machine: no server, no API key.

[![CI](https://github.com/go-bumbu/sense/actions/workflows/validate.yml/badge.svg)](https://github.com/go-bumbu/sense/actions/workflows/validate.yml)
[![License: LGPL v3](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](LICENSE)

> **Pre-release:** the API still changes between minor versions; see
> [docs/upgrading.md](docs/upgrading.md).

## Features

- Search your documents by meaning, in English, Spanish, German and other languages
- Find images and videos that look like a picture, or that match a text description
- Find copies of a picture (re-encoded, resized, lightly edited) without any model
- Suggest tags for a text from the tags you already use, learning from the ones people set by hand
- Write new tags for a text with a local chat model
- Keep the vectors in SQLite or Postgres; models and runtimes download on first use

## Try it

The example service has a page per feature: media search, text search, duplicates and
auto tagging. It runs from a clone and needs the system packages in
[docs/requirements.md](docs/requirements.md), plus `curl` and `ffmpeg`.

```sh
git clone https://github.com/go-bumbu/sense.git && cd sense
make example-prepare  # fetch sample photos and clips, write sample documents
make example          # index them and serve the pages
```

Open <http://127.0.0.1:8085/>. The models download the first time a page needs them.

## Add to your project

```sh
go get github.com/go-bumbu/sense@latest
```

Needs Go 1.26+. The llama.cpp provider needs `libffi8` at runtime; SQLite and ONNX
Runtime need a C compiler and `libsqlite3-dev`; videos need `ffmpeg`. Details are in
[docs/requirements.md](docs/requirements.md).

### Search text by meaning

Error checks are left out of the examples for brevity.

```go
// open the SQLite database that stores the vectors
db, err := sql.Open("sqlite3", "search.db?_busy_timeout=5000&_journal_mode=WAL")

// load the embedding model (llama.cpp and the model download on first use)
cfg := yzma.Config{
    ModelsDir: "data/models",
    Runtime:   yzma.Runtime{LibDir: "data/llama-cpp", ManagedLibDir: true},
}
emb, err := yzma.New(ctx, cfg)
store, err := sqlitevec.New(ctx, db, emb.Dimensions())
ix, err := search.New(emb, store, "tenant-1", "space-1", "notes")

// index a document (a no-op while its text is unchanged)
err = ix.Index(ctx, "note-1", "Cook the spaghetti for nine minutes in salted water.")

// search by meaning, best match first
hits, err := ix.Search(ctx, "how long to boil pasta", 10)
for _, h := range hits {
    fmt.Println(h.DocID, h.Score, h.Text)
}
```

### Find similar images

```go
// load the image model (SigLIP 2, downloaded on first use)
emb, err := onnx.New(ctx, onnx.Config{
    ModelsDir: "data/models",
    Runtime:   onnx.Runtime{LibDir: "data/onnxruntime", ManagedLibDir: true},
})
store, err := sqlitevec.New(ctx, db, emb.Dimensions())
ix, err := search.NewMediaIndex(emb, store, "tenant-1", "space-1", "photos")

// index a picture
f, err := os.Open("beach.jpg")
err = ix.IndexImage(ctx, "beach", f)

// find the pictures that look like it
hits, err := ix.SearchIndexed(ctx, "beach", 10)
```

### Write tags for a text

```go
// load the chat model that writes the tags (2.5 GB, downloaded on first use)
llm, err := yzma.NewCompleter(ctx, cfg)

// write tags for a text; the embedding model scores each one
gen, err := tagging.NewGenerator(llm, emb, tagging.GeneratorConfig{})
tags, err := gen.Generate(ctx, text, tagging.DefaultGeneratorMinScore)
```

More in the docs: [packages](docs/packages.md), [semantic search](docs/search.md),
[image and video search](docs/media-search.md) (videos, duplicates, text-to-image),
[tagging](docs/tagging.md) (the `Tagger` that learns from your tags), and
[upgrading](docs/upgrading.md).

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md).

## License

Licensed under the [GNU Lesser General Public License v3.0](LICENSE).

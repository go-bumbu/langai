# Developing sense

## Overview

sense is one Go module. The root package holds the shared ports (`Embedder`,
`ImageEmbedder`, `VideoEmbedder`, `Completer`) and the lazy loaders; the capabilities
live in `search/` and `tagging/`; implementations are grouped by kind in `provider/`
(model runtimes), `vector/` (vector stores) and `media/` (video decoding). `sensetest/`
has fakes of the ports for tests, `internal/` the shared helpers, and `zarf/` the tooling
that is not part of the library: the model evaluation (`zarf/eval`) and the example
service (`zarf/examples`). [docs/packages.md](docs/packages.md) has the full tree and
the dependency rules; [TODO.md](TODO.md) lists the planned packages.

## Prerequisites

- Go 1.26 or newer (`go.mod`; the pinned yzma release needs it).
- A C compiler and `libsqlite3-dev`: `vector/sqlitevec` and `provider/onnx` are CGO.
- `ffmpeg` and `ffprobe` 5.1+, or the `media/ffmpeg` tests skip and the package falls
  under the coverage gate.
- For the real-model tests: `libffi8`, `libgomp1` and `libstdc++6`.
- Docker, for `make coverage`: it sets `TESTDBS_ALL=1`, which starts a pgvector
  container so the `vector/pgvec` and `tagging/pgcache` tests run.
- [golangci-lint](https://github.com/golangci/golangci-lint) v2 (CI pins v2.13.2) and
  [go-licence-detector](https://github.com/elastic/go-licence-detector) (CI pins v0.10.0).

On Debian/Ubuntu:

```sh
sudo apt-get install -y libsqlite3-dev ffmpeg libffi8 libgomp1 libstdc++6
```

## Common tasks

```sh
make help          # list every target
make test          # go tests (pgvector ones skip without TESTDBS_ALL=1 and Docker, ffmpeg ones without ffmpeg)
make coverage      # per-package coverage gate (>= 70%), pgvector tests included
make lint          # golangci-lint
make license-check # go-licence-detector against allowedLicenses.json
make benchmark     # go benchmarks
make verify        # the full check: test, license-check, lint, benchmark and coverage
```

Run `make verify` before opening a PR; CI (`.github/workflows/validate.yml`) runs the
same checks on every push, plus `make test-models` and `make test-images`.

### Real-model tests

These download the pinned runtimes and models on first run and cache them in
`~/.cache/semsearch-yzma` and `~/.cache/semsearch-onnx`.

```sh
make test-models   # embedding smoke test and golden-set quality floors (~350 MB)
make test-images   # ONNX Runtime + SigLIP 2 and the media index end to end (~380 MB)
make test-vision   # the vision embedder on the real model (2.7 GB; loads it twice)
make test-gentags  # the tag Generator on the real models (adds the 2.5 GB chat model; minutes on a CPU)
```

`SEMSEARCH_ONNX_MODEL=facebook/dinov2-base make test-images` runs the image tests on the
other pinned model.

### Evaluating embedding models

`zarf/eval` scores an embedding model on a labelled, multilingual golden set
(`zarf/eval/testdata/golden.yaml`, English, Spanish and German): free-text retrieval
(recall@k, MRR, nDCG) over the queries in their documents' language and, separately,
over the queries that must cross a language; tagging by tag name (precision, recall and
F1 at 0.46, each tag judged on its own, plus the floor each model would need); and
timings. tagging's `TestGoldenTagging` replays the same set through a `Tagger` with
simulated hand-tagging (a tag's members are up to every other document carrying it) and
checks tagging by name only and with members, how many documents stay untagged once the
whole families of their tags are removed (and how many get none but an ancestor once
only their own tags are), and two-topic pages.

```sh
make test-models   # the default model against the floors CI enforces (provider/yzma/golden_test.go, tagging/golden_test.go)
make semeval       # every candidate in zarf/eval/testdata/models.yaml, side by side
```

Add a candidate to `zarf/eval/testdata/models.yaml` with its pooling and prefixes; run
`make semeval SEMEVAL_ONLY=<its name>` once without a digest and paste the one it
prints. The dataset format is plain YAML, so an export of your own documents can be
scored the same way (`make semeval SEMEVAL_DATASET=export.yaml`); `SEMEVAL_JSON=out.json`
keeps every metric, and the Makefile lists the other settings.

## Run the example service

`zarf/examples/` is a single-user web service that exercises the features by hand, one
page per feature, each a debug view with the numbers behind every result.

```sh
make example-prepare  # fetch sample photos, make copies and clips, write sample documents (needs curl and ffmpeg)
make example          # index them and serve the pages on http://127.0.0.1:8085/
```

- **Status** (the home page): every model with the state of its lazy loader, its
  identity and what runs it, and a *load now*; every collection with its store, the
  model the store recorded for it and a *reset*; the Tagger's cache and its last prune.
- **Media search**: a grid of thumbnails; click one to see the items that look like it
  with their scores, the frame of a video that matched and when, the frames the index
  saw of a video, and how long the search took; or search with a file of your own, or
  by text with the vision model.
- **Text search**: `search.Index` over a directory of Markdown and text files, with what
  the last `IndexBatch` did, the chunk behind every hit, a document cut into the chunks
  that were embedded, and "more like this".
- **Duplicates**: `provider/phash` both ways, as groups of pictures by the Hamming
  distance of their raw hashes and as a `MediaIndex` over the hash, with the cosine next
  to the distance it stands for.
- **Auto tagging**: a notebook whose notes get tags from your own tags (`Tagger`) and new
  ones written by a chat model (`Generator`), with every score and cutoff.

The models load through `LazyEmbedder`, `LazyImageEmbedder` and `LazyCompleter`, so the
service is up at once. Settings pass through to `zarf/examples`' own Makefile, for
example `make example MODEL=qwen3vl ADDR=0.0.0.0:8085`, and `make example POSTGRES=<dsn>`
switches the stores from SQLite files to `vector/pgvec` and `tagging/pgcache`. See
[zarf/examples/README.md](zarf/examples/README.md).

## Release

sense is a library, so a release is a Git tag; the Go module proxy serves it from there,
and CI publishes nothing.

```sh
make tag version=v1.2.3
```

`make tag` refuses to run off `main` or with unstaged changes, runs `make verify`,
then creates the annotated tag and pushes it to `origin`.

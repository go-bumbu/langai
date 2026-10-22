# langai

Libraries that use language models, run locally: semantic search over your own
documents, and automatic tagging that learns from the tags people assign by hand.
Nothing leaves the process: the embedding model runs in-process through llama.cpp.

## Packages

| Package | What it does |
|---|---|
| `semsearch` | Turns documents into embedding vectors and answers free-text queries with the closest ones. The `Embedder` and `Store` interfaces, the markdown-aware chunker, `MemStore` (exact, in memory, for tests and evaluation), `LazyEmbedder` (defers loading a model to its first use) and `FakeEmbedder` (deterministic, for tests). |
| `semsearch/sqlitevec` | `Store` on SQLite with the sqlite-vec extension (CGO). |
| `semsearch/pgvec` | `Store` on Postgres with pgvector. |
| `semsearch/yzma` | `Embedder` that runs a GGUF sentence-embedding model through llama.cpp, loaded at runtime by [yzma](https://github.com/hybridgroup/yzma) (purego + libffi, no CGO). Downloads and verifies the pinned llama.cpp build and model on first use. |
| `semsearch/eval` | Measures an `Embedder` on a labelled, multilingual golden set: retrieval, tagging, timings. |
| `autotag` | Assigns tags to documents: each tag's vector blends its name with the documents tagged with it by hand, and each tag is judged against its own cutoff. |
| `cmd/semeval` | Compares embedding models side by side on the golden set. |

Every package stands alone and you only compile the dependencies of the packages you
import: CGO and sqlite-vec come in only with `semsearch/sqlitevec`, pgvector only with
`semsearch/pgvec`, llama.cpp only with `semsearch/yzma`.

## Install

```
go get github.com/go-bumbu/langai
```

## Requirements

- **Go 1.26** or newer: the pinned yzma release needs it.
- **`semsearch/sqlitevec`** is CGO and needs the system SQLite headers to compile:
  `sudo apt-get install -y libsqlite3-dev` (Debian/Ubuntu). The extension registers
  itself for every SQLite connection the process opens after the package is imported,
  so import it (blank if need be) before opening the first connection.
- **`semsearch/pgvec`** needs a Postgres with the pgvector extension, or a user allowed
  to run `CREATE EXTENSION vector`.
- **`semsearch/yzma`** needs `libffi8` wherever the binary runs: the Go bindings open it
  in a package `init()` and panic without it, so a binary that imports the package needs
  it even when it never embeds anything. The llama.cpp libraries also need `libgomp1`,
  `libstdc++6` and glibc 2.34 or newer:
  `sudo apt-get install -y libffi8 libgomp1 libstdc++6`.
- **Keep yzma where langai pins it.** The llama.cpp build `semsearch/yzma` downloads
  (`yzma.LibTag`) matches the yzma version in this module's `go.mod`, whose Go
  bindings mirror exactly that build's ABI. A module that imports `semsearch/yzma` must
  not require another yzma version: minimal version selection would hand these
  bindings to libraries they do not match.

## Use

### Semantic search

```go
emb, err := yzma.New(ctx, yzma.Config{
    ModelsDir:     "data/models",
    LibDir:        filepath.Join("data", "llama-cpp", yzma.LibTag), // named after the pinned build
    ManagedLibDir: true,                                            // download the pinned build into it
})
store, err := sqlitevec.New(ctx, db, emb.Dimensions())

// index (e.g. a tenant) is the hard isolation boundary; partition and collection
// locate the documents within it
ix, err := semsearch.New(emb, store, "tenant-1", "space-7", "notes", semsearch.WithMinScore(0.3))
err = ix.Index(ctx, "note-42", markdownText) // a no-op while the text is unchanged
hits, err := ix.Search(ctx, "how long to boil spaghetti", 10)
```

Every `Store` backend keeps one physical table set per index, so one index's vectors are
never queryable from another's. A store opened with `WithGeneration` keeps further table
sets of its own, so a migration can build a second model's vectors next to the current
ones before switching over.

The first `yzma.New` downloads the llama.cpp libraries (~18 MB) and the default model,
`google/embeddinggemma-300m` (~329 MB, 768 dimensions, multilingual); both are pinned by
digest. To keep that off your startup path, wrap the loader in a `LazyEmbedder`, which
reports its dimensions from the model's spec without loading anything:

```go
spec := yzma.DefaultSpec("data/models")
emb := semsearch.NewLazyEmbedder(func() (semsearch.Embedder, error) {
    return yzma.New(ctx, cfg)
}, spec.Dims)
```

A failed load is cached and returned by every later call; `State` reports it, e.g. for
a health endpoint.

### Automatic tagging

```go
tagger := autotag.New(emb, store, autotag.Config{Model: yzma.DefaultModel})

vec, err := tagger.Embed(ctx, essence) // the document, embedded as a query
ids, err := tagger.Assign(ctx, autotag.Scope{Index: "tenant-1", Partition: "space-7"},
    space, // autotag.TagStore: your storage behind this scope's tags
    tags,  // []autotag.Tag: every current tag of the scope, {ID, Name, NameHash}
    vec)
```

`TagStore` is three methods: a cache of each tag's name embedding, and its *members*,
the embeddings of the documents tagged with it by hand (embedded with `tagger.Embed`;
cache them with the model name, since vectors of two models must never mix). Use a
tag's full path as its `Name` (`colors / warm / red` says more than `red`).

How a tag is judged:

- **Its vector** blends an embedding of its name with the centroid of its members,
  weighted by member count: `w = n/(n+SmoothingK)`. A brand-new tag tags by its name
  alone, and shifts toward "tag by example" as members accumulate.
- **Its cutoff** mixes `NameMinScore` (default 0.46) and `MemberMinScore` (default
  0.70) by the same weight. A document scores ~0.49 on average against a right tag's
  name but ~0.75 against its members' centroid, because documents resemble each other
  far more than they resemble a short tag path, so each part is judged on its own scale:
  scoring the blended vector against the cutoff is exactly
  `(1-w)·(cos(name) - NameMinScore) + w·(cos(centroid) - MemberMinScore) >= 0`.
- **Every tag that clears its own cutoff is assigned**, with no limit on how many.
- **Ghosts:** the search has one slot per current tag, so a vector left behind by a
  deleted tag could take a current tag's place. `Assign` deletes every such vector its
  search meets and searches again until none is new. Deleting a tag's vector when the
  tag itself is deleted is still the caller's job (`Store.Delete` under the scope's
  coordinates and `Config.Collection`).
- **Rejected alternatives:** kNN voting on its own (it never abstains); a kNN agreement
  check (it blocks second tags on two-topic pages); one global cutoff (it fits only one
  stage, names or examples); examples only with one cutoff (closeness to examples
  drifts with their number); a cap on tags per document (every qualifying tag is
  wanted).

The golden numbers below measure tagging over *essences*: a page's title, excerpt and
site name followed by the first 2 KB of its text, one per line
(`eval.Document.Essence`). Hand the tagger your documents built the same way to get
what the numbers say.

### Known limitations

- **Second and side topics.** A page is one embedding of its first ~2 KB. When pairs of
  single-tag golden documents are stitched into one page, both tags are assigned a bit
  more than half the time (0.59, 60 of 102 pages); a minor side topic is rarely found.
- **Documents that fit no tag** can still get one, at most about 1 time in 4: with a
  document's own tags removed, 0.75 of the golden documents stay untagged. That is an
  upper bound on wrong tags, not a measured rate: the rest of the tag family stays, and
  some picks are related tags.
- **Tags that mix unrelated topics** average their members into one centroid that
  matches neither topic well.
- **The two cutoffs come from a small golden set** (84 documents, 15 tags, one model);
  the model's scores sit in a narrow band, so ±0.04 on the name cutoff already costs a
  lot of precision or recall. Re-check them on real data, and measure them again for
  another model.

## Evaluating embedding models

`semsearch/eval` scores an embedding model on a labelled, multilingual golden set
(`semsearch/eval/testdata/golden.yaml`, English, Spanish and German): free-text retrieval
(recall@k, MRR, nDCG) over the queries in their documents' language and, separately,
over the queries that must cross a language; tagging by tag name (precision, recall and
F1 at `NameMinScore`, plus the floor each model would need); and timings. autotag's
`TestGoldenTagging` replays the same set through a `Tagger` with simulated hand-tagging
(a tag's members are up to every other document carrying it) and checks tagging by name
only and with members, how many documents stay untagged once their own tags are removed,
and two-topic pages. Both need the runtime libraries above and cache the downloads in
`~/.cache/semsearch-yzma`:

```
# the default model against the floors CI enforces
# (semsearch/yzma/golden_test.go and autotag/golden_test.go)
make test-models

# every candidate in cmd/semeval/models.yaml, side by side
make semeval
```

Add a candidate to `cmd/semeval/models.yaml` with its pooling and prefixes; run the
command once without a digest and paste the one it prints. The dataset format is plain
YAML, so an export of your own documents can be scored the same way
(`go run ./cmd/semeval -dataset export.yaml`).

## Development

```
make test          # go tests (the pgvector ones skip without TESTDBS_ALL=1 and Docker)
make test-models   # the real-model tests (downloads ~350 MB on first run)
make coverage      # per-package coverage gate (>= 70%), pgvector tests included
make lint          # golangci-lint
make license-check # go-licence-detector
make verify        # run every check
```

## License

langai is licensed under the [GNU Lesser General Public License v3.0](LICENSE).

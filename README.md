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
| `semsearch/eval` | Measures an `Embedder` on a labelled, multilingual golden set: retrieval, tagging, timings. `make semeval` compares embedding models side by side on it. |
| `autotag` | Suggests tags for items and learns from the tags people put on them by hand: each tag's vector blends its name with the items tagged with it, and each tag is judged against its own cutoff. It keeps the tags, items and hand-made links in a `Store`: `MemStore` (in memory), `autotag/sqlitestore` (SQLite) or `autotag/pgstore` (Postgres), the last two over plain tables with no extension. |

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
tagger, err := autotag.New(emb, store, autotag.Config{Model: yzma.DefaultModel})
space := tagger.Scope("tenant-1", "space-7")
space.SetTag(ctx, "bread", "cooking / bread")   // create or rename
space.SetItem(ctx, "item-42", essence)          // create or edit; embeds only when text or model changed
space.TagByHand(ctx, "item-42", "bread")        // a person tagged it: teaches the tag
space.UntagByHand(ctx, "item-42", "bread")
ids, err := space.Suggest(ctx, "item-42")       // auto tags, best first; never stored as links
space.DeleteItem(ctx, "item-42"); space.DeleteTag(ctx, "bread")
space.Rebuild(ctx)                              // optional warm-up, e.g. after switching models
```

autotag owns what it learns from: your application reports its tags and its items
(bookmarks, notes, pages) by id and text, plus the tags people put on items by hand, and
asks for suggestions. As in semsearch, the first argument of `Scope` is the hard isolation
boundary (a tenant) and the second a space within it; scopes never see each other's tags,
items or links. Use a tag's full path as its name (`colors / warm / red` says more than
`red`). `Config.Model` names the embedding model, and every stored vector is kept with the
model it came from.

**Storage.** A `Store` keeps the tags, the items with their embeddings and the hand-made
links. `autotag.NewMemStore()` keeps them in memory, for tests and tiny corpora.
`autotag/sqlitestore` and `autotag/pgstore` keep them in a SQLite or Postgres database
that you open and pass to `New(ctx, db)`; your application imports the driver. Both use
plain tables and need no extension: a space has few tags, so the Tagger scores them in
Go. Each index gets a table set of its own, and a store opened with `WithGeneration`
keeps further table sets of its own, so a migration can build a second model's tags and
vectors next to the current ones before switching over. SQLite allows one writer at a
time: when several goroutines write, open it with `?_busy_timeout=5000&_txlock=immediate`
(for `go-sqlite3`: `sql.Open("sqlite3", "autotag.db?_busy_timeout=5000&_txlock=immediate")`),
so a writer waits for its turn instead of failing with "database is locked". Postgres
needs no such setup: concurrent writes queue on row locks. That locking relies on READ
COMMITTED, Postgres's default isolation level. `autotag/pgstore` begins its own
transactions at that level, but its single-statement writes run at the session's
default, so keep `default_transaction_isolation` at `read committed` for its
connections: under `repeatable read` or `serializable` they can fail with serialization
errors, which the store does not retry. `autotag/storetest` is the conformance suite for
a `Store` of your own.

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
- **Every tag that clears its own cutoff is suggested**, best score first, with no limit
  on how many. A suggestion is never stored as a link: only `TagByHand` teaches a tag.
- **Changes rebuild the affected tags right away.** Every change to what a tag's vectors
  depend on bumps the tag's version, and a rebuild stores its result only while the tag
  is still at the version it read, so a hand tag that lands mid-rebuild is never
  overwritten. A write whose rebuild fails stays stored and returns an error wrapping
  `autotag.ErrNotRebuilt`: retrying it is a safe no-op, and the next `Suggest` or
  `Rebuild` rebuilds the tag, as it does one whose rebuild was overtaken. A `Suggest`
  that cannot rebuild a tag still ranks the others and returns their ids with that
  error, leaving the tag out or scoring it with its previous vectors. A change to
  `NameMinScore`, `MemberMinScore` or `SmoothingK` leaves tags stale the same way and
  they are rebuilt lazily, without re-embedding anything.
- **Bulk writes rebuild as they go:** every hand tag rebuilds its tag at once, from all
  of its members, so importing n hand tags of one tag reads its members n times (about
  n²/2 rows), and every edit of an item rebuilds each tag it carries.
- **One model and one set of scoring settings per store:** every Tagger that shares a
  `Store` must use the same `Config.Model`, `NameMinScore`, `MemberMinScore` and
  `SmoothingK`. A Tagger takes tags built under another model or with other settings for
  stale and rebuilds them, re-embedding their members when the model differs, so two
  Taggers that differ rebuild each other's tags at every `Suggest` and never settle.
- **Switching models:** autotag stores each item's text next to its embedding, so after
  you change `Config.Model` it re-embeds the stored texts it needs, lazily, and never
  mixes the vectors of two models: a tag's members when the tag is rebuilt, an item at
  its next `Suggest`. To switch in place, stop every instance together, start them with
  the new model, and call `Scope.Rebuild` on every scope. To switch without stopping,
  open a new generation of the store with `WithGeneration`: its tables start empty, so
  replay every tag, item and hand-made link into it, and write to both stores until the
  cutover.
- **Rejected alternatives:** kNN voting on its own (it never abstains); a kNN agreement
  check (it blocks second tags on two-topic pages); one global cutoff (it fits only one
  stage, names or examples); examples only with one cutoff (closeness to examples
  drifts with their number); a cap on tags per document (every qualifying tag is
  wanted).

The golden numbers below measure tagging over *essences*: a page's title, excerpt and
site name followed by the first 2 KB of its text, one per line
(`eval.Document.Essence`). Pass `SetItem` your documents built the same way to get what
the numbers say.

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

# every candidate in semsearch/eval/testdata/models.yaml, side by side
make semeval
```

Add a candidate to `semsearch/eval/testdata/models.yaml` with its pooling and prefixes;
run `make semeval SEMEVAL_ONLY=<its name>` once without a digest and paste the one it
prints. The dataset format is plain YAML, so an export of your own documents can be
scored the same way (`make semeval SEMEVAL_DATASET=export.yaml`); `SEMEVAL_JSON=out.json`
keeps every metric, and the Makefile lists the other settings.

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

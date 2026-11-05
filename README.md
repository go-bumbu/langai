# langai

Libraries that use language models, run locally: semantic search over your own
documents, image and video similarity search, automatic tagging that learns from the
tags people assign by hand, and tags written for a text. Nothing leaves the process: the
models run in-process through llama.cpp and ONNX Runtime.

## Packages

| Package | What it does |
|---|---|
| `langai` | The ports: `Embedder` (text embedding models), `ImageEmbedder` (image embedding models), `Completer` (chat models) and the `Role` a text is embedded as. `LazyEmbedder` defers loading a model to its first use. |
| `search` | Turns documents into embedding vectors and answers free-text queries with the closest ones: `Index` and the markdown-aware chunker. `MediaIndex` does the same for pictures: it embeds images and the sampled frames of videos and answers an image or a video with the items that look most like it, across both kinds. |
| `media` | The port for video input: `Sampler` turns a video file into the `Frame`s that stand for it. |
| `media/ffmpeg` | `Sampler` over the ffmpeg and ffprobe binaries: the first frame, every scene cut and one frame per interval, scaled down, capped per video; or the keyframes only, for long videos. |
| `tagging` | Tags for a text. `Tagger` suggests tags among the ones an application already has and learns from the tags people put on items by hand: each tag's vector blends its name with the items tagged with it, and each tag is judged against its own cutoff. It keeps the tags, items and hand-made links in a `Store`: `MemStore` (in memory), `tagging/sqlitestore` (SQLite) or `tagging/pgstore` (Postgres), the last two over plain tables with no extension. `Generator` writes tags with no list to start from: a chat model proposes them and the embedding model scores each against the text, so one cutoff means the same on every text. |
| `vector` | The `Store` interface for search vectors, and `MemStore` (exact, in memory, for tests and evaluation). |
| `vector/sqlitevec` | `Store` on SQLite with the sqlite-vec extension (CGO). |
| `vector/pgvec` | `Store` on Postgres with pgvector. |
| `provider/yzma` | `Embedder` that runs a GGUF sentence-embedding model, and `Completer` that runs a GGUF chat model (the one the `Generator` writes tags with), through llama.cpp, loaded at runtime by [yzma](https://github.com/hybridgroup/yzma) (purego + libffi, no CGO). Downloads and verifies the pinned llama.cpp build and models on first use. |
| `provider/onnx` | `ImageEmbedder` that runs an ONNX vision model through [ONNX Runtime](https://onnxruntime.ai) (CGO, via [onnxruntime_go](https://github.com/yalue/onnxruntime_go)): SigLIP 2 base by default, DINOv2 base as the other pinned model, any other through a `ModelSpec`. CPU by default, CUDA or CoreML by configuration. Downloads and verifies the pinned runtime build and model on first use. |
| `eval` | Measures an `Embedder` on a labelled, multilingual golden set: retrieval, tagging, timings. `make semeval` compares embedding models side by side on it. |
| `langaitest` | `FakeEmbedder` and `FakeImageEmbedder`, deterministic, for tests. |

Every package stands alone and you only compile the dependencies of the packages you
import: CGO and sqlite-vec come in only with `vector/sqlitevec`, pgvector only with
`vector/pgvec`, llama.cpp only with `provider/yzma`, CGO and ONNX Runtime only with
`provider/onnx`, and the ffmpeg binary is only run by `media/ffmpeg`.

## Layout

The packages form a tree, and the tree is also the plan: packages marked *planned* do
not exist yet (see TODO.md).

```
langai/                 the shared ports: Embedder, Completer, Role
│
├── search/             text → similar documents; image or video → similar images and videos
├── tagging/            text → tags from your corpus (Tagger), text → new tags (Generator)
│   ├── sqlitestore/    the Tagger's Store on SQLite
│   ├── pgstore/        the Tagger's Store on Postgres
│   ├── storetest/      conformance suite for a Store
│   └── internal/       sqlstore: the SQL the two stores share
├── mediatag/           planned: image or video → tags
├── caption/            planned: image or video → description
│
├── provider/           model runtimes; each implements some of the ports
│   ├── yzma/           llama.cpp in process: text embeddings, chat model
│   ├── onnx/           ONNX Runtime in process (CGO): SigLIP 2 and DINOv2 image embeddings
│   ├── kronk/          planned: llama.cpp via kronk: embeddings, chat, vision, rerank
│   ├── openai/         planned: OpenAI-compatible HTTP, remote APIs or a kronk server
│   └── cache/          planned: wraps any provider, caches by content hash
├── vector/             Store for search vectors, in-memory MemStore
│   ├── sqlitevec/      SQLite with sqlite-vec (CGO)
│   ├── pgvec/          Postgres with pgvector
│   └── qdrant/         planned, if a dedicated vector database is needed
├── media/              video input: the Sampler port and Frame
│   └── ffmpeg/         video → frames via the ffmpeg binary
│
├── eval/               golden-set evaluation of embedding models
├── langaitest/         fakes of the ports, for tests
├── internal/           tablename: SQL table names; fetch: verified downloads and release
│                       archives for the providers; planned: rank, score roll-up
└── examples/           planned: end-to-end wiring
```

- **The root is the front door:** the shared ports and the capabilities (`search`,
  `tagging`, `mediatag`, `caption`) are what you import.
- **Each group holds one kind of implementation:** model runtimes in `provider/`, vector
  stores in `vector/`, video input in `media/`. A new backend is a new folder in its
  group.
- **Dependencies point one way:** a capability never imports an implementation, an
  implementation imports the package whose port it implements, and your application
  wires the two together.
- **One module for now.** The heavy implementations (kronk, hugot, the database stores)
  may later get a `go.mod` of their own, so that a text-only user pulls in no
  third-party code.

## Install

```
go get github.com/go-bumbu/langai
```

## Requirements

- **Go 1.26** or newer: the pinned yzma release needs it.
- **`vector/sqlitevec`** is CGO and needs the system SQLite headers to compile:
  `sudo apt-get install -y libsqlite3-dev` (Debian/Ubuntu). The extension registers
  itself for every SQLite connection the process opens after the package is imported,
  so import it (blank if need be) before opening the first connection.
- **`vector/pgvec`** needs a Postgres with the pgvector extension, or a user allowed
  to run `CREATE EXTENSION vector`.
- **`provider/yzma`** needs `libffi8` wherever the binary runs: the Go bindings open it
  in a package `init()` and panic without it, so a binary that imports the package needs
  it even when it never embeds anything. The llama.cpp libraries also need `libgomp1`,
  `libstdc++6` and glibc 2.34 or newer:
  `sudo apt-get install -y libffi8 libgomp1 libstdc++6`.
- **The chat model** (`yzma.NewCompleter`, for the tag `Generator`) is a 2.5 GB download on first use,
  and a process with it and the embedding model loaded holds about 5 GB of RAM.
- **Keep yzma where langai pins it.** The llama.cpp build `provider/yzma` downloads
  (`yzma.LibTag`) matches the yzma version in this module's `go.mod`, whose Go
  bindings mirror exactly that build's ABI. A module that imports `provider/yzma` must
  not require another yzma version: minimal version selection would hand these
  bindings to libraries they do not match.
- **`provider/onnx`** is CGO and needs a C compiler to build (no ONNX Runtime headers:
  the bindings carry their own). At runtime it needs the ONNX Runtime shared library:
  a managed `LibDir` downloads the pinned CPU build (`onnx.LibVersion`, ~11 MB) on
  linux/amd64, linux/arm64 and darwin/arm64. For **CUDA** set `Config.Provider` to
  `onnx.ProviderCUDA`: a managed `LibDir` on linux/amd64 downloads the CUDA 12 build
  (~450 MB), and the host needs CUDA 12 and cuDNN 9 installed; **CoreML** is built into
  the macOS build. The bindings' headers target one runtime release, so keep
  `onnxruntime_go` where this module pins it, as with yzma. The default model is a
  ~370 MB download on first use, and a loaded process holds about 1 GB.
- **`media/ffmpeg`** runs the `ffmpeg` and `ffprobe` binaries, 5.1 or newer, found on
  PATH or named in its `Config`: `sudo apt-get install -y ffmpeg`.

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
ix, err := search.New(emb, store, "tenant-1", "space-7", "notes", search.WithMinScore(0.3))
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
emb := langai.NewLazyEmbedder(func() (langai.Embedder, error) {
    return yzma.New(ctx, cfg)
}, spec.Dims)
```

A failed load is cached and returned by every later call; `State` reports it, e.g. for
a health endpoint.

### Image and video search

```go
emb, err := onnx.New(ctx, onnx.Config{
    ModelsDir:     "data/models",
    LibDir:        filepath.Join("data", "onnxruntime", onnx.LibVersion), // named after the pinned build
    ManagedLibDir: true,                                                  // download the pinned build into it
    Provider:      onnx.ProviderCPU,                                      // or ProviderCUDA, ProviderCoreML
})
frames, err := ffmpeg.New(ffmpeg.Config{}) // scene cuts, one frame every 5 s, at most 64, 512 px
store, err := sqlitevec.New(ctx, db, emb.Dimensions())

// an image model's vectors get a collection of their own, never a text model's
ix, err := search.NewMediaIndex(emb, store, "tenant-1", "space-7", "media",
    search.WithSampler(frames), search.WithMediaMinScore(0.6))
err = ix.IndexImage(ctx, "photo-42", file)             // JPEG, PNG, GIF, WebP, BMP, TIFF; a no-op while the bytes are unchanged
err = ix.IndexVideo(ctx, "clip-7", "/media/clip-7.mp4") // one vector per sampled frame; a no-op while the file is unchanged
hits, err := ix.SearchImage(ctx, file, 10)              // []search.MediaMatch: DocID, Score, At (the matching frame's time)
hits, err := ix.SearchVideo(ctx, "/media/query.mp4", 10) // and QueryAt, the query frame that matched
err = ix.Delete(ctx, "photo-42")
```

An image is one frame at time zero and a video the frames its `Sampler` picked, each
embedded as an image into one vector space, so an image query finds videos and a video
query finds images. A search embeds every query frame, asks the store for the closest
frames of each and scores an item by its best frame, so a video is found by the one shot
that resembles the query most; `At` says where in the video that is, and for a video
query `QueryAt` which of its frames matched. An item that is the query itself comes back
first with a score near 1, so drop your own id. `IndexFrames` and `SearchFrames` take
frames you extracted yourself, with your own content hash; more image formats register
through `image.RegisterFormat`.

- **Models.** The default, `google/siglip2-base-patch16-224` (Apache-2.0, 768
  dimensions), was trained with a text tower, so its vectors can later be matched by
  text too; with it a resized or recompressed copy scores above 0.9 and an unrelated
  picture well below. `onnx.DINOv2Model` (`facebook/dinov2-base`, Apache-2.0, 768
  dimensions) has no text side and keeps instance-level detail, so it suits
  near-duplicate matching: it puts unrelated pictures further apart. The two live in
  different spaces: one model per collection, as `tagging` keeps one model per store.
  `Open` runs any other ONNX vision model from a `ModelSpec` (tensor names, pooling,
  input size, resize and normalization).
- **Videos.** `ffmpeg.Config` sets the scene-change threshold (0.4), the shortest and
  longest gap between frames (1 s and 5 s), the cap per video (64 frames, with the
  shortest gap raised to `duration/64` so a cut-heavy video still yields about that
  many) and the size frames are scaled to (512 px). Scene detection decodes every frame;
  `KeyframesOnly` decodes only the encoder's keyframes, which sit at cuts and at a fixed
  interval anyway, and makes hours of video cheap at the price of a coarser sample.
- **Cost.** On a CPU with the default four threads the SigLIP 2 model embeds a
  1024-pixel picture in about 100 ms, eight per batch (an AMD Ryzen AI 9 HX PRO 370,
  `BenchmarkEmbedImagesReal`), and the process holds about 1.3 GB with the model
  loaded; `Config.Threads` and `BatchSize` tune it, and CUDA takes batches. Indexing a
  video costs its decode plus one embedding per frame. Embedding, the ffmpeg run and the
  hash of a video all run per call; the background job that indexes decides how many run
  at once.
- **Limits.** Pictures are embedded as decoded: a JPEG whose EXIF orientation says to
  rotate it is seen unrotated. There is no text-to-image search yet (the SigLIP 2 text
  tower is a later provider). The CUDA and CoreML paths are wired but were not run on a
  GPU in this release; the CPU path is what the real-model tests cover.

### Automatic tagging

```go
tagger, err := tagging.NewTagger(emb, store, tagging.TaggerConfig{Model: yzma.DefaultModel})
space := tagger.Scope("tenant-1", "space-7")
space.SetTag(ctx, "bread", "cooking / bread")   // create or rename
space.SetItem(ctx, "item-42", essence)          // create or edit; embeds only when text or model changed
space.TagByHand(ctx, "item-42", "bread")        // a person tagged it: teaches the tag
space.UntagByHand(ctx, "item-42", "bread")
ids, err := space.Suggest(ctx, "item-42")       // auto tags, best first; never stored as links
space.DeleteItem(ctx, "item-42"); space.DeleteTag(ctx, "bread")
space.Rebuild(ctx)                              // optional warm-up, e.g. after switching models
```

The Tagger owns what it learns from: your application reports its tags and its items
(bookmarks, notes, pages) by id and text, plus the tags people put on items by hand, and
asks for suggestions. As in search, the first argument of `Scope` is the hard isolation
boundary (a tenant) and the second a space within it; scopes never see each other's tags,
items or links. Use a tag's full path as its name (`colors / warm / red` says more than
`red`). `TaggerConfig.Model` names the embedding model, and every stored vector is kept with the
model it came from.

**Storage.** A `Store` keeps the tags, the items with their embeddings and the hand-made
links. `tagging.NewMemStore()` keeps them in memory, for tests and tiny corpora.
`tagging/sqlitestore` and `tagging/pgstore` keep them in a SQLite or Postgres database
that you open and pass to `New(ctx, db)`; your application imports the driver. Both use
plain tables and need no extension: a space has few tags, so the Tagger scores them in
Go. Each index gets a table set of its own, and a store opened with `WithGeneration`
keeps further table sets of its own, so a migration can build a second model's tags and
vectors next to the current ones before switching over. SQLite allows one writer at a
time: when several goroutines write, open it with `?_busy_timeout=5000&_txlock=immediate`
(for `go-sqlite3`: `sql.Open("sqlite3", "autotag.db?_busy_timeout=5000&_txlock=immediate")`),
so a writer waits for its turn instead of failing with "database is locked". Postgres
needs no such setup: concurrent writes queue on row locks. That locking relies on READ
COMMITTED, Postgres's default isolation level. `tagging/pgstore` begins its own
transactions at that level, but its single-statement writes run at the session's
default, so keep `default_transaction_isolation` at `read committed` for its
connections: under `repeatable read` or `serializable` they can fail with serialization
errors, which the store does not retry. `tagging/storetest` is the conformance suite for
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
  `tagging.ErrNotRebuilt`: retrying it is a safe no-op, and the next `Suggest` or
  `Rebuild` rebuilds the tag, as it does one whose rebuild was overtaken. A `Suggest`
  that cannot rebuild a tag still ranks the others and returns their ids with that
  error, leaving the tag out or scoring it with its previous vectors. A change to
  `NameMinScore`, `MemberMinScore` or `SmoothingK` leaves tags stale the same way and
  they are rebuilt lazily, without re-embedding anything.
- **Bulk writes rebuild as they go:** every hand tag rebuilds its tag at once, from all
  of its members, so importing n hand tags of one tag reads its members n times (about
  n²/2 rows), and every edit of an item rebuilds each tag it carries.
- **One model and one set of scoring settings per store:** every Tagger that shares a
  `Store` must use the same `TaggerConfig.Model`, `NameMinScore`, `MemberMinScore` and
  `SmoothingK`. A Tagger takes tags built under another model or with other settings for
  stale and rebuilds them, re-embedding their members when the model differs, so two
  Taggers that differ rebuild each other's tags at every `Suggest` and never settle.
- **Switching models:** the Tagger stores each item's text next to its embedding, so after
  you change `TaggerConfig.Model` it re-embeds the stored texts it needs, lazily, and never
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

### Generating tags

```go
emb, err := yzma.New(ctx, cfg)          // the embedding model scores the tags
llm, err := yzma.NewCompleter(ctx, cfg) // the chat model writes them
gen, err := tagging.NewGenerator(llm, emb, tagging.GeneratorConfig{})
tags, err := gen.Generate(ctx, essence, tagging.DefaultGeneratorMinScore) // []tagging.Tag{Name, Score}, best first
```

The Tagger picks among tags your application already has. The Generator needs none: it writes them
for one text, stores nothing, and returns the tags that score at least the cutoff you pass.
Hand it a text built like the essences above.

- **The model writes, the embedder scores.** The chat model (Qwen3 4B Instruct, Apache-2.0)
  is asked for up to `GeneratorConfig.Candidates` (10) tags of one to three words, in the text's
  language. A model's own confidence is poorly calibrated, so each tag is scored by the
  cosine between the embedding of the text (as a query) and of the tag (as a document):
  the roles, and so the scale, of the Tagger's `NameMinScore`. One cutoff then means the same
  on every text, and a tag the model invented falls under it.
- **The cutoff is yours.** `-1` returns every candidate. `tagging.DefaultGeneratorMinScore` (0.45)
  fits the default models: over ten golden documents in English, Spanish and German the
  model's tags scored 0.31 to 0.68, the main topics at 0.5 and up, and what fell under
  0.45 was side detail (the steam for a loaf, a side salad). That is a judgment by eye on a
  small set, and another embedder scores on another scale, so choose again for it.
- **Only the first 2 KB are read** (`GeneratorConfig.MaxBytes`), by the model and the embedder
  alike, so a tag is never scored against words the embedder did not see.
- **Cost.** The chat model is a 2.5 GB download, and a process with it and the embedding
  model loaded holds about 5 GB. A text takes 7 to 13 s with the default 4 threads on an
  AMD Ryzen AI 9 HX PRO 370 (English about 7 s, German 11), nearly all of it writing the
  tags; `yzma.Config.Threads` raises that. Decoding is greedy, so a text gets the same tags
  every time.
- **Limits.** The tags are as good as a 4B model's: their casing is the model's, a rare
  one is malformed (`FitnessimHome`), and they come back in the text's language. They
  cover what the text says; to tag against a vocabulary you keep, use the Tagger.

## Evaluating embedding models

`eval` scores an embedding model on a labelled, multilingual golden set
(`eval/testdata/golden.yaml`, English, Spanish and German): free-text retrieval
(recall@k, MRR, nDCG) over the queries in their documents' language and, separately,
over the queries that must cross a language; tagging by tag name (precision, recall and
F1 at `NameMinScore`, plus the floor each model would need); and timings. tagging's
`TestGoldenTagging` replays the same set through a `Tagger` with simulated hand-tagging
(a tag's members are up to every other document carrying it) and checks tagging by name
only and with members, how many documents stay untagged once their own tags are removed,
and two-topic pages. Both need the runtime libraries above and cache the downloads in
`~/.cache/semsearch-yzma`:

```
# the default model against the floors CI enforces
# (provider/yzma/golden_test.go and tagging/golden_test.go)
make test-models

# every candidate in eval/testdata/models.yaml, side by side
make semeval
```

Add a candidate to `eval/testdata/models.yaml` with its pooling and prefixes;
run `make semeval SEMEVAL_ONLY=<its name>` once without a digest and paste the one it
prints. The dataset format is plain YAML, so an export of your own documents can be
scored the same way (`make semeval SEMEVAL_DATASET=export.yaml`); `SEMEVAL_JSON=out.json`
keeps every metric, and the Makefile lists the other settings.

## Development

```
make test          # go tests (the pgvector ones skip without TESTDBS_ALL=1 and Docker; the ffmpeg ones without ffmpeg)
make test-models   # the real-model tests (downloads ~350 MB on first run)
make test-gentags  # the tag Generator on the real models (adds the 2.5 GB chat model; minutes on a CPU)
make test-images   # the real image-model tests: ONNX Runtime + SigLIP 2 (downloads ~380 MB on first run)
make coverage      # per-package coverage gate (>= 70%), pgvector tests included
make lint          # golangci-lint
make license-check # go-licence-detector
make verify        # run every check
```

## License

langai is licensed under the [GNU Lesser General Public License v3.0](LICENSE).

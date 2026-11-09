# langai

Libraries that use language models, run locally: semantic search over your own
documents, image and video similarity search, automatic tagging that learns from the
tags people assign by hand, and tags written for a text. Nothing leaves the process: the
models run in-process through llama.cpp and ONNX Runtime.

## Packages

| Package | What it does |
|---|---|
| `langai` | The ports: `Embedder` (text embedding models), `ImageEmbedder` (image embedding models), `VideoEmbedder` (video embedding models, one vector per `Segment` of a video), `Completer` (chat models), the `Role` a text is embedded as, and the optional `Identified`, by which a model names itself so that stores never mix two models' vectors. `Normalize` scales a vector to unit length for the providers, and `ErrClosed` is what every call on something closed wraps. `LazyEmbedder`, `LazyImageEmbedder` and `LazyCompleter` defer loading a model to its first use and release it on `Close`. |
| `search` | Turns documents into embedding vectors and answers free-text queries with the closest ones: `Index` and the markdown-aware chunker. `MediaIndex` does the same for pictures: it embeds images and videos (as sampled frames through an image model, or as whole segments through a video model) and answers an image, a video or, with a model that has a text side, a text with the items that look most like it, across both kinds. |
| `media` | The ports for decoding video: `Sampler` turns a video file into the `Frame`s that stand for it, `Cutter` measures and cuts it (for a `VideoEmbedder` that embeds in segments). |
| `media/ffmpeg` | `Sampler` and `Cutter` over the ffmpeg and ffprobe binaries: the first frame, every scene cut and one frame per interval, scaled down, capped per video, or the keyframes only, for long videos; stream-copy cuts for the segment embedder. |
| `tagging` | Tags for a text. `Tagger` suggests tags among the ones an application already has and learns from the tags people put on items by hand: each tag's vector blends its name with the items tagged with it, and each tag is judged against its own cutoff. The application provides its tags and hand-made links through a `Catalog`; the Tagger keeps only an embedding cache (`MemCache`, `tagging/sqlitecache`, `tagging/pgcache`) and an in-memory index of tag vectors. `Generator` writes tags with no list to start from: a chat model proposes them and the embedding model scores each against the text, so one cutoff means the same on every text. |
| `vector` | The `Store` interface for search vectors: documents in collections located by a `Coord` (index, partition, collection), batch writes, listing, and one exact `Search` that returns the best chunk of each of the best documents. `Validate` and the sentinel errors (`ErrDimensions`, `ErrInvalidVector`, `ErrModelMismatch`), and `MemStore` (exact, in memory, for tests and evaluation). |
| `vector/vectortest` | The conformance suite every `Store` passes. |
| `vector/sqlitevec` | `Store` on SQLite with the sqlite-vec extension (CGO). |
| `vector/pgvec` | `Store` on Postgres with pgvector. |
| `provider/yzma` | `Embedder` that runs a GGUF sentence-embedding model, `Completer` that runs a GGUF chat model (the one the `Generator` writes tags with), and `Vision`, which runs Qwen3-VL-Embedding as a text, image and video embedder in one space, through llama.cpp, loaded at runtime by [yzma](https://github.com/hybridgroup/yzma) (purego + libffi, no CGO). Each opens two ways: `New`, `NewCompleter` and `NewVision` from a `Config` naming a pinned model (`Models`, `CompleterModels`, `VisionModels`) with the `Runtime` embedded, or `Open`, `OpenCompleter` and `OpenVision` from a spec (`LookupSpec`, `LookupCompleterSpec`, `LookupVisionSpec`, or your own) and a `Runtime`; a `Vision` also takes its `VisionOptions`. Downloads and verifies the pinned llama.cpp build and models on first use, and picks the build and the GPU offload from the hardware (`Runtime.Accel`). |
| `provider/onnx` | `ImageEmbedder` that runs an ONNX vision model through [ONNX Runtime](https://onnxruntime.ai) (CGO, via [onnxruntime_go](https://github.com/yalue/onnxruntime_go)): SigLIP 2 base by default, DINOv2 base as the other pinned model (`Models`, `LookupSpec`), any other through a `ModelSpec`; `New` and `Open` have yzma's shape. CPU by default, CUDA or CoreML by configuration. Downloads and verifies the pinned runtime build and model on first use. |
| `provider/phash` | `ImageEmbedder` without a model: a 64- or 256-bit perceptual hash (pHash or dHash) as a unit vector whose cosine is `1 - 2·hamming/bits`, for finding exact and near-exact duplicates (re-encoded, resized, lightly edited copies) with the same `MediaIndex` and stores. Pure Go, nothing to download, a few milliseconds per picture. `Hash` gives the bits themselves for grouping without a store. |
| `zarf/eval` | Measures an `Embedder` on a labelled, multilingual golden set: retrieval, tagging, timings. `make semeval` compares embedding models side by side on it. |
| `langaitest` | Fakes of the ports, for tests. `FakeEmbedder`, `FakeImageEmbedder` and `FakeVideoEmbedder` are deterministic and each names a fixed model (`FakeEmbedderModel`, `FakeImageEmbedderModel`, `FakeVideoEmbedderModel`); a `FakeVideoEmbedder`'s segments are pictures embedded in `FakeImageEmbedder`'s space, listed per file name or derived from it; `FakeCompleter` is a scripted chat model: canned replies in order or a function of the prompt, an error on demand, and a record of the prompts it received. All four fail with the error of a context that is done and with their `Err` field when it is set, and count their calls; `FakeEmbedder.Fail` fails the calls that hold a text of your choosing. |

Every package stands alone and you only compile the dependencies of the packages you
import: CGO and sqlite-vec come in only with `vector/sqlitevec`, pgvector only with
`vector/pgvec`, llama.cpp only with `provider/yzma`, CGO and ONNX Runtime only with
`provider/onnx`, and the ffmpeg binary is only run by `media/ffmpeg`. `provider/phash`
is pure Go and brings in nothing.

## Layout

The packages form a tree, and the tree is also the plan: packages marked *planned* do
not exist yet (see TODO.md).

```
langai/                 the shared ports: Embedder, ImageEmbedder, VideoEmbedder, Completer,
│                       Role, Identified, ErrClosed; the lazy loaders
│
├── search/             text → similar documents; image or video → similar images and videos
├── tagging/            text → tags from your corpus (Tagger), text → new tags (Generator);
│   │                   Tagger takes an Embedder, Generator a Completer and an Embedder
│   ├── sqlitecache/    the Tagger's EmbeddingCache on SQLite
│   ├── pgcache/        the Tagger's EmbeddingCache on Postgres
│   ├── cachetest/      conformance suite for an EmbeddingCache
│   └── internal/       sqlcache: the SQL the two caches share
├── mediatag/           planned: image or video → tags
├── caption/            planned: image or video → description
│
├── provider/           model runtimes; each implements some of the ports
│   ├── yzma/           llama.cpp in process: Embedder (New), Completer (NewCompleter),
│   │                   Vision (NewVision): text, images and videos in one space
│   ├── onnx/           ONNX Runtime in process (CGO): SigLIP 2 and DINOv2 image embeddings
│   ├── phash/          no model: perceptual hashes (pHash, dHash) as image embeddings, for duplicates
│   ├── kronk/          planned: llama.cpp via kronk: embeddings, chat, vision, rerank
│   ├── openai/         planned: OpenAI-compatible HTTP, remote APIs or a kronk server
│   └── cache/          planned: wraps any provider, caches by content hash
├── vector/             Store for search vectors, in-memory MemStore
│   ├── sqlitevec/      SQLite with sqlite-vec (CGO)
│   ├── pgvec/          Postgres with pgvector
│   ├── vectortest/     conformance suite for a Store
│   ├── internal/       sqlstore: the SQL the two stores share
│   └── qdrant/         planned, if a dedicated vector database is needed
├── media/              video decoding: the Sampler and Cutter ports, and Frame
│   └── ffmpeg/         video → frames, and cuts, via the ffmpeg binary
│
├── langaitest/         fakes of the ports, for tests
├── internal/           tablename: SQL table names; fetch: verified, resumable downloads and
│                       locked library installs for the providers; hw: which accelerators
│                       the machine can run; imageutil: the picture scaling the providers
│                       share; planned: rank, score roll-up
│
└── zarf/               tooling around the library, not part of it
    ├── eval/           golden-set evaluation of embedding models
    └── examples/       the example service: a page per feature, media search and auto tagging
```

A capability never names a model: it calls a port, and the app hands it a provider that
implements that port. For tagging, with the default llama.cpp runtime:

```
                       port (langai)           provider (your choice)
tagging.Tagger    ──▶  Embedder.Embed     ◀──  yzma.New            embedding model (GGUF)
tagging.Generator ──▶  Completer.Complete ◀──  yzma.NewCompleter   chat model (GGUF): writes candidate tags
                  ──▶  Embedder.Embed     ◀──  yzma.New            embedding model: scores them
```

The Tagger calls only the embedding model: it embeds the text and the tag names, and
reads the tags and the hand-made links from your `Catalog`. The Generator calls both: the
chat model proposes tag names, and the embedding model scores each against the text. The
chat and embedding models are separate instances, and one embedding instance can serve
both; nothing in `tagging` loads or picks a model.

- **The root is the front door:** the shared ports and the capabilities (`search`,
  `tagging`, `mediatag`, `caption`) are what you import.
- **Each group holds one kind of implementation:** model runtimes in `provider/`, vector
  stores in `vector/`, video decoding in `media/`. A new backend is a new folder in its
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
  so import it (blank if need be) before opening the first connection. Open a file
  database with a busy timeout, so concurrent writers queue instead of failing with
  "database is locked", and in WAL mode, so readers do not wait for them:
  `search.db?_busy_timeout=5000&_journal_mode=WAL` with mattn/go-sqlite3. Every write
  transaction of the store starts with a write, so `_txlock=immediate` is not needed.
- **`vector/pgvec`** needs a Postgres with the pgvector extension, or a user allowed
  to run `CREATE EXTENSION vector`. Only writes create tables, so a role that may only
  read can search what another role indexed.
- **`provider/yzma`** needs `libffi8` wherever the binary runs: the Go bindings open it
  in a package `init()` and panic without it, so a binary that imports the package needs
  it even when it never embeds anything. The llama.cpp libraries also need `libgomp1`,
  `libstdc++6` and glibc 2.34 or newer:
  `sudo apt-get install -y libffi8 libgomp1 libstdc++6`.
- **The chat model** (`yzma.NewCompleter`, for the tag `Generator`) is a 2.5 GB download on first use,
  and a process with it and the embedding model loaded holds about 5 GB of RAM.
- **GPUs with `provider/yzma`.** `Runtime.Accel` left empty picks from the hardware: a
  managed `LibDir` gets the pinned llama.cpp build that carries the backend the machine
  has (CUDA 13.4 on linux/amd64 with the CUDA runtime installed, Vulkan on linux/amd64
  and linux/arm64, Metal on macOS, the CPU build otherwise; `internal/hw` looks for the
  driver and, for CUDA, for `libcudart.so.13` and `libcublas.so.13` on the loader's
  path, so an NVIDIA driver without the CUDA 13 runtime gets the Vulkan build), the
  model is offloaded to the GPU that registers, and the load log line says which; a
  GPU build that registers no GPU is logged as a warning, since the models then run on
  the CPU. `AccelCPU` stays on the CPU, `AccelVulkan`, `AccelCUDA` and `AccelMetal`
  require their backend. An operator-supplied `LibDir` is used as it is: whatever
  backends it holds register. Vulkan was run on an AMD iGPU for this release; the CUDA
  build is pinned but was not run.
- **A managed `LibDir`** (`ManagedLibDir: true`, in both providers) belongs to the
  provider, so give it a directory of its own. Only a missing or empty directory, or one
  an earlier install left its record in, is filled or replaced; anything else is an
  error, never a deletion. Installs take a lock file next to it (`<LibDir>.lock`), so
  goroutines and processes that start together download once and none loads a
  half-installed library. Libraries a process has loaded are never replaced under it:
  asking by name for another build in the same process (`AccelCPU` first and
  `AccelVulkan` later, `ProviderCPU` and then `ProviderCUDA`) is an error, so open the
  most capable one first; `Accel` left empty keeps what is loaded.
- **llama.cpp's log** goes to `Runtime.Logger`: its warnings and errors, and nothing of
  its progress. A call whose context ends mid-evaluation stops llama.cpp, which then
  logs errors such as `failed to decode, ret = 2`; those come at Debug with
  `cancelled=true`, since the caller already has its context's error, while a decode
  that fails for any other reason stays an error.
- **The vision model** (`yzma.NewVision`) is a 2.7 GB download on first use (the Q8_0
  GGUF and its f16 projector) and needs `ffmpeg` and `ffprobe` on PATH for videos; a
  loaded process holds about 3 GB.
- **Keep yzma where langai pins it.** The llama.cpp build `provider/yzma` downloads
  (`yzma.LibTag`) matches the yzma version in this module's `go.mod`, whose Go
  bindings mirror exactly that build's ABI. A module that imports `provider/yzma` must
  not require another yzma version: minimal version selection would hand these
  bindings to libraries they do not match.
- **`provider/onnx`** is CGO and needs a C compiler to build (no ONNX Runtime headers:
  the bindings carry their own). At runtime it needs the ONNX Runtime shared library:
  a managed `LibDir` downloads the pinned CPU build (`onnx.LibVersion`, ~11 MB) on
  linux/amd64, linux/arm64 and darwin/arm64. For **CUDA** set `Runtime.Provider` to
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
cfg := yzma.Config{
    ModelsDir: "data/models",
    Runtime: yzma.Runtime{
        LibDir:        filepath.Join("data", "llama-cpp", yzma.LibTag), // named after the pinned build
        ManagedLibDir: true,                                            // download the pinned build into it
    },
}
emb, err := yzma.New(ctx, cfg) // cfg.Model "" is yzma.DefaultModel; yzma.Models() lists the others
store, err := sqlitevec.New(ctx, db, emb.Dimensions())

// index (e.g. a tenant) is the hard isolation boundary; partition and collection
// locate the documents within it
ix, err := search.New(emb, store, "tenant-1", "space-7", "notes", search.WithMinScore(0.3))
err = ix.Index(ctx, "note-42", markdownText) // a no-op while the text is unchanged
res, err := ix.IndexBatch(ctx, docs)         // many at once: res.Indexed, res.Unchanged, res.Failed
hits, err := ix.Search(ctx, "how long to boil spaghetti", 10) // []search.Match: DocID, Score, Seq and Text of the best chunk
more, err := ix.SearchIndexed(ctx, "note-42", 10)            // "more like this" with the stored chunks: no model runs
for d, err := range ix.Docs(ctx) { /* d.ID, d.Hash: what is indexed, by id */ }
err = ix.Delete(ctx, "note-7", "note-9")                     // e.g. what was deleted while the app was offline
```

`Search` and `SearchIndexed` return a `search.Match` per document: its best chunk, by
position and text (a snippet), and its score. `SearchIndexed` searches with every stored
chunk of the document and scores each other document by its best chunk against any of
them; the document itself is never among the hits, and an id with nothing stored is
`search.ErrNotIndexed`. Every error the package returns starts with `search:`, and an
empty document id anywhere is `search.ErrEmptyID`, refused before a model or the store
is asked anything.

`IndexBatch` indexes many documents (`[]search.Doc{ID, Text}`) the way `Index` does
one, in bounded batches: one hash read for all of them, at most 32 chunk texts per
embedder call, one store write per such batch. Unchanged documents cost only their share
of the hash read, so it suits a full re-index at startup. The `BatchResult` lists what
was `Indexed`, what was `Unchanged` and what `Failed` for a reason of its own (a chunk
that embeds to the zero vector), by id; such a failure does not stop the others. An
embedder or store error does: it is returned, `Indexed` is exactly what was stored before
it, and the same call again picks up where it stopped.

Text is cleaned before it is chunked and hashed: invalid UTF-8 becomes U+FFFD and NUL
bytes are dropped, which a Postgres `TEXT` column refuses and SQLite takes, so both
backends store the same. A blank query finds nothing, and so does one that embeds to the
zero vector, which has no direction to rank by; a document chunk that embeds to it fails
that document, by name.

"Unchanged" means the hash stored with the document matches, and that hash covers
everything that shapes its vectors: the text, the chunker's settings and the embedding
model (its name, when it implements `langai.Identified`, and its dimensions). A chunker
describes its settings through the optional `search.Fingerprinter`; `MarkdownChunker`
does (`markdown/1 max=1200`), and one without it is described by its Go type name, so
swapping chunkers re-indexes and changing a custom one's settings does not. The
`MediaIndex` hashes cover the model in the same way, and a video's also the `Sampler`'s or
the `VideoEmbedder`'s `Fingerprint`, when it has one.

`MarkdownChunker`, the default (`WithChunker` replaces it), cuts a document at its
headings and packs its paragraphs into chunks of at most `MaxChars` bytes (1200), each
prefixed with its heading path ("Title › Section"), which counts towards that size and is
cut to a quarter of it. CRLF text and blank lines holding spaces chunk like plain ones,
and a paragraph too long for one chunk is split between lines, so code and lists keep
their shape, and only a line too long for one chunk between words.

Every `Store` backend keeps one physical table set per index, so one index's vectors are
never queryable from another's. Within an index a search sees exactly one collection,
the one at its partition and collection: an empty partition or collection is a value
like any other, not a wildcard. A `vector.Query` opens a coordinate only when asked
(`AllPartitions`, `AllCollections`), and can keep or leave out given document ids.
`Store.Search` ranks documents, not chunks: it returns the best chunk of each of the
best documents, so one long document or one many-framed video never crowds out the
rest, in score order, then by document id, then by chunk. `DeleteIndex` removes an
index with everything in it, e.g. when a tenant leaves.

Vectors of different models never mix. A store refuses a vector of the wrong length
(`vector.ErrDimensions`) and a zero, NaN or infinite one (`vector.ErrInvalidVector`); a
SQL store opened with other dimensions than an index's tables hold refuses that index.
When the embedder names its model (`langai.Identified`), `search.New` and
`search.NewMediaIndex` pass that name on: the first write to a collection records it,
and a write or search with another model fails with `vector.ErrModelMismatch`. An
embedder that names no model is not checked.

A store opened with `WithGeneration` keeps further table sets of its own, so a migration
can build a second model's vectors next to the current ones before switching over; the
new generation's store then retires the old one with `DropGeneration`. The SQL stores
refuse the tables of the schema before this version with an error that says so: delete
the index (`DeleteIndex`) or build a new generation, and index again.

The first `yzma.New` downloads the llama.cpp libraries (~18 MB) and the default model,
`google/embeddinggemma-300m` (~329 MB, 768 dimensions, multilingual); both are pinned by
digest. `ctx` bounds the downloads: with no deadline of its own it gets a default one
(15 minutes here, an hour for the chat and vision models, 30 minutes in `provider/onnx`),
and a deadline you set stands, shorter or longer. A download that is cut short keeps what
it got in `<file>.partial` and the next call asks the server for the rest; the whole is
verified against the digest either way. A model file that is not the pinned size (a
crash, a full disk) is fetched again instead of being loaded. To keep all that off your
startup path, wrap the loader in a `LazyEmbedder`, which reports its model and
dimensions from the model's spec without loading anything:

```go
spec, err := yzma.LookupSpec(cfg.Model, cfg.ModelsDir) // the spec yzma.New(ctx, cfg) loads
emb := langai.NewLazyEmbedder(spec.ModelID(), spec.Dims, func(ctx context.Context) (langai.Embedder, error) {
    return yzma.New(ctx, cfg) // ctx is the LazyEmbedder's own: only Close cancels it
})
defer emb.Close()

err = emb.Load(ctx)        // optional: load now, e.g. in the background at startup
state, err := emb.State() // not started, loading, loaded, failed or closed, for a health endpoint
```

`LazyImageEmbedder` and `LazyCompleter` do the same for an image model and a chat model.
All three behave alike:

- **One load, shared.** The first call (or `Load`) starts it on a goroutine of its own
  and every caller waits for it under its own context: a request that times out
  returns, and the load goes on for the others. The load's context belongs to the lazy
  value and is cancelled by `Close` alone.
- **A failed load is kept.** The error is returned by every later call and nothing
  loads again, so a model that cannot be loaded costs one attempt, not one per request.
  A loader that panics (`*langai.LoadPanicError`, with the stack) or returns nil fails
  the same way. To try again once the cause is fixed, build a new lazy value
  (`langai.NewLazyEmbedder(...)` again), use it in place of the failed one and `Close`
  that one: a lazy value never retries on its own.
- **The model is checked.** The identity and dimensions given at construction are what
  the stores record before anything loads (an empty identity is unknown and unchecked).
  A loaded model with other dimensions, or whose `Model()` is another string, fails the
  load and is closed instead of mixing two vector spaces. So pass the identity the
  loaded model will report, which each provider computes from its spec without loading
  (`yzma.ModelSpec.ModelID()`, `yzma.VisionSpec.ModelID(opts)`,
  `onnx.ModelSpec.ModelID()`), not the spec's bare `Name`.
- **`Close` releases the model.** It cancels a load in progress and waits for it, closes
  the loaded model when it has a `Close`, and makes every waiting and later call fail.

`Model()` on `yzma.Embedder`, `yzma.Vision` and `onnx.Embedder` is the identity of the
vectors they produce, the string to store next to them: the model's name followed by
what else decides a vector (the pooling and the prefixes for a text model; the
instruction, the image token bounds and the video sampling for `Vision`; the tensor and
the picture preparation for an ONNX model). It is the same on every run and changes
whenever a vector would, so vectors stored under another identity are stale; the
specs' `ModelID` gives the same string before the model loads. A call on
a closed provider, lazy loader or `Tagger` returns an error that wraps `langai.ErrClosed`.

### Image and video search

```go
emb, err := onnx.New(ctx, onnx.Config{
    ModelsDir: "data/models", // Model "" is onnx.DefaultModel; onnx.Models() lists the others
    Runtime: onnx.Runtime{
        LibDir:        filepath.Join("data", "onnxruntime", onnx.LibVersion), // named after the pinned build
        ManagedLibDir: true,                                                  // download the pinned build into it
        Provider:      onnx.ProviderCPU,                                      // or ProviderCUDA, ProviderCoreML
    },
})
frames, err := ffmpeg.New(ffmpeg.Config{}) // scene cuts, one frame every 5 s, at most 64, 512 px
store, err := sqlitevec.New(ctx, db, emb.Dimensions())

// an image model's vectors get a collection of their own, never a text model's
ix, err := search.NewMediaIndex(emb, store, "tenant-1", "space-7", "media",
    search.WithSampler(frames), search.WithMediaMinScore(0.6))
err = ix.IndexImage(ctx, "photo-42", file)             // JPEG, PNG, GIF, WebP, BMP, TIFF; a no-op while the bytes are unchanged
err = ix.IndexVideo(ctx, "clip-7", "/media/clip-7.mp4") // one vector per sampled frame; a no-op while the file is unchanged
hits, err := ix.SearchImage(ctx, file, 10)              // []search.MediaMatch: DocID, Score, At (the matching frame's time)
hits, err = ix.SearchVideo(ctx, "/media/query.mp4", 10)  // and QueryAt, the query frame that matched
hits, err = ix.SearchIndexed(ctx, "clip-7", 10)          // "more like this" with the stored vectors: no model runs
err = ix.Delete(ctx, "photo-42", "clip-7")               // Docs lists what is indexed, as for Index
```

An image is one frame at time zero and a video the frames its `Sampler` picked, each
embedded as an image into one vector space, so an image query finds videos and a video
query finds images. A search embeds every query frame, asks the store for the best
frame of each of the closest items and scores an item by its best frame, so a video is
found by the one shot that resembles the query most, and its other frames take no
places from other items; `At` says where in the video that is, and for a video
query `QueryAt` which of its frames matched. `SearchIndexed` leaves the item it starts
from out of its hits; a search by a file that is also indexed finds that item first, with
a score near 1. `IndexFrames` and `SearchFrames` take
frames you extracted yourself, with your own content hash; more image formats register
through `image.RegisterFormat`.

`IndexImage` and `SearchImage` read at most 64 MiB from their reader
(`WithMaxImageBytes`) and refuse a longer stream with `search.ErrTooLarge`, and refuse a
picture of more than 256 megapixels before decoding it. A video the `Sampler` takes no
frames from, or the `VideoEmbedder` no segments, is `search.ErrNoFrames` and nothing is
stored, so the next run tries it again rather than keep an item that is never found.

- **Models.** The default, `google/siglip2-base-patch16-224` (Apache-2.0, 768
  dimensions), was trained with a text tower, so its vectors can later be matched by
  text too; with it a resized or recompressed copy scores above 0.9 and an unrelated
  picture well below. `onnx.DINOv2Model` (`facebook/dinov2-base`, Apache-2.0, 768
  dimensions) has no text side and keeps instance-level detail, so it suits
  near-duplicate matching: it puts unrelated pictures further apart. The two live in
  different spaces: one model per collection, as `tagging` keys every embedding by model.
  `Open` runs any other ONNX vision model from a `ModelSpec` (tensor names, pooling,
  input size, resize and normalization). Finding the copies of a picture (the same one
  re-encoded, resized or lightly edited) needs no model at all: see
  [Duplicates without a model](#duplicates-without-a-model).
- **Videos.** `ffmpeg.Config` sets the scene-change threshold (0.4), the shortest and
  longest gap between frames (1 s and 5 s), the cap per video (64 frames, with the
  shortest gap raised to `duration/64` so a cut-heavy video still yields about that
  many) and the size frames are scaled to (512 px). Scene detection decodes every frame;
  `KeyframesOnly` decodes only the encoder's keyframes, which sit at cuts and at a fixed
  interval anyway, and makes hours of video cheap at the price of a coarser sample.
  The `Sampler`'s `Fingerprint` covers every one of these settings, so changing one
  indexes the videos again. A path is always a local file, relative or absolute: a colon,
  a leading dash or a `%d` in a name is never read as a protocol, an option or an image
  sequence, and ffmpeg and ffprobe open their input with only the `file` protocol on
  their whitelist, so a crafted playlist or concat list that names an `http://` (or
  `rtp://`) input fails at once instead of reaching the network. A frame's time comes
  from the line ffmpeg's own filter logs for it, by the frame's number, so a title or a
  chapter name that spells such a line moves nothing. Frames are read
  from ffmpeg one at a time, each with pixels of its own, so memory holds the frames
  taken (at most 64 × 512 × 512 × 4 bytes, 67 MB, with the defaults) and a frame you
  keep does not keep the others alive. A video no frame comes out of is an error, and
  cancelling the context kills ffmpeg and ffprobe.
- **Cost.** On a CPU with the default four threads the SigLIP 2 model embeds a
  1024-pixel picture in about 100 ms, eight per batch (an AMD Ryzen AI 9 HX PRO 370,
  `BenchmarkEmbedImagesReal`), and the process holds about 1.3 GB with the model
  loaded; `Runtime.Threads` and `BatchSize` tune it, and CUDA takes batches. Indexing a
  video costs its decode plus one embedding per frame. Embedding, the ffmpeg run and the
  hash of a video all run per call; the background job that indexes decides how many run
  at once.
- **Limits.** A JPEG is turned upright by its EXIF orientation (all eight, read by a
  small parser of its own that takes a broken block for none) before it is embedded;
  other formats are embedded as decoded. Transparent parts are composed on white, here
  and in the vision model below, so a logo on a transparent background is the same
  picture to every model. A picture of an extreme shape (one pixel wide, twenty thousand
  tall) costs no more memory to prepare than any other. The SigLIP 2 text tower is not
  wired, so text queries need the vision model below. The CUDA and CoreML paths are wired
  but were not run on a GPU in this release; the CPU path is what the real-model tests
  cover.

### Duplicates without a model

```go
dup, err := phash.New(phash.Config{})                  // the 64-bit pHash; or Algorithm: phash.AlgorithmDHash, Bits: 256
store, err := sqlitevec.New(ctx, db, dup.Dimensions()) // 64 dimensions, one per bit
ix, err := search.NewMediaIndex(dup, store, "tenant-1", "space-7", "duplicates",
    search.WithMediaMinScore(0.875))                   // at most 4 of the 64 bits differ
err = ix.IndexImage(ctx, "photo-42", file)
hits, err := ix.SearchIndexed(ctx, "photo-42", 10)     // its copies; Score is 1 - 2·hamming/64

h, err := dup.Hash(img)          // or without a store: the bits themselves
same := h == other               // comparable, so also a map key
near := h.Distance(other) <= 4   // Hamming distance; h.String() is the hex
```

`provider/phash` is an `ImageEmbedder` that runs no model: it reduces a picture to a
perceptual hash and returns the bits as a unit vector of ±1/√bits entries, so the cosine
of two pictures is exactly `1 - 2·hamming/bits` and `MediaIndex` and every store work on
it as on a model's vectors. It is pure Go, downloads nothing and hashes a 1920×1080
picture in about 2 ms (5 ms for a decoded JPEG; `BenchmarkHash`, one core of an AMD Ryzen
AI 9 HX PRO 370), so a de-duplication job does not pay for a model.

- **What it finds.** The same picture again: saved as a JPEG, scaled up or down,
  slightly brightened, or with a small mark stamped on it. Not a crop, a rotated or
  mirrored copy, or a picture that only looks similar; those need a model
  (`onnx.DINOv2Model` for near-duplicates, SigLIP 2 for "looks like"). A picture with
  almost no structure has almost nothing to hash: every flat picture has the same hash.
- **Algorithms.** `AlgorithmPHash` (the default) thresholds the 8×8 lowest frequencies of
  the DCT of a 32×32 grey grid at their median; `AlgorithmDHash` compares neighbouring
  cells of a 9×8 grid. pHash keeps unrelated pictures furthest apart; dHash moves least
  under a local edit. `Bits: 256` widens the gap between duplicates and the rest.
- **Scores.** With the 64-bit pHash a re-encoded, resized or brightened copy stayed
  within 2 bits on the package's test pictures and one with a mark within 10, while the
  closest of 79,800 unrelated pairs was 14 bits apart: a floor of 0.875 (4 bits) keeps
  the plain copies, 0.6875 (10 bits) the marked ones too. The package doc has the table
  for all four hashes.
- **One hash per collection.** `Model()` names the algorithm and size (`phash-64`,
  `dhash-64`, `phash-256`, `dhash-256`), so a store refuses to mix two of them, or a
  hash with a model's vectors.
- **Deterministic.** Integer arithmetic only: the same pixels give the same bits on
  every OS and architecture. Transparency is composed on white, and any `image.Image`
  is accepted (paletted, 16-bit, CMYK, YCbCr) up to 256 megapixels.

### Text, images and videos in one space

```go
v, err := yzma.NewVision(ctx, cfg, yzma.VisionOptions{ // cfg as for yzma.New: the Runtime is shared
    Cutter: frames, // the ffmpeg Sampler cuts videos into segments too
})
ix, err := search.NewMediaIndex(v, store, "tenant-1", "space-7", "media",
    search.WithVideoEmbedder(v), search.WithTextEmbedder(v))
err = ix.IndexImage(ctx, "photo-42", file)              // as above
err = ix.IndexVideo(ctx, "clip-7", "/media/clip-7.mp4") // one vector per 20 s segment
hits, err := ix.SearchText(ctx, "a walrus on a beach", 10) // []search.MediaMatch, At = the segment's start
vecs, err := v.Embed(ctx, langai.RoleQuery, []string{"..."})  // the same object is the text Embedder
```

`yzma.Vision` runs `Qwen/Qwen3-VL-Embedding-2B` (Apache-2.0, 2048 dimensions) through
llama.cpp and its libmtmd: the model's chat template wraps every input in an
instruction (`Represent the user's input.` unless `VisionOptions.Instruction` says
otherwise), the projector turns pictures and frames into tokens, and the vector is the
last token's hidden state. Texts, images and videos land in one space, so a text finds
pictures and clips, and an image finds the stretch of a video it appears in.

- **Videos as segments.** A video goes through the model in segments of
  `SegmentLength` (20 s), each sampled at `FPS` (1) and embedded as one input: about 190
  tokens per second of video, so a segment is a few thousand tokens and a feature film
  never overflows the context (`ContextTokens`, 16384). The `Cutter` (the ffmpeg
  `Sampler`) cuts by stream copy, so a segment starts at the last keyframe before its
  time. A cut is written under a name of its own and renamed into place when it is
  whole: one that fails or is cancelled leaves nothing, and never removes a file that
  was already there, and a cut onto its own input is refused. `MaxSegments` caps
  the work per video: 720 segments unless you set it (`DefaultMaxSegments`, four hours
  at 20 s), and what lies past the cap is not embedded. A segment is one vector: the
  model sees the whole stretch, which is what makes a clip match a description, and
  what loses the "matched at 4.00 s" precision of the frame route.
- **A video's length is read, not believed.** The length is the video stream's (a
  sound track that runs on after the picture does not add segments), and a header can
  still be wrong. A length no video has (none, or more than a week, `media.MaxDuration`)
  counts as unknown, and a walk through a video that ends before its header says stops
  at the `Cutter`'s `media.ErrPastEnd` instead of embedding the last pictures again.
  A file without a length (a recording that was never finalized, a stream dump) is
  first copied into a Matroska file in the temp directory, at most `MaxSegments`
  segments of it, so that it has one, and is then cut like any other.
- **Disk.** Segment files go into a directory of the call's own inside
  `VisionOptions.TempDir` (the system's temp directory unless set), one at a time: each
  is removed once it is embedded. A video takes the room of one segment there; one
  without a length also that of its copy, up to `MaxSegments` segments long, so point
  `TempDir` at a disk when the system's is kept in memory.
- **Segment containers.** ffmpeg picks a segment's container by its file extension, so
  a segment keeps the video's extension only where that names a container known to take
  such a file's codecs by stream copy: `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.ts`,
  the MPEG program streams and raw H.264/HEVC streams (`.m4v` is cut into `.mp4`).
  Anything else is cut into `.mkv`: a name without an extension, one ffmpeg has no
  container for (`.divx`, `.rmvb`), a backup or a half-renamed download (`.mp4.part`).
- **Texts are plain text.** The chat template is laid out once, around stand-ins, and
  the instruction and every input are tokenized as parts of their own: a text that
  spells one of the model's control tokens (`<|im_end|>`), libmtmd's media marker or a
  NUL byte is read as those characters and nothing else. A text that does not fit the
  context next to the instruction is cut at its end, as the text `Embedder` cuts,
  instead of failing the batch it came in; a picture or a video segment that does not
  fit is still an error (`ImageMaxTokens`, `SegmentLength`).
- **Cancelling.** `ctx` is passed to llama.cpp as an abort callback in all three
  (`Embedder`, `Completer`, `Vision`): on the CPU a cancelled call stops within one
  step of the computation and returns the context's error, and the model stays usable.
  A GPU backend does not ask the callback, so there the batch that is running (up to
  2048 tokens for `Vision`, 512 for the `Completer`) finishes first.
- **Scores.** Copies of a photo score 0.97 to 0.98, a pug against another pug about
  0.8, unrelated photos about 0.1, a short description against the photo it describes
  0.6 to 0.7: a floor of 0.4 keeps related things and drops the rest.
- **Cost.** A 800×600 photo is about 480 tokens, a 1080×1350 one about 1,400;
  `ImageMaxTokens` lowers that (the projector scales the picture down). On this
  module's test laptop (AMD Ryzen AI 9 HX PRO 370 with its Radeon 890M iGPU through
  Vulkan) a photo takes 1.5 s and an 8 s clip 4.5 s; on the CPU alone, 7 s and 22 s. A
  text takes 0.1 s. The model and projector are 2.7 GB on disk and the process holds
  about 3 GB.
- **One model per collection.** A `MediaIndex` built on the vision model must not
  share a collection with one built on SigLIP 2 frames: different vectors for the same
  files.

### Tagging flows

1. **Suggest** (existing tags)
   1. The app calls `Suggest` with the text and the space's revision.
   2. The library embeds the text, using the cache with the text's hash as the key.
   3. The library keeps an in-memory index of tag vectors. When the revision changes, it
      rebuilds the changed tags from the `Catalog` and the cached embeddings. A member
      text the embedder cannot take is left out, logged and listed by `Incomplete`; an
      embedder that is down fails the rebuild instead.
   4. A tag vector is its name blended with the average of its members' vectors, and
      each tag has its own cutoff.
   5. The library compares the text's vector with every tag vector by cosine.
   6. Tags that reach their cutoff are picked, and on one path only the better fit is
      kept.
   7. The library returns the picked tag ids.
2. **Scores** (the reasons behind Suggest)
   1. The app calls `Scores` with the text and the space's revision.
   2. It runs the same steps as `Suggest`.
   3. It returns every tag with its score, its cutoff and the tag that shadowed it, if
      any. The ones not picked are included.
3. **Warm** (pre-compute)
   1. The app calls `Warm` with the revision and some texts, for example on item save or
      after a model switch.
   2. The library rebuilds the space if the revision changed.
   3. It embeds and caches any texts the cache doesn't have yet, a batch at a time, so a
      cancelled Warm keeps its progress.
4. **Generate** (new tags)
   1. The app calls `Generate` with the text and a minimum score.
   2. The library cuts the text to `MaxBytes` (64 KB by default).
   3. A text of up to `ChunkBytes` (2 KB by default) goes to an LLM in one call. A longer
      one is read in windows of about 6 KB, one LLM call each, with a short first line (the
      title) repeated at the head of every window after the first, and the names of all
      the windows are merged.
   4. The LLM writes up to `Candidates` tags (10 by default) per call, either as short
      words or as tree paths, optionally extending the app's existing paths. A call that
      fails or replies with nothing is made once more; a window that fails twice is left
      out and reported with `ErrPartial`. A closed model, or one that fails on the first
      two windows, is not asked about the rest.
   5. The library embeds the text (in 2 KB chunks when it is long) and each candidate
      tag. Nothing is cached.
   6. Each candidate is scored by the cosine between its vector and the text's vector;
      for a long text, by its best cosine over the chunks.
   7. Candidates below the minimum score are dropped, and a long text's are capped at
      `Candidates`.
   8. The library returns the remaining tags with their scores, best first.
5. **Cache maintenance**
   1. The app calls `Tagger.Prune` on a schedule, with a cutoff and the spaces still in
      use.
   2. The library reads each space's `Catalog` and marks the embeddings of its tag names
      and members as used now. Nothing is embedded.
   3. It deletes the embeddings not used since the cutoff, across all tenants.
   4. The app calls `DeleteIndex` on the cache to delete all embeddings of one tenant,
      for example when the tenant is removed.

There's no "learn" call. The Tagger learns only when the app changes the revision.

### Automatic tagging

```go
// catalog is your tagging.Catalog: Tags lists a space's tags (id, name, revision),
// Members the texts of the items tagged with some of them by hand
cache, err := sqlitecache.New(ctx, db)
tagger, err := tagging.NewTagger(emb, catalog, cache, tagging.TaggerConfig{Model: emb.Model(), Logger: logger})
defer tagger.Close() // cancels and waits for what is in flight; later calls wrap langai.ErrClosed
space := tagger.Scope(tagging.Space{Index: "tenant-1", Partition: "space-7"})
ids, err := space.Suggest(ctx, rev, essence)    // auto tags, best first; never stored as links
scores, err := space.Scores(ctx, rev, essence)  // every tag's score and cutoff: why a tag was or was not picked
err = space.Warm(ctx, rev, essences...)         // optional warm-up, e.g. after switching models
```

Your application owns what the Tagger learns from: its tags, its items (bookmarks, notes,
pages) and the tags people put on items by hand. It never tells the Tagger about a
change; the Tagger reads the `Catalog` when the space's revision `rev` changes. `rev` is an
opaque string the application changes whenever a tag, a hand-made link or a tagged item
of the space does (a counter on the space's row will do), and each tag's own revision
changes whenever its members do: a link added or removed, a member's text edited, a
member deleted. At the revision it was last built at, a Suggest never calls the catalog.
A space is named by one `tagging.Space{Index, Partition}` wherever the Tagger takes one,
`Scope` and `Prune` alike, and `space.Space()` gives a scope's back. As in search, `Index`
is the hard isolation boundary (a tenant) and `Partition` a space within it; scopes never
see each other's tags, and no embedding is read on behalf of another index. Use a tag's
full path as its name (`colors / warm / red` says
more than `red`). `TaggerConfig.Model` is the identity of the embedding model, its
`Model()` (or its spec's `ModelID()`), and every cached vector is kept under the model
it came from.

**Long items.** An embedder reads a window of tokens and silently drops the rest: the
default yzma model reads 512 tokens, some 2 KB of prose, so only the start of a longer item
counts. `TaggerConfig.ChunkBytes` (off by default) embeds a longer item in pieces of about
equal size cut at whitespace (`tagging.Chunk`), each cached on its own; set it a little
under the window, 2000 for the default model. An item of three pieces or more scores
against a tag by the mean of its two best pieces, so a topic one section is about counts
in full and one paragraph alone does not carry a tag; an item of two pieces by the best of
its two halves and their mean, since a short page's header and body land in different
halves. A long member teaches its tag with its piece closest to the tag's name, not with
the mean of all of them. On the 24 long golden documents (7 to 31 KB, 93 side topics past
the first 2 KB), against a catalog of 158 name-only tags made of the topic labels of all
of them, every main topic and 0.90 of the side topics were found, against 0.65 of the side
topics without `ChunkBytes`, at a precision of 0.80 (0.86 without) and with one passing
mention of 54 tagged in both (`TestTaggerLongTags`). In the experiment that chose the rule,
scored by exact label, the mean of the pieces found 0.75 of the side topics where this rule
found 0.89, and the best piece alone 0.96, but that let single paragraphs pull in the tags
of other documents (5.0 a document, against 3.5). The two cutoffs
hold unchanged: pieces score on the scale of short items. Changing `ChunkBytes` embeds only
the pieces the cache does not hold.

**Storage.** The Tagger keeps only derived data, and losing it costs time, never
correctness. An `EmbeddingCache` holds every embedding, keyed by index, model, role and the
hash of the text, so an edited text or another model simply misses and is embedded again.
`tagging.NewMemCache()` keeps it in memory, for tests and tiny corpora;
`tagging/sqlitecache` and `tagging/pgcache` keep it in one plain table of a SQLite or
Postgres database that you open and pass to `New(ctx, db)`; your application imports the
driver. An index or a model name with a NUL byte or invalid UTF-8 is refused by all three,
since Postgres cannot store one. Processes that start together may all call `New`: the table
is created under a lock. SQLite allows one writer at a time: when several goroutines write,
open it with `?_busy_timeout=5000&_txlock=immediate`, so a writer waits for its turn instead
of failing with "database is locked". `tagging/cachetest` is the conformance suite for a
cache of your own; it runs on a clock it moves by hand, so your cache must be able to take
its last-used times from the `now` function the suite passes. Each space's tag vectors and
cutoffs live in a `TagIndex`, by default a `MemIndex`
(an LRU of spaces under a 256 MB budget; `WithTagIndex` sets another), rebuilt from the
cache, never from the model, after a restart or an eviction. A space has few tags, so the
Tagger scores them all in Go.

**Pruning.** `tagger.Prune(ctx, unusedSince, spaces...)` removes the embeddings last used
before `unusedSince`, across every tenant: the texts of items edited or deleted since, and
the rows of a model you no longer use. Run it on a schedule, and call it instead of the
cache's own `Prune`:

```go
spaces := []tagging.Space{{Index: "tenant-1", Partition: "space-7"}} // every space still in use
removed, err := tagger.Prune(ctx, time.Now().AddDate(0, 0, -30), spaces...)
```

A tag that does not change is built once and never read from the cache again, so the
embeddings of its name and members look unused however often the tag is suggested. `Prune`
therefore first reads the `Catalog` of each space you pass and marks what its tags are built
from as used (`EmbeddingCache.Touch`), and only then deletes. A space you leave out loses
its old embeddings and embeds them again at its next rebuild: time, never correctness. If a
space's catalog fails, nothing is deleted. A read rewrites a last-used time only once it
is 24 hours stale (`tagging.CacheRefreshAfter`), so the recorded time can be up to a day
older than the last read, and a cutoff of age A keeps for certain only what was read in
the last A minus 24 hours: 30 days removes what was unread for 30 days and may remove
what was last read 29 days ago. The cutoff must therefore be at least two days old
(`tagging.MinPruneAge`), which protects the last day's reads; `Prune` refuses a more
recent one with an error, since it could delete embeddings read minutes ago.
When Taggers of several models that are all in use share one cache, call `Scope.Touch` on
the other Taggers' spaces first, since a `Prune` keeps only its own model's embeddings.

How a tag is judged:

- **Its vector** blends an embedding of its name with the mean of its members'
  vectors, weighted by member count: `w = n/(n+SmoothingK)`. A brand-new tag tags by
  its name alone, and shifts toward "tag by example" as members accumulate.
- **Its cutoff** mixes `NameMinScore` (default 0.42) and `MemberMinScore` (default
  0.58) by the same weight. Documents resemble each other far more than they resemble a
  short tag path, so each part is judged on its own scale: scoring the blended vector
  against the cutoff is exactly
  `(1-w)·(cos(name) - NameMinScore) + w·(mean cos(member) - MemberMinScore) >= 0`.
  The mean of a document's cosines to the members stays put as members are added,
  where its cosine to their centroid rises with every one, so one `MemberMinScore`
  holds for a tag with two examples and for one with fifty.
- **Every tag that clears its own cutoff is suggested**, best score first, with no limit
  on how many, but along a path of the tag tree (names split on `/`: `dev / go` is a
  child of `dev`) the better fit wins. Every ancestor meets every descendant by margin
  (score minus cutoff): a descendant at least as wide shadows the ancestor, and a
  narrower one is shadowed by it. A Go page gets `dev / go` and not also `dev`, which
  the path implies; a general programming page gets `dev` and not the `dev / web` it
  clears by less; two children that both beat their parent are both suggested.
  `TagScore.ShadowedBy` names the winner; `TaggerConfig.AllPathLevels` turns the rule
  off. A suggestion is never stored as a link: only the application's hand-made links
  teach a tag.
- **A new revision rebuilds only what moved.** The Tagger reads the catalog's tags,
  rebuilds from the cache the ones whose revision, name, model or settings changed, with
  one `Members` call per 64 tags, and drops the ones gone; the callers that ask at one
  revision together share one rebuild. So a bulk import costs one rebuild per changed
  tag, at the next call. The rebuilds of one space run one after the other: a caller at
  another revision (a second instance of your application that has not seen the latest
  change yet) waits, then rebuilds only if the space is still not at its revision. While
  two instances disagree on the revision, each call of one re-reads the catalog's tags
  after a call of the other, so let them converge.
- **A rebuild belongs to its callers.** It runs on a goroutine of the Tagger's that the
  waiting callers share. A caller whose context ends returns at once and the rebuild goes
  on for the others. When the last one has gone it goes on for
  `TaggerConfig.RebuildGrace` (30 seconds by default): a caller that asks about the space
  within that time joins the same rebuild, so a rebuild that takes longer than your
  request deadline still completes, for the requests after the ones that gave up on it,
  instead of being started over by every request and never finishing. With nobody back
  when the time is up, the rebuild's context is cancelled, so a hung embedder is not left
  running; a negative `RebuildGrace` cancels it as soon as its last caller has gone. A
  rebuild that hangs is joined by the callers that come within the grace period, each
  waiting until its own deadline, so give an embedder that can hang a timeout of its
  own. A panic of your `Catalog` or `Embedder` in a rebuild
  does not end the process: the callers get an error wrapping `ErrNotRebuilt` with the
  panic's value, and the stack goes to the `Logger`. `tagger.Close()` cancels the calls
  and rebuilds in flight, waits for them, and makes every later call return an error
  wrapping `langai.ErrClosed`.
- **A failure keeps what was done.** Texts go to the embedder 32 at a time and each batch
  is cached as it comes back, so a rebuild or a `Warm` that fails or is cancelled loses at
  most one batch, and the next call embeds only what is missing. A catalog failure, an
  embedder that is down, or an embedder failure on a tag's *name* (the error then names
  the tag) fails the call with no tags and an error wrapping `tagging.ErrNotRebuilt`; the
  index stays as it was, so the next call tries again.
- **One bad member does not block a space.** When a batch fails, its texts are embedded
  again one at a time. A member's text the embedder still fails on, or answers with a
  vector of NaN, infinite or all-zero values, is left out: its tag is built from its name
  and its other members, nothing is cached for it, and the Tagger tries it again at the
  space's next revision. `space.Incomplete(ctx)` lists the ids of the tags built that
  way, and `TaggerConfig.Logger` (a `*slog.Logger`, nil discards) gets a
  warning for each member, with the index, the partition, the tag's id and the SHA-256 and
  length of the text, so you can find the item; the text itself is not logged. `Warm`
  returns an error naming the texts it could not embed by their position, after
  embedding the rest; `Suggest` and `Scores` fail when their own text cannot be embedded.
- **An embedder that is down is not a bad member.** A member is left out only on evidence
  that the embedder works: it answered with a useless vector, or an `Embed` after the one
  that failed succeeded, of a later text or, when there is none, of the tag's name
  embedded once more as a canary. When that fails too, or the embedder's error wraps
  `langai.ErrClosed`, `context.Canceled` or `context.DeadlineExceeded` (a closed model, a
  request timeout of its own), the rebuild fails with an error wrapping
  `tagging.ErrNotRebuilt` and the cause, and nothing is recorded: the next call tries
  again, so an outage never leaves tags built from their names alone. After three texts in
  a row have failed the Tagger checks the canary before going on, so a dead embedder costs
  a handful of calls, not one per text. `Warm` ends the same way, with the embedder's
  error.
- **A cache that cannot be written still answers.** On a read-only replica or a locked
  SQLite file the vectors are read but their last-used time cannot be refreshed, and new
  embeddings cannot be stored. Neither fails a call: the Tagger uses the vectors it has
  and logs a warning.
- **One configuration per shared index:** Taggers that share a `TagIndex` (through
  `WithTagIndex`) must use the same `Model`, `NameMinScore`, `MemberMinScore`,
  `SmoothingK` and `ChunkBytes` (`AllPathLevels` may differ), or each takes the other's
  tags for stale and they rebuild at every call. Taggers of different models can share
  a cache: the model is part of every key.
- **Switching models** needs no migration: after you change `TaggerConfig.Model` the new
  model misses the cache and embeds what it needs, lazily, never mixing two models'
  vectors. `Scope.Warm` with the corpus's texts does it ahead of time, and `Tagger.Prune`
  later drops the old model's rows. Give every model its own `Model` name. If another
  model does end up under an old name, a cached vector that is not as long as the
  embedder's `Dimensions()` counts as a miss and is embedded again and overwritten. Where
  the lengths cannot be told apart beforehand (an embedder whose `Dimensions()` is 0, or a
  shared `TagIndex` built by the other model), `Suggest` and `Scores` return an error
  naming the tag rather than a score of 0. Two models of the same dimensions under one
  name cannot be told apart at all.
- **Rejected alternatives:** kNN voting on its own (it never abstains); a kNN agreement
  check (it blocks second tags on two-topic pages); one global cutoff (it fits only one
  stage, names or examples); the cosine to the members' centroid (it drifts with their
  number, so a cutoff right for five examples is too strict for two); centering the
  vectors on the scope's mean, or keeping only each document's best tags (both lift
  single-topic F1 and lose most second topics); EmbeddingGemma's classification or
  clustering prompts, and bge-m3, multilingual-e5-small and nomic-embed-text-v2 (all
  below EmbeddingGemma with its search prompts); a cap on tags per document (every
  qualifying tag is wanted).

The golden numbers below measure tagging over *essences*: a page's title, excerpt and
site name followed by the first 2 KB of its text, one per line
(`eval.Document.Essence`). Pass `Suggest`, and your catalog's `Members`, your documents
built the same way to get what the numbers say.

**Upgrading long items.** With `ChunkBytes` set, a long item used to be cached as the mean
of its pieces under the model marked with the chunk size (`<model>+chunks=<n>`); now each
piece is cached under the model itself. The first call after the upgrade embeds the pieces
of every long item once and rebuilds the tags that have long members; `Prune` later drops
the old rows. Without `ChunkBytes` nothing changes.

**Upgrading from the Store.** The Tagger used to keep its own copy of your tags, items and
links in a `tagging.Store`; now it reads them from your `Catalog`, so the application's
writes no longer go through it. `NewTagger(emb, store, cfg)` became
`NewTagger(emb, catalog, cache, cfg)`, and `SetTag`, `SetItem`, `TagByHand` and their
kin are gone; `Suggest` and `Scores` now take `(ctx, rev, text)`. No data needs to move:
the cache starts empty and every embedding is recomputed on first use. The old tables are
left behind for your application to drop, as nothing reads them any more:
`autotag_<first 16 hex of sha256(index)>_{tags,items,links}`, and
`autotag_<generation>_<hash>_{tags,items,links}` for generations. The example service's
old `data/autotag-<model>.sqlite` files can be deleted.

### Known limitations

- **Second and side topics.** Without `ChunkBytes` a page is one embedding of its first
  ~2 KB. When pairs of single-tag golden documents are stitched into one page (2.3 to
  2.9 KB), both tags are assigned 0.80 of the time, 0.87 with `ChunkBytes` 2000. On long
  items one side topic in ten is still missed with `ChunkBytes` (see **Long items**),
  and a topic a single paragraph covers rarely gets its tag, by design.
- **Documents that fit no tag** got none on the golden set, by a small margin: with the
  whole family of a document's tags removed (the tag, its parent, its siblings and its
  children), none of the 84 documents got one of the tags left, and the closest came
  within 0.016 of its cutoff. That is 15 tags in five families and one model, so expect
  a wrong tag now and then on a larger vocabulary.
- **A document whose own tag is missing gets a related one.** With only a document's own
  tags removed, 0.64 of the golden documents get no tag, and 0.81 none but an ancestor
  of their own tags (`cooking` for a recipe once `cooking / recipes` is gone), which the
  path implies. The rest get a sibling or a child, a related tag but a wrong one.
- **Siblings are the usual mistake.** Nearly every wrong suggestion is a tag of the
  right family: `cooking / baking` for a recipe, a parent next to its child. Unrelated
  tags are all but never suggested.
- **Tags that mix unrelated topics** average their members into one vector that
  matches neither topic well.
- **The two cutoffs come from a small golden set** (84 documents, 15 tags, one model);
  the model's scores sit in a narrow band, so ±0.04 on the name cutoff already costs a
  lot of precision or recall. Re-check them on real data, and measure them again for
  another model. Both the essences and the long documents score a little better at a
  name cutoff of 0.43 (F1 +0.02 and +0.03), which costs stitched two-topic pages 0.04;
  the default stays 0.42.

### Generating tags

```go
emb, err := yzma.New(ctx, cfg)          // the embedding model scores the tags
llm, err := yzma.NewCompleter(ctx, cfg) // the chat model writes them
gen, err := tagging.NewGenerator(llm, emb, tagging.GeneratorConfig{})
tags, err := gen.Generate(ctx, essence, tagging.DefaultGeneratorMinScore) // []tagging.Tag{Name, Score}, best first

// options: the language of the tags, and tags as paths of a tree
gen, err = tagging.NewGenerator(llm, emb, tagging.GeneratorConfig{},
	tagging.WithLanguageTags("en"),                       // or WithNativeLanguageTags(), the default
	tagging.WithTreeTags("dev / go", "cooking / baking")) // extend these paths; none: a tree of its own
```

**Options.** `WithNativeLanguageTags()` (the default) writes the tags in the language of the
text; `WithLanguageTags(lang)` in one language whatever the text's, `lang` a BCP 47 tag
(`en`, `es`, `pt-BR`); the prompt opens with that rule, asks the model to translate the
text's topics rather than copy its words (proper names stay), and repeats the rule last.
A 4B model copies a text's words readily: with the rule stated once, last, English tags
for the 55 German and Spanish golden documents had 88 of 550 tags still in the text's
language; stated first and last, 28, most of them place and dish names. With `WithTreeTags`,
`WithNativeLanguageTags` states its rule first and last too; without, last only (first as
well cost long texts 0.07 of side recall). `WithTreeTags(existing...)` writes each tag as a path from a broad
category down (`dev / go / error wrapping`, at most `MaxTreeLevels` deep) and the tags of a
text as a small tree sharing upper levels. With existing paths, the model is shown them (up
to `MaxTreeBytes`) and asked to extend them, written as they are, and to start a new
top-level category only when none fits; a tag is scored by its whole path. A path the model
reuses, whole or as the start of a longer one, comes back in your spelling whatever case or
spacing the model wrote (`cooking/baking` becomes your `Cooking / Baking`), and is kept even
when one of its levels is longer than the four words and 40 characters a level the model
adds may have. On a Spanish
text about Go errors, with the golden tags as the tree, it wrote `dev / go / error wrapping`
and three siblings, in English although the text is Spanish: the tree's language wins over
`WithNativeLanguageTags`, although the prompt also says that new levels follow the text.
Measured 2026-11-07 with an English tree on six German and Spanish documents, every tag
came back English; `WithLanguageTags` turned the German ones German, existing levels
translated too, but left two of the three Spanish ones English. So set `WithLanguageTags` to
the tree's language. Without existing paths the native rule holds; before it was stated
first, short German and Spanish texts got English trees. On a text that fits no
branch it opened new roots (`tech / agentic web`) rather than reaching for `dev / web`. A
Generator keeps no state, so building one per call, with the tree as it is now, is cheap.

The Tagger picks among tags your application already has. The Generator needs none: it writes them
for one text, stores nothing, and returns the tags that score at least the cutoff you pass.
For a short text, hand it one built like the essences above; for a long page, pass the
title, a newline and then the full body (see **Long texts**). A text just over
`ChunkBytes` (2 KB) still costs one call, but its tags are scored per chunk, a scale
measured on long documents and not on essences.

- **The model writes, the embedder scores.** The chat model (Qwen3 4B Instruct, Apache-2.0)
  is asked for up to `GeneratorConfig.Candidates` (10) tags of one to three words, in the text's
  language unless the options above say otherwise. A model's own confidence is poorly calibrated, so each tag is scored by the
  cosine between the embedding of the text (as a query) and of the tag (as a document):
  the roles, and so the scale, of the Tagger's `NameMinScore`. One cutoff then means the same
  on every text, and a tag the model invented falls under it.
- **The reply is read leniently.** The model is asked for one tag per line, and what it
  wraps around them is dropped: a `<think>…</think>` block (one never closed takes the
  rest of the reply), code fences, a label such as `Tags:`, `Here are the tags:` or
  `标签：`, list markers, numbers and checkboxes, quotes, brackets, Markdown emphasis,
  and a remark in brackets after a tag (`sourdough (bread)`). Several tags on a line are
  split at commas and semicolons, the Chinese, Japanese and Arabic ones included
  (`、` `，` `；` `،` `؛`), and a run of hashtags (`#go #rust`) is one tag each. A reply
  that is JSON gives only the strings that are tags: the strings of an array, and in an
  object the members named `tags`, `keywords`, `labels` or `topics` and each list item's
  `name`, `tag` or `label`; scores, flags and other members (`"language": "en"`) are
  never tags, and such a reply is not read by lines as well. A reply that is only
  `None`, `No tags.`, `N/A` or the like, in English, Spanish or German, means no tags.
  What belongs to a tag stays: a colon unless a label word stands before it
  (`Topic modeling: LDA`, `Star Trek: Voyager`), a bracket that closes inside it
  (`(re)insurance`), an inch mark (`5" floppy`), a lone `<` or `>` (`x < y`). An arrow
  is kept in a plain tag (`a -> b`) and is the step to the next level in a tree tag
  (`a / b`). A name left with a bracket or a quote that does not close, with an HTML
  element, or longer than four words or 40 characters, is not a tag.
- **A bad answer is asked for again, once.** A model call that fails or replies with
  nothing is repeated, unless the context has ended or the error wraps
  `langai.ErrClosed`: a closed model is asked once. If it fails again on a short text,
  `Generate` returns the error; a model that twice says nothing is
  `tagging.ErrEmptyReply`, not "no tags". On a long text the other windows still count:
  `Generate` returns their tags *and* an error wrapping `tagging.ErrPartial` that names
  the windows it lost. A model that is gone is not asked about the rest of the text: a
  closed one stops the reading at once, and when the first two windows fail with none
  read yet the others are skipped, so a model that is down costs four calls, not two per
  window. Check the error before you discard the tags:

  ```go
  tags, err := gen.Generate(ctx, text, tagging.DefaultGeneratorMinScore)
  if err != nil && !errors.Is(err, tagging.ErrPartial) {
      return err // nothing was read
  }
  // tags are good; with ErrPartial, topics of the lost windows may be missing
  ```
- **The cutoff is yours.** `-1` returns every candidate. `tagging.DefaultGeneratorMinScore` (0.45)
  fits the default models: over ten golden documents in English, Spanish and German the
  model's tags scored 0.31 to 0.68, the main topics at 0.5 and up, and what fell under
  0.45 was side detail (the steam for a loaf, a side salad). That is a judgment by eye on a
  small set, and another embedder scores on another scale, so choose again for it.
- **Up to 64 KB are read** (`GeneratorConfig.MaxBytes`); nothing after that reaches the
  model or the embedder, so a tag is never scored against words the embedder did not see.
- **Long texts.** A text over 2 KB (`GeneratorConfig.ChunkBytes`) is read in windows of
  about 6 KB, one LLM call each, a short first line (the title) repeated at the head of
  every window, so pass the text as `title\n` + body; the names merge, and each is scored by its best cosine
  over the 2 KB chunks of the text. On 24 synthetic documents of 7 to 31 KB (93 side
  topics, all past the first 2 KB) that scored a mean F1 of 0.82 against 0.49 for reading
  the first 2 KB only: side-topic recall 0.76 against 0.24, precision 0.89 against 0.78,
  main-topic recall 1.0 in both, and no passing mention tagged (`TestGenerateLongTags`).
  Scoring by the best chunk won because the mean of the chunks and
  the top-k mean dilute exactly the section topics the windows find. 6 KB windows beat
  2 KB ones (F1 0.84 against 0.74 in the experiment, under an earlier scoring rule) because the model needs some context to name a
  section's topic, and 2 KB windows give many near-duplicates at 2.7 times the calls.
  The price is time: about 3 LLM calls and 15 s for a document of 16 KB on the test
  laptop's iGPU, and a 64 KB text is about 11 calls. `zarf/examples`' autotag pays this
  for long notes. The default chat context is 4096 tokens so that a window fits, which
  roughly doubles the model's KV cache (a few hundred MB).
- **Upgrading.** The defaults changed: `MaxBytes` went from 2 KB to 64 KB, so a long text
  that took one call now takes up to 11, and yzma's default chat context went from 2048
  to 4096 tokens. `GeneratorConfig{MaxBytes: 2048}` restores today's behaviour: one call
  on the first 2 KB. With a chat context under 4096 tokens keep `MaxBytes` at or under
  `ChunkBytes`: a 6 KB window, in German about 2,500 tokens, may not fit, and then that
  window fails (twice, see above). If you never tag long texts, set
  `CompleterSpec.ContextTokens` to 2048 to keep the smaller KV cache.
- **Cost.** The chat model is a 2.5 GB download, and a process with it and the embedding
  model loaded holds about 5 GB. A text takes 7 to 13 s with the default 4 threads on an
  AMD Ryzen AI 9 HX PRO 370 (English about 7 s, German 11), nearly all of it writing the
  tags; `yzma.Runtime.Threads` raises that. Decoding is greedy, so a text gets the same tags
  every time.
- **Limits.** The tags are as good as a 4B model's: their casing is the model's, a rare
  one is malformed (`FitnessimHome`), and they come back in the text's language. They
  cover what the text says; to tag against a vocabulary you keep, use the Tagger.

## Evaluating embedding models

`eval` scores an embedding model on a labelled, multilingual golden set
(`zarf/eval/testdata/golden.yaml`, English, Spanish and German): free-text retrieval
(recall@k, MRR, nDCG) over the queries in their documents' language and, separately,
over the queries that must cross a language; tagging by tag name (precision, recall and
F1 at 0.46, each tag judged on its own, plus the floor each model would need); and timings. tagging's
`TestGoldenTagging` replays the same set through a `Tagger` with simulated hand-tagging
(a tag's members are up to every other document carrying it) and checks tagging by name
only and with members, how many documents stay untagged once the whole families of their
tags are removed (and how many get none but an ancestor once only their own tags are),
and two-topic pages. Both need the runtime libraries above and cache the downloads in
`~/.cache/semsearch-yzma`:

```
# the default model against the floors CI enforces
# (provider/yzma/golden_test.go and tagging/golden_test.go)
make test-models

# every candidate in zarf/eval/testdata/models.yaml, side by side
make semeval
```

Add a candidate to `zarf/eval/testdata/models.yaml` with its pooling and prefixes;
run `make semeval SEMEVAL_ONLY=<its name>` once without a digest and paste the one it
prints. The dataset format is plain YAML, so an export of your own documents can be
scored the same way (`make semeval SEMEVAL_DATASET=export.yaml`); `SEMEVAL_JSON=out.json`
keeps every metric, and the Makefile lists the other settings.

## Examples

`zarf/examples/` is a single-user web service that exercises the features by hand,
one page per feature. Media search is the first: `make example-prepare` fetches
sample photos and renders two slideshow clips, `make example` indexes them
and serves a grid of thumbnails on http://127.0.0.1:8085/mediasearch/; click one
to see the items that look like it with their scores, the frame of a video that
matched and when, the frames the index saw of a video, and how long the search
took; or search with a file of your own. See [zarf/examples/README.md](zarf/examples/README.md).

## Development

```
make test          # go tests (the pgvector ones skip without TESTDBS_ALL=1 and Docker; the ffmpeg ones without ffmpeg)
make test-models   # the real-model tests (downloads ~350 MB on first run)
make test-vision   # the vision embedder on the real model (2.7 GB; loads it twice, the second time on the CPU)
make test-gentags  # the tag Generator on the real models (adds the 2.5 GB chat model; minutes on a CPU)
make test-images   # the real image-model tests: ONNX Runtime + SigLIP 2 (downloads ~380 MB on first run)
make coverage      # per-package coverage gate (>= 70%), pgvector tests included
make lint          # golangci-lint
make license-check # go-licence-detector
make verify        # run every check
```

## License

langai is licensed under the [GNU Lesser General Public License v3.0](LICENSE).

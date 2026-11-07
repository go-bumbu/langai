# langai

Libraries that use language models, run locally: semantic search over your own
documents, image and video similarity search, automatic tagging that learns from the
tags people assign by hand, and tags written for a text. Nothing leaves the process: the
models run in-process through llama.cpp and ONNX Runtime.

## Packages

| Package | What it does |
|---|---|
| `langai` | The ports: `Embedder` (text embedding models), `ImageEmbedder` (image embedding models), `Completer` (chat models) and the `Role` a text is embedded as. `LazyEmbedder` defers loading a model to its first use. |
| `search` | Turns documents into embedding vectors and answers free-text queries with the closest ones: `Index` and the markdown-aware chunker. `MediaIndex` does the same for pictures: it embeds images and videos (as sampled frames through an image model, or as whole segments through a video model) and answers an image, a video or, with a model that has a text side, a text with the items that look most like it, across both kinds. |
| `media` | The ports for video input: `Sampler` turns a video file into the `Frame`s that stand for it, `VideoEmbedder` embeds it as whole `Segment`s, `Cutter` measures and cuts it. |
| `media/ffmpeg` | `Sampler` and `Cutter` over the ffmpeg and ffprobe binaries: the first frame, every scene cut and one frame per interval, scaled down, capped per video, or the keyframes only, for long videos; stream-copy cuts for the segment embedder. |
| `tagging` | Tags for a text. `Tagger` suggests tags among the ones an application already has and learns from the tags people put on items by hand: each tag's vector blends its name with the items tagged with it, and each tag is judged against its own cutoff. The application provides its tags and hand-made links through a `Catalog`; the Tagger keeps only an embedding cache (`MemCache`, `tagging/sqlitecache`, `tagging/pgcache`) and an in-memory index of tag vectors. `Generator` writes tags with no list to start from: a chat model proposes them and the embedding model scores each against the text, so one cutoff means the same on every text. |
| `vector` | The `Store` interface for search vectors, and `MemStore` (exact, in memory, for tests and evaluation). |
| `vector/sqlitevec` | `Store` on SQLite with the sqlite-vec extension (CGO). |
| `vector/pgvec` | `Store` on Postgres with pgvector. |
| `provider/yzma` | `Embedder` that runs a GGUF sentence-embedding model, `Completer` that runs a GGUF chat model (the one the `Generator` writes tags with), and `Vision`, which runs Qwen3-VL-Embedding as a text, image and video embedder in one space, through llama.cpp, loaded at runtime by [yzma](https://github.com/hybridgroup/yzma) (purego + libffi, no CGO). Downloads and verifies the pinned llama.cpp build and models on first use, and picks the build and the GPU offload from the hardware (`Config.Accel`). |
| `provider/onnx` | `ImageEmbedder` that runs an ONNX vision model through [ONNX Runtime](https://onnxruntime.ai) (CGO, via [onnxruntime_go](https://github.com/yalue/onnxruntime_go)): SigLIP 2 base by default, DINOv2 base as the other pinned model, any other through a `ModelSpec`. CPU by default, CUDA or CoreML by configuration. Downloads and verifies the pinned runtime build and model on first use. |
| `zarf/eval` | Measures an `Embedder` on a labelled, multilingual golden set: retrieval, tagging, timings. `make semeval` compares embedding models side by side on it. |
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
│   ├── sqlitecache/    the Tagger's EmbeddingCache on SQLite
│   ├── pgcache/        the Tagger's EmbeddingCache on Postgres
│   ├── cachetest/      conformance suite for an EmbeddingCache
│   └── internal/       sqlcache: the SQL the two caches share
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
├── langaitest/         fakes of the ports, for tests
├── internal/           tablename: SQL table names; fetch: verified downloads and release
│                       archives for the providers; planned: rank, score roll-up
│
└── zarf/               tooling around the library, not part of it
    ├── eval/           golden-set evaluation of embedding models
    └── examples/       the example service: a page per feature, media search and auto tagging
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
- **GPUs with `provider/yzma`.** `Config.Accel` left empty picks from the hardware: a
  managed `LibDir` gets the pinned llama.cpp build that carries the backend the machine
  has (CUDA 13.4 on linux/amd64 with the CUDA runtime installed, Vulkan on linux/amd64
  and linux/arm64, Metal on macOS, the CPU build otherwise; `internal/hw` looks for the
  driver), the model is offloaded to the GPU that registers, and the load log line says
  which. `AccelCPU` stays on the CPU, `AccelVulkan`, `AccelCUDA` and `AccelMetal`
  require their backend. An operator-supplied `LibDir` is used as it is: whatever
  backends it holds register. Vulkan was run on an AMD iGPU for this release; the CUDA
  build is pinned but was not run.
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
hits, err := ix.SearchIndexed(ctx, "clip-7", 10)         // "more like this" with the stored vectors: no model runs
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
  different spaces: one model per collection, as `tagging` keys every embedding by model.
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
  rotate it is seen unrotated. The SigLIP 2 text tower is not wired, so text queries
  need the vision model below. The CUDA and CoreML paths are wired but were not run on a
  GPU in this release; the CPU path is what the real-model tests cover.

### Text, images and videos in one space

```go
v, err := yzma.NewVision(ctx, yzma.VisionConfig{
    Config: yzma.Config{ModelsDir: "data/models", LibDir: filepath.Join("data", "llama-cpp", yzma.LibTag), ManagedLibDir: true},
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
instruction (`Represent the user's input.` unless `VisionConfig.Instruction` says
otherwise), the projector turns pictures and frames into tokens, and the vector is the
last token's hidden state. Texts, images and videos land in one space, so a text finds
pictures and clips, and an image finds the stretch of a video it appears in.

- **Videos as segments.** A video goes through the model in segments of
  `SegmentLength` (20 s), each sampled at `FPS` (1) and embedded as one input: about 190
  tokens per second of video, so a segment is a few thousand tokens and a feature film
  never overflows the context (`ContextTokens`, 16384). The `Cutter` (the ffmpeg
  `Sampler`) cuts by stream copy, so a segment starts at the last keyframe before its
  time. `MaxSegments` caps the work per video. A segment is one vector: the model sees
  the whole stretch, which is what makes a clip match a description, and what loses the
  "matched at 4.00 s" precision of the frame route.
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
      rebuilds the changed tags from the `Catalog` and the cached embeddings.
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
   3. It embeds and caches any texts the cache doesn't have yet.
4. **Generate** (new tags)
   1. The app calls `Generate` with the text and a minimum score.
   2. The library cuts the text to `MaxBytes` (64 KB by default).
   3. A text of up to `ChunkBytes` (2 KB by default) goes to an LLM in one call. A longer
      one is read in windows of about 6 KB, one LLM call each, with a short first line (the
      title) repeated at the head of every window after the first, and the names of all
      the windows are merged.
   4. The LLM writes up to `Candidates` tags (10 by default) per call, either as short
      words or as tree paths, optionally extending the app's existing paths.
   5. The library embeds the text (in 2 KB chunks when it is long) and each candidate
      tag. Nothing is cached.
   6. Each candidate is scored by the cosine between its vector and the text's vector;
      for a long text, by its best cosine over the chunks.
   7. Candidates below the minimum score are dropped, and a long text's are capped at
      `Candidates`.
   8. The library returns the remaining tags with their scores, best first.
5. **Cache maintenance** (the app calls these on the cache directly)
   1. `Prune` deletes the embeddings not used since a given time, across all tenants.
      The app runs it on a schedule.
   2. `DeleteIndex` deletes all embeddings of one tenant, for example when the tenant is
      removed.

There's no "learn" call. The Tagger learns only when the app changes the revision.

### Automatic tagging

```go
// catalog is your tagging.Catalog: Tags lists a space's tags (id, name, revision),
// Members the texts of the items tagged with some of them by hand
cache, err := sqlitecache.New(ctx, db)
tagger, err := tagging.NewTagger(emb, catalog, cache, tagging.TaggerConfig{Model: yzma.DefaultModel})
space := tagger.Scope("tenant-1", "space-7")
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
As in search, the first argument of `Scope` is the hard isolation boundary (a tenant) and
the second a space within it; scopes never see each other's tags, and no embedding is read
on behalf of another index. Use a tag's full path as its name (`colors / warm / red` says
more than `red`). `TaggerConfig.Model` names the embedding model, and every cached vector
is kept under the model it came from.

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
of them, side topics were found 0.68 of the time, against 0.58 by the mean of the pieces
and 0.45 without `ChunkBytes`, with precision unchanged (0.78) and one passing mention of
54 tagged, as before (`TestTaggerLongTags`). The best piece alone found 0.74 but let single
paragraphs pull in tags that match none of a document's topics (3.5 a document, against
2.0). The two cutoffs
hold unchanged: pieces score on the scale of short items. Changing `ChunkBytes` embeds only
the pieces the cache does not hold.

**Storage.** The Tagger keeps only derived data, and losing it costs time, never
correctness. An `EmbeddingCache` holds every embedding, keyed by index, model, role and the
hash of the text, so an edited text or another model simply misses and is embedded again.
`tagging.NewMemCache()` keeps it in memory, for tests and tiny corpora;
`tagging/sqlitecache` and `tagging/pgcache` keep it in one plain table of a SQLite or
Postgres database that you open and pass to `New(ctx, db)`; your application imports the
driver. `Prune` drops the rows not used since a given time, such as the texts of edited
items. SQLite allows one writer at a time: when several goroutines write, open it with
`?_busy_timeout=5000&_txlock=immediate`, so a writer waits for its turn instead of failing
with "database is locked". `tagging/cachetest` is the conformance suite for a cache of your
own. Each space's tag vectors and cutoffs live in a `TagIndex`, by default a `MemIndex`
(an LRU of spaces under a 256 MB budget; `WithTagIndex` sets another), rebuilt from the
cache, never from the model, after a restart or an eviction. A space has few tags, so the
Tagger scores them all in Go.

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
  one `Members` call, and drops the ones gone; the callers that ask at one revision
  together share one rebuild. So a bulk import costs one rebuild per changed tag, at the
  next call. A catalog or embedder failure in a rebuild fails the call with no tags and an error
  wrapping `tagging.ErrNotRebuilt`; the index stays as it was, so the next call tries
  again, and what was embedded before the failure stays cached.
- **One configuration per shared index:** Taggers that share a `TagIndex` (through
  `WithTagIndex`) must use the same `Model`, `NameMinScore`, `MemberMinScore`,
  `SmoothingK` and `ChunkBytes` (`AllPathLevels` may differ), or each takes the other's
  tags for stale and they rebuild at every call. Taggers of different models can share
  a cache: the model is part of every key.
- **Switching models** needs no migration: after you change `TaggerConfig.Model` the new
  model misses the cache and embeds what it needs, lazily, never mixing two models'
  vectors. `Scope.Warm` with the corpus's texts does it ahead of time, and `Prune` later
  drops the old model's rows.
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
  items a third of the side topics are still missed with `ChunkBytes` (see **Long items**),
  and a topic a single paragraph covers rarely gets its tag, by design.
- **Documents that fit no tag** can still get one: with a document's own tags removed,
  0.64 of the golden documents get no tag, and 0.81 none but an ancestor of their own
  tags (`cooking` for a recipe once `cooking / recipes` is gone), which the path
  implies. The rest get a sibling or a child, a related tag but a wrong one.
- **Siblings are the usual mistake.** Nearly every wrong suggestion is a tag of the
  right family: `cooking / baking` for a recipe, a parent next to its child. Unrelated
  tags are all but never suggested.
- **Tags that mix unrelated topics** average their members into one vector that
  matches neither topic well.
- **The two cutoffs come from a small golden set** (84 documents, 15 tags, one model);
  the model's scores sit in a narrow band, so ±0.04 on the name cutoff already costs a
  lot of precision or recall. Re-check them on real data, and measure them again for
  another model. Both the essences and the long documents score a little better at a
  name cutoff of 0.43 (F1 +0.02), which costs stitched two-topic pages 0.04; the default
  stays 0.42.

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
language; stated first and last, 28, most of them place and dish names. `WithTreeTags(existing...)` writes each tag as a path from a broad
category down (`dev / go / error wrapping`, at most `MaxTreeLevels` deep) and the tags of a
text as a small tree sharing upper levels. With existing paths, the model is shown them (up
to `MaxTreeBytes`) and asked to extend them, written as they are, and to start a new
top-level category only when none fits; a tag is scored by its whole path. On a Spanish
text about Go errors, with the golden tags as the tree, it wrote `dev / go / error wrapping`
and three siblings, in English although the text is Spanish: the tree's language wins over
`WithNativeLanguageTags`, so set `WithLanguageTags` to the tree's. On a text that fits no
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
  topics, all past the first 2 KB) that scored F1 0.82 against 0.54 for reading the first
  2 KB only: side-topic recall 0.71 against 0.25, main-topic recall 1.0 in both, and no
  passing mention tagged. Scoring by the best chunk won because the mean of the chunks and
  the top-k mean dilute exactly the section topics the windows find. 6 KB windows beat
  2 KB ones (F1 0.84 against 0.74 in the experiment) because the model needs some context to name a
  section's topic, and 2 KB windows give many near-duplicates at 2.7 times the calls.
  The price is time: about 3 LLM calls and 15 s for a document of 16 KB on the test
  laptop's iGPU, and a 64 KB text is about 11 calls. `zarf/examples`' autotag pays this
  for long notes. The default chat context is 4096 tokens so that a window fits, which
  roughly doubles the model's KV cache (a few hundred MB).
- **Upgrading.** The defaults changed: `MaxBytes` went from 2 KB to 64 KB, so a long text
  that took one call now takes up to 11, and yzma's default chat context went from 2048
  to 4096 tokens. `GeneratorConfig{MaxBytes: 2048}` restores today's behaviour: one call
  on the first 2 KB. With a chat context under 4096 tokens keep `MaxBytes` at or under
  `ChunkBytes`: a 6 KB window, in German about 2,500 tokens, may not fit, and then the
  whole Generate call fails. If you never tag long texts, set
  `CompleterSpec.ContextTokens` to 2048 to keep the smaller KV cache.
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
(`zarf/eval/testdata/golden.yaml`, English, Spanish and German): free-text retrieval
(recall@k, MRR, nDCG) over the queries in their documents' language and, separately,
over the queries that must cross a language; tagging by tag name (precision, recall and
F1 at 0.46, each tag judged on its own, plus the floor each model would need); and timings. tagging's
`TestGoldenTagging` replays the same set through a `Tagger` with simulated hand-tagging
(a tag's members are up to every other document carrying it) and checks tagging by name
only and with members, how many documents get no unrelated tag once their own tags are removed,
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
one page per feature. Media search is the first: `cd zarf/examples && make prepare`
fetches sample photos and renders two slideshow clips, `make run` indexes them
and serves a grid of thumbnails on http://127.0.0.1:8085/mediasearch/; click one
to see the items that look like it with their scores, the frame of a video that
matched and when, the frames the index saw of a video, and how long the search
took; or search with a file of your own. See [zarf/examples/README.md](zarf/examples/README.md).

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

# Packages and layout

What each package of sense does, how the tree is organised, and how capabilities, ports and providers fit together.

| Package | What it does |
|---|---|
| `sense` | The ports: `Embedder` (text embedding models), `ImageEmbedder` (image embedding models), `VideoEmbedder` (video embedding models, one vector per `Segment` of a video), `Completer` (chat models), the `Role` a text is embedded as, and the optional `Identified`, by which a model names itself so that stores never mix two models' vectors. `Normalize` scales a vector to unit length for the providers, and `ErrClosed` is what every call on something closed wraps. `LazyEmbedder`, `LazyImageEmbedder` and `LazyCompleter` defer loading a model to its first use and release it on `Close`. |
| `search` | Turns documents into embedding vectors and answers free-text queries with the closest ones: `Index` and the markdown-aware chunker. `MediaIndex` does the same for pictures: it embeds images and videos (as sampled frames through an image model, or as whole segments through a video model) and answers an image, a video or, with a model that has a text side, a text with the items that look most like it, across both kinds. |
| `media` | The ports for decoding video: `Sampler` turns a video file into the `Frame`s that stand for it, `Cutter` measures and cuts it (for a `VideoEmbedder` that embeds in segments). |
| `media/ffmpeg` | `Sampler` and `Cutter` over the ffmpeg and ffprobe binaries: the first frame, every scene cut and one frame per interval, scaled down, capped per video, or the keyframes only, for long videos; stream-copy cuts for the segment embedder. |
| `tagging` | Tags for a text. `Tagger` suggests tags among the ones an application already has and learns from the tags people put on items by hand: each tag's vector blends its name with the items tagged with it, and each tag is judged against its own cutoff. The application provides its tags and hand-made links through a `Catalog`; the Tagger keeps only an embedding cache (`MemCache`, `tagging/sqlitecache`, `tagging/pgcache`) and an in-memory index of tag vectors. `Generator` writes tags with no list to start from: a chat model proposes them and the embedding model scores each against the text, so one cutoff means the same on every text. |
| `vector` | The `Store` interface for search vectors: documents in collections located by a `Coord` (index, partition, collection), batch writes, deletes by id, by collection (`DeleteCollection`) and by index, listing, the collection's model (`Model`), and one exact `Search` that returns the best chunk of each of the best documents. `Validate`, the argument checks every backend shares (`CheckCoord`, `CheckID`, `CheckWrite`, `CheckSearch`…) and the sentinel errors (`ErrDimensions`, `ErrInvalidVector`, `ErrModelMismatch`, `ErrEmptyIndex`, `ErrInvalidID`), and `MemStore` (exact, in memory, for tests and evaluation). |
| `vector/vectortest` | The conformance suite every `Store` passes. |
| `vector/sqlitevec` | `Store` on SQLite with the sqlite-vec extension (CGO). |
| `vector/pgvec` | `Store` on Postgres with pgvector. |
| `provider/yzma` | `Embedder` that runs a GGUF sentence-embedding model, `Completer` that runs a GGUF chat model (the one the `Generator` writes tags with), and `Vision`, which runs Qwen3-VL-Embedding as a text, image and video embedder in one space, through llama.cpp, loaded at runtime by [yzma](https://github.com/hybridgroup/yzma) (purego + libffi, no CGO). Each opens two ways: `New`, `NewCompleter` and `NewVision` from a `Config` naming a pinned model (`Models`, `CompleterModels`, `VisionModels`) with the `Runtime` embedded, or `Open`, `OpenCompleter` and `OpenVision` from a spec (`LookupSpec`, `LookupCompleterSpec`, `LookupVisionSpec`, or your own) and a `Runtime`; a `Vision` also takes its `VisionOptions`. Downloads and verifies the pinned llama.cpp build and models on first use, and picks the build and the GPU offload from the hardware (`Runtime.Accel`). |
| `provider/onnx` | `ImageEmbedder` that runs an ONNX vision model through [ONNX Runtime](https://onnxruntime.ai) (CGO, via [onnxruntime_go](https://github.com/yalue/onnxruntime_go)): SigLIP 2 base by default, DINOv2 base as the other pinned model (`Models`, `LookupSpec`), any other through a `ModelSpec`; `New` and `Open` have yzma's shape. CPU by default, CUDA or CoreML by configuration. Downloads and verifies the pinned runtime build and model on first use. |
| `provider/phash` | `ImageEmbedder` without a model: a 64- or 256-bit perceptual hash (pHash or dHash) as a unit vector whose cosine is `1 - 2·hamming/bits`, for finding exact and near-exact duplicates (re-encoded, resized, lightly edited copies) with the same `MediaIndex` and stores. Pure Go, nothing to download, a few milliseconds per picture. `Hash` gives the bits themselves for grouping without a store. |
| `zarf/eval` | Measures an `Embedder` on a labelled, multilingual golden set: retrieval, tagging, timings. `make semeval` compares embedding models side by side on it. |
| `sensetest` | Fakes of the ports, for tests. `FakeEmbedder`, `FakeImageEmbedder` and `FakeVideoEmbedder` are deterministic and each names a fixed model (`FakeEmbedderModel`, `FakeImageEmbedderModel`, `FakeVideoEmbedderModel`); a `FakeVideoEmbedder`'s segments are pictures embedded in `FakeImageEmbedder`'s space, listed per file name or derived from it; `FakeCompleter` is a scripted chat model: canned replies in order or a function of the prompt, an error on demand, and a record of the prompts it received. All four fail with the error of a context that is done and with their `Err` field when it is set, and count their calls; `FakeEmbedder.Fail` fails the calls that hold a text of your choosing. |

Every package stands alone and you only compile the dependencies of the packages you
import: CGO and sqlite-vec come in only with `vector/sqlitevec`, pgvector only with
`vector/pgvec`, llama.cpp only with `provider/yzma`, CGO and ONNX Runtime only with
`provider/onnx`, and the ffmpeg binary is only run by `media/ffmpeg`. `provider/phash`
is pure Go and brings in nothing.

## Layout

The packages form a tree, and the tree is also the plan: packages marked *planned* do
not exist yet (see [TODO.md](../TODO.md)).

```
sense/                 the shared ports: Embedder, ImageEmbedder, VideoEmbedder, Completer,
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
├── sensetest/         fakes of the ports, for tests
├── internal/           tablename: SQL table names; fetch: verified, resumable downloads and
│                       locked library installs for the providers; hw: which accelerators
│                       the machine can run; imageutil: the picture scaling the providers
│                       share; planned: rank, score roll-up
│
└── zarf/               tooling around the library, not part of it
    ├── eval/           golden-set evaluation of embedding models
    └── examples/       the example service: a page per feature (media search, text search,
                        duplicates, auto tagging) and a status page
```

A capability never names a model: it calls a port, and the app hands it a provider that
implements that port. For tagging, with the default llama.cpp runtime:

```
                       port (sense)           provider (your choice)
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

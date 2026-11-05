# Image and video similarity search

Date: 2026-11-05. Status: approved by default (the author asked for an
unattended build and left the choices to the builder; every choice below is
the builder's unless marked *given*).

## Intent

*Given:* a library that other applications use to find images and videos that
look alike. The application runs a background job that indexes its images and
videos; later it hands the library an image or a video and gets back the
indexed items closest to it, across both kinds: an image query finds videos,
a video query finds images. Everything runs locally on self-hosted hardware,
CPU or GPU; CGO is acceptable.

## Goals

- Index an image or a video under an opaque id; re-indexing unchanged content
  is a cheap no-op (same contract as `search.Index`).
- Query with an image or a video; get ids ranked by similarity, with a score
  and, for videos, where in the video the match is.
- Cross-media: images and videos share one vector space (a video is the set
  of its sampled frames, each embedded as an image).
- Reuse `vector.Store` (MemStore, sqlitevec, pgvec) unchanged.
- CPU works out of the box; a GPU (CUDA) is a configuration change.
- Mirror `provider/yzma`'s operational model: pinned, digest-verified
  downloads of the runtime and the model on first use; offline afterwards.

## Non-goals

- Text-to-image search (the SigLIP 2 text tower). The embedder is built so it
  can be added as another spec later; nothing here depends on it.
- A video-native model (X-CLIP, V-JEPA). Frames through the image model are
  the cheapest start and keep one vector space.
- Captioning or tagging of images (`mediatag`, `caption` stay planned).
- EXIF orientation of JPEGs: a rotated phone photo is embedded as decoded.
  Follow-up in TODO.md.
- Exact near-duplicate detection (perceptual hashes). A semantic model scores
  near-duplicates very high anyway; a dedicated hash is a later addition.

## Anti-goals

- Pulling a heavy pipeline framework (hugot, gomlx) for one model: the
  dependency is the ONNX Runtime binding and `golang.org/x/image`, nothing else.
- Any change to the text search API or to the `vector.Store` port.
- Running ffmpeg through CGO bindings: the binary on PATH is the dependency.

## Architecture

```
langai/                 + ImageEmbedder port
├── search/             + MediaIndex: images and videos in, ranked ids out
├── media/              Frame, Sampler port (video file → frames), SamplerFunc
│   └── ffmpeg/         Sampler over the ffmpeg and ffprobe binaries
├── provider/
│   └── onnx/           ImageEmbedder over ONNX Runtime (CGO): SigLIP 2, DINOv2
└── langaitest/         + FakeImageEmbedder
```

Dependencies point one way, as in the rest of the module: `search` imports
`langai`, `vector` and `media` (ports only); `provider/onnx` imports `langai`
and `vector` (for `Normalize`); `media/ffmpeg` imports `media`. The
application wires them together.

### Port: `langai.ImageEmbedder`

```go
type ImageEmbedder interface {
    EmbedImages(ctx context.Context, images []image.Image) ([][]float32, error)
    Dimensions() int
}
```

One vector per image, in input order, all `Dimensions()` long, unit length,
compared by cosine. Images have no role: the model is symmetric. A type may
implement both `Embedder` and `ImageEmbedder` (a two-tower model), which is
why the method is not called `Embed`.

### Port: `media.Sampler`

```go
type Frame struct { Image image.Image; At time.Duration }
type Sampler interface { Sample(ctx context.Context, path string) ([]Frame, error) }
type SamplerFunc func(ctx context.Context, path string) ([]Frame, error)
```

A video is a file path: containers like MP4 need seeking, so a reader is not
enough. The sampler decides which frames stand for the video and how many at
most. An audio-only or unreadable file is an error, not an empty result.

### `media/ffmpeg`

`ffmpeg.New(Config)` returns a `Sampler`. One `ffprobe` call reads the
duration, then one `ffmpeg` run decodes the video and writes the chosen
frames as raw RGBA to stdout:

```
ffmpeg -nostdin -v info -nostats [-skip_frame nokey] -i <path>
  -vf "select='<expr>',scale=<fit>,showinfo" -fps_mode vfr -frames:v <max>
  -f rawvideo -pix_fmt rgba pipe:1
```

- Selection: a frame is taken when it is the first, or it scores above the
  scene-change threshold and at least `gap` after the previous taken frame,
  or `MaxInterval` has passed since the previous taken frame.
  `gap = max(MinInterval, duration/MaxFrames)` bounds the count from above
  whatever the cut rate; `-frames:v MaxFrames` is the hard stop. `MaxInterval`
  is raised to `gap` when `gap` is larger.
- `KeyframesOnly` adds `-skip_frame nokey`: only I-frames are decoded, which
  makes long videos cheap; encoders put I-frames at cuts and at a fixed
  interval, so the sample is still representative. Scene scoring is off in
  that mode (consecutive keyframes always differ).
- `scale` fits the longer side into `MaxSide` (default 512) keeping the
  aspect ratio; `showinfo` logs every frame that passed the filters, in
  order, with `pts_time:` and `s:WxH`. stdout holds the frames back to back;
  stderr is parsed after exit and the first `len(stdout)/frameBytes` showinfo
  lines give each frame its time and size. Frames are `*image.RGBA` over the
  bytes, no copy.
- Defaults: `SceneThreshold 0.4`, `MinInterval 1s`, `MaxInterval 5s`,
  `MaxFrames 64`, `MaxSide 512`, binaries `ffmpeg` and `ffprobe` on PATH.
- Needs ffmpeg 5.1 or newer (`-fps_mode`). A failing ffmpeg returns an error
  carrying the tail of its stderr.

### `search.MediaIndex`

```go
ix, err := search.NewMediaIndex(emb, store, "tenant-1", "space-7", "media",
    search.WithSampler(ffmpeg.New(ffmpeg.Config{})), search.WithMediaMinScore(0.6))
err = ix.IndexImage(ctx, "photo-42", reader)        // any registered image format
err = ix.IndexVideo(ctx, "clip-7", "/media/clip.mp4") // through the sampler
err = ix.IndexFrames(ctx, "x", contentHash, frames)  // the caller's own frames
hits, err := ix.SearchImage(ctx, reader, 10)         // []search.MediaMatch
hits, err := ix.SearchVideo(ctx, "/media/query.mp4", 10)
hits, err := ix.SearchFrames(ctx, frames, 10)
err = ix.Delete(ctx, "photo-42")
```

- Storage: one `vector.Chunk` per frame; an image is one frame at 0.
  `Chunk.Seq` is the frame time in milliseconds, `Chunk.Text` is empty. The
  index/partition/collection coordinates are the caller's, as in `Index`; an
  image space must not share a collection with a text space (different
  models, different vectors).
- Content hash: `IndexImage` hashes the bytes it reads (SHA-256, hex) and
  skips an unchanged image; `IndexVideo` hashes the file. `IndexFrames`
  takes the caller's hash; an empty hash always indexes.
- Zero frames store the hash with no chunks (as `Index` does for empty text):
  the item is known and unfindable, and is not retried on every run.
- Search: every query frame is embedded; each vector asks the store for
  `limit*4` chunks; hits are merged per document keeping the best score, with
  `At` (the indexed frame's time) and `QueryAt` (the query frame's time) of
  that best pair; sorted by score, then id; cut at `limit`; dropped below the
  floor. `WithMediaMinScore` is the same policy knob as `WithMinScore`, with
  the same no-floor default.
- A query that is itself indexed comes back first with a score near 1; the
  caller drops its own id.
- Image formats: `search` registers JPEG, PNG, GIF (stdlib) and WebP, BMP,
  TIFF (`golang.org/x/image`); an application registers more with
  `image.RegisterFormat`.

```go
type MediaMatch struct {
    DocID   string
    Score   float64
    At      time.Duration // best-matching indexed frame; 0 for an image
    QueryAt time.Duration // the query frame it matched; 0 for an image query
}
```

### `provider/onnx`

```go
emb, err := onnx.New(ctx, onnx.Config{
    ModelsDir:     "data/models",
    LibDir:        filepath.Join("data", "onnxruntime", onnx.LibVersion),
    ManagedLibDir: true,
    Provider:      onnx.ProviderCPU, // or ProviderCUDA, ProviderCoreML
})
```

- `Config`: `ModelsDir`, `Model` (a pinned name, `""` = `DefaultModel`),
  `LibDir`, `ManagedLibDir`, `Provider`, `DeviceID`, `Threads` (intra-op, ≤ 0
  = at most four), `BatchSize` (images per run, ≤ 0 = 8), `Logger`.
- `ModelSpec` (like `yzma.ModelSpec`): `Name`, `File`, `URL`, `SHA256`,
  `Size`, `Input` (tensor name, default `pixel_values`), `Output` (default
  `pooler_output`), `Size` (224), `Resize` (`squash` | `shortest-edge-crop`),
  `ShortestEdge` (256 for the crop mode), `Mean`, `Std` (per channel),
  `Interpolation` (`bilinear` | `bicubic`), `Dims`. `Spec(name, modelsDir)`
  returns the pinned ones; `Open(ctx, spec, Runtime)` runs any.
- Pinned models (digests from Hugging Face LFS metadata, files from the
  `onnx-community` Optimum exports):
  - `google/siglip2-base-patch16-224` (default): `vision_model.onnx`, fp32,
    371,807,752 bytes, squash to 224, bilinear, mean/std 0.5, 768-d.
    Apache-2.0. Its vectors share a space with the text tower, so text
    queries can be added later.
  - `facebook/dinov2-base`: `model.onnx`, fp32, 346,608,174 bytes, shortest
    edge 256 then centre crop 224, bicubic, ImageNet mean/std, 768-d.
    Apache-2.0. Better for instance-level and near-duplicate matching.
- Pinned runtime: ONNX Runtime 1.31.1 (`LibVersion`), the version
  `onnxruntime_go v1.38.0`'s headers target. Managed `LibDir` downloads the
  CPU build for linux/amd64, linux/arm64 and darwin/arm64, or the
  `gpu_cuda12` build when `Provider` is CUDA on linux/amd64, verified by the
  digests GitHub publishes for the release assets; `lib/libonnxruntime*` are
  extracted flat. An `onnxruntime-install.json` in the directory records
  version and variant; a managed directory with another variant is refilled,
  an operator directory with another one is refused. The loaded library must
  report version 1.31.x.
- The ORT environment is process-wide and loaded once (`runtimeLoader`, as in
  yzma): a second directory is refused, a failure is cached.
- Preprocessing in Go (`golang.org/x/image/draw`): draw into RGBA, resize per
  spec, `(v/255 − mean)/std` into a `[n,3,S,S]` float32 tensor. The scaler's
  kernels widen when downscaling, so results stay close to PIL's.
- Session: `NewDynamicAdvancedSession(file, [Input], [Output], opts)`; output
  dims read from `GetInputOutputInfo` and checked against `spec.Dims`; input
  and output names checked too, with the model's names in the error. One
  `Run` per batch; vectors normalized with `vector.Normalize`. Runs are
  serialized with a mutex.
- Providers: CPU (nothing appended), CUDA (`AppendExecutionProviderCUDA` with
  `device_id`), CoreML (`AppendExecutionProviderCoreMLV2`). Untested on a GPU
  in this build; the CPU path is tested with the real model.
- Seams for tests: `runtimeAPI` (load, version) and `sessionAPI` (open →
  `session` with `run`, `close`) with the only real implementation in
  `ort.go`; everything that decides something runs against fakes.

### `langaitest.FakeImageEmbedder`

Deterministic: a 8×8 grey thumbnail plus the mean colour, normalized. Resized
or recompressed copies score high, different pictures lower. It is the
embedder the `search` tests and an application's tests use.

## Constraints

- Go 1.26, module `github.com/go-bumbu/langai`. New modules:
  `github.com/yalue/onnxruntime_go v1.38.0` (MIT) and
  `golang.org/x/image v0.47.0` (BSD-3), both on the licence allowlist.
- `make lint` (golangci-lint v2.13.2: gocyclo ≤ 20, nestif ≤ 5, gosec, dupl)
  and `make coverage` (≥ 70 % per package) must pass. CI installs `ffmpeg` in
  the test job so `media/ffmpeg`'s tests run there.
- Git: one-line commits, `git add` then `git commit`, no push.

## Verification

- Unit tests with fakes for every package (`search` media index, `media/ffmpeg`
  command and parser, `provider/onnx` preprocessing, spec, fetch and the
  embedder over a fake session).
- `media/ffmpeg` end to end on synthetic videos it generates with ffmpeg
  (`lavfi` colour segments give hard cuts), skipped without ffmpeg.
- Real-model test (`make test-images`, `SEMSEARCH_MODELS=1`): downloads the
  runtime and the default model into `SEMSEARCH_ONNX_TEST_DIR`, embeds
  rendered pictures, and checks: unit length; a resized and a JPEG-recompressed
  copy score > 0.9 against the original; a different picture scores below
  both; the `MediaIndex` end to end with a `MemStore` and a generated video.

## Left to the builder

- Exact option and field names where this document does not fix them.
- The shape of the ffmpeg stderr parser and the fakes.
- README, TODO.md and Makefile wording.

# Image and video search implementation plan

> **For agentic workers:** built inline, task by task, test first. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Image-to-image and image-to-video similarity search as a library: `search.MediaIndex` over the existing `vector.Store`, an `ImageEmbedder` port with an ONNX Runtime provider (SigLIP 2 / DINOv2, CPU or CUDA), and a video frame sampler over the ffmpeg binary.

**Spec:** `docs/superpowers/specs/2026-11-05-image-video-search-design.md`.

**Tech Stack:** Go 1.26, `github.com/yalue/onnxruntime_go v1.38.0` (CGO), `golang.org/x/image v0.47.0`, ONNX Runtime 1.31.1, ffmpeg ≥ 5.1.

## Global constraints

- One-line commit messages; `git add` as a separate step; never push. Stage TODO.md with the task that changes it.
- `make lint` and `make test` green after every task; `make coverage` ≥ 70 % per package at the end.
- Every `//nolint` names its linter and gives a reason.

## Tasks

### Task 1: Ports — `langai.ImageEmbedder`, `media.Frame`/`Sampler`, `langaitest.FakeImageEmbedder`

- [x] `langai.go`: add `ImageEmbedder`; extend the package doc.
- [x] `media/media.go`: `Frame`, `Sampler`, `SamplerFunc`; package doc.
- [x] `langaitest/fake_image.go` + test: `FakeImageEmbedder` (4×4 RGB thumbnail centred on mid-grey, normalized); resized copy ≈ original, different picture lower.
- [x] Commit.

### Task 2: `media/ffmpeg` sampler

- [x] `media/ffmpeg/ffmpeg.go`: `Config`, defaults, `New`, `Sample`: probe duration (`ffprobe`), build the select expression and the command, run it, split stdout by the showinfo lines, return `[]media.Frame` of `*image.RGBA`.
- [x] Unit tests without the binary: expression and argument building for the defaults, the gap/MaxFrames arithmetic, keyframes-only mode; the stderr parser on recorded lines; frame splitting (including a short stdout and a mid-stream size change).
- [x] End-to-end tests (skip when ffmpeg is missing): a generated 10 s video with two hard cuts yields the cut frames at the right times, respects `MaxFrames`, scales to `MaxSide`; an audio-only file is an error; a cancelled context stops.
- [x] Commit.

### Task 3: `search.MediaIndex`

- [x] `search/media.go`: `MediaIndex`, `MediaOption` (`WithSampler`, `WithMediaMinScore`), `NewMediaIndex`, `IndexImage`, `IndexVideo`, `IndexFrames`, `Delete`, `SearchImage`, `SearchVideo`, `SearchFrames`, `MediaMatch`; image format registrations.
- [x] `search/media_test.go` with `FakeImageEmbedder`, `MemStore` and a `SamplerFunc`: index/search images; unchanged image skips embedding; update replaces frames; delete; video indexed as frames with `Seq` = ms; image query finds the video with `At`; video query finds image and video with `QueryAt`; min score; limit and dedupe; zero frames; errors (nil embedder, empty id, no sampler, undecodable image).
- [x] Commit.

### Task 4: `provider/onnx` — spec, preprocessing, fetch

- [x] `provider/onnx/onnx.go`: package doc, `DefaultModel`, `Provider`, `Resize`, `Interpolation`, `Pooling`, `Config`, `ModelSpec`, `Runtime`, validation.
- [x] `provider/onnx/models.go`: `Spec`, `DefaultSpec`, `Models`, the two pinned models.
- [x] `provider/onnx/preprocess.go`: draw into RGBA, resize per spec, tensor fill; tests on synthetic images (layout, normalization, squash vs crop geometry, sub-images and grey pictures).
- [x] `internal/fetch` (verified downloads, unpinned downloads, tarball extraction) shared by `provider/yzma` and `provider/onnx`; `provider/onnx/fetch.go`: pinned runtime assets per platform and variant, `ensureLibs`, `ensureModelFile`, `extractLibs` (only `lib/libonnxruntime*`), install record; tests over httptest and in-memory tarballs.
- [x] Commit.

### Task 5: `provider/onnx` — embedder over a session seam

- [x] `provider/onnx/backend.go`: `runtimeAPI`, `sessionAPI`, `session`, `runtimeLoader` (once, version lock, one path); `onnx.go`: `Open` flow, `Embedder` (`EmbedImages` batched, pooling, `Dimensions`, `Close`).
- [x] `provider/onnx/ort.go`: the only file importing `onnxruntime_go`: load, version, open (names checked, dims read), run (runtime-allocated output, shape returned), close, providers.
- [x] Tests with fakes: load once, version refused, second path refused, name/dims validation, batching across `BatchSize`, pooling, closed embedder, context cancellation, normalization.
- [x] `provider/onnx/real_test.go`: gated by `SEMSEARCH_MODELS` and `SEMSEARCH_ONNX_TEST_DIR`; downloads, checks dims, unit length, the similarity assertions, `MediaIndex` end to end, and a benchmark.
- [x] Ran the real tests locally for both pinned models. DINOv2's export has no `pooler_output`, only `last_hidden_state`: the spec gained `Pooling` (none, cls, mean) and the session returns the output's shape.
- [x] Commit.

### Task 6: Docs, Makefile, CI, TODO

- [x] README: packages table, layout tree (`provider/onnx`, `media`, `media/ffmpeg`, `internal/fetch`), requirements, a "Image and video search" section, development targets.
- [x] Makefile: `test-images` target.
- [x] `.github/workflows/validate.yml`: install ffmpeg in the test job; an `images-real` job with the cache keyed on `provider/onnx/fetch.go` and `models.go`.
- [x] TODO.md: mark the two search items done, add the follow-ups (EXIF orientation, text-to-image through the SigLIP 2 text tower, perceptual hash, GPU verification).
- [x] `make verify` (tests, licences, lint, benchmarks, coverage) passed; commit.

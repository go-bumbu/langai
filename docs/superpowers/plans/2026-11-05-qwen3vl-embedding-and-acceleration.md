# Qwen3-VL-Embedding and hardware-aware acceleration — implementation plan

> **For agentic workers:** built inline, task by task, test first. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `provider/yzma` runs Qwen3-VL-Embedding-2B in process as a text, image and video (segment) embedder in one space, picks the llama.cpp build and the GPU offload from the hardware, and the example service can switch to it and search by text.

**Spec:** `docs/superpowers/specs/2026-11-05-qwen3vl-embedding-and-acceleration-design.md`.

**Tech Stack:** Go 1.26, yzma v1.29.1 (`pkg/llama`, `pkg/mtmd`, no CGO), llama.cpp b11429 release assets (CPU, Vulkan, CUDA 13.4, macOS), ffmpeg for segment cuts.

## Global constraints

- One-line commit messages; `git add` as a separate step; never push. Stage TODO.md with the task that changes it.
- `make lint` and `make test` green after every task; `make coverage` ≥ 70 % per package at the end.
- No new entries in `go.mod`.

## Tasks

### Task 1: `internal/hw` — what the machine has

- [x] `hw.go`: `Accelerator` (`cuda`, `rocm`, `vulkan`, `metal`, `cpu`), `Finding{Accelerator, Reason}`, `Report{Findings []Finding}` in preference order with `Best()`, `Detect()` over the real filesystem and PATH, `detect(probes)` for tests. Probes: NVIDIA driver (`/proc/driver/nvidia/version`, `nvidia-smi`), ROCm (`/opt/rocm`, `rocminfo`), Vulkan ICD (`/usr/share/vulkan/icd.d`, `/etc/vulkan/icd.d`, `$XDG_DATA_DIRS`, `VK_ICD_FILENAMES`/`VK_DRIVER_FILES`), Metal (darwin/arm64).
- [x] Tests with a fake filesystem and PATH: each probe, the order, a machine with nothing, the reasons.
- [x] Commit.

### Task 2: `provider/yzma` — acceleration

- [x] `fetch.go`: `libAssets` keyed by `GOOS/GOARCH/backend`; the Vulkan, CUDA, arm64 and macOS assets with digests; `libRecordFile` (`langai-libs.json`: tag, backend) written on a managed install; `ensureLibsFor` takes the wanted backend, reinstalls a managed CPU directory when a GPU backend is wanted, keeps a GPU install for a CPU request; `wantLib` keeps `libmtmd`.
- [x] `accel.go`: `Accel` type and constants; `chooseBackend(accel, hw.Report, pinned)`: auto picks the best pinned backend the hardware has, explicit requires it; `resolveDevices(accel, devices)` after load: all layers on the first GPU device for auto, the named backend's device for explicit (error when absent), CPU otherwise; the log line.
- [x] `backend.go`/`llama.go`: `runtimeAPI.devices()` (ggml device count, type, backend name, description); `loadModel(path, gpuLayers)`; `Config`/`Runtime` gain `Accel` and `GPULayers`; `openLlama` and `openCompleter` thread them through.
- [x] Tests: asset choice per hardware, the reinstall rule, device resolution with fakes, the log line; the existing fakes gain `devices`.
- [x] Commit.

### Task 3: ports and `search.MediaIndex`

- [x] `media`: `Segment`, `VideoEmbedder`, `Cutter` (duration and stream-copy cut) ports.
- [x] `media/ffmpeg`: `Sampler` implements `Cutter` (`Duration`, `Cut`), e2e test on a generated clip.
- [x] `search`: `WithVideoEmbedder`, `WithTextEmbedder`, `SearchText`; `IndexVideo` stores segments when a `VideoEmbedder` is set; tests with fakes.
- [x] Commit.

### Task 4: `provider/yzma.Vision`

- [x] `vision.go`: `VisionConfig` (Config, Instruction, ImageMinTokens, ImageMaxTokens, ContextTokens, SegmentLength, FPS, MaxSegments, Cutter), `DefaultVisionModel`, the pinned spec (Q8_0 + f16 projector), `NewVision`/`OpenVision`, `Vision` implementing `langai.Embedder`, `langai.ImageEmbedder`, `media.VideoEmbedder`; the prompt through the model's chat template; serialization; `Close`.
- [x] `vision_ffi.go`: the `visionAPI` seam over `pkg/mtmd` (projector load, bitmap from RGB, bitmap from a video file at a target fps, tokenize with markers, eval chunks, pooled embeddings, frees).
- [x] Tests with a fake `visionAPI`: prompt layout, one vector per image, frees on every path, segment cutting and times, the token budget check, errors.
- [x] `vision_real_test.go` (`SEMSEARCH_MODELS=1`): a photo, its resized copy, a text query, a two-segment generated clip; `Makefile` `test-models` includes it.
- [x] Commit.

### Task 5: the example

- [x] `-model qwen3-vl-embedding-2b`, `-accel`; wiring with `yzma.NewVision`; info bar with the acceleration and the segment length; text search box; segment hits in the page; `prepare.sh` faststart.
- [x] README (root and examples), TODO.md (CUDA untested, Matryoshka, 8B), `make verify`.
- [x] Commit.

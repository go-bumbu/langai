# Qwen3-VL-Embedding and hardware-aware acceleration

Date: 2026-11-05. Status: approved by the author on the summary ("Build it")
and built; every choice is the builder's unless marked *given*.

## Intent

*Given:* run Qwen3-VL-Embedding, a model that puts text, images and videos in one
vector space, inside the process, and make the example service use it. *Given:*
rather than a backend flag per machine, a generic approach that looks at the
hardware and picks and configures the right acceleration itself.

The spike (`llama-server` from the pinned llama.cpp build b11429, community GGUF
of `Qwen/Qwen3-VL-Embedding-2B`) settled the open questions:

- Quality is excellent and self-consistent: copies of a photo score 0.97 to 0.98,
  unrelated photos about 0.10, a pug against another pug 0.77 to 0.81; six text
  queries each ranked the right photo first at 0.61 to 0.67. Scores agree to
  within 0.005 between f16 and Q8_0 and between CPU and Vulkan.
- Speed is the problem: an 800×600 photo is about 480 tokens (a 1080×1350 one
  about 1,400) and takes 7 to 8 s on this CPU (16 threads), 1.5 to 2 s on the
  laptop's Radeon 890M through Vulkan; an 8 s clip at 1 fps is 1,500 tokens,
  22 s on the CPU, 4.5 s on Vulkan. Text takes 0.1 s.
- A video is one vector for the whole input. At 1 fps a two-minute clip is more
  than 16k tokens, so videos must go through the model in segments. A clip
  handed over as bytes must be a faststart MP4 (ffprobe reads it from a pipe);
  by path there is no such limit.

## Goals

- One `provider/yzma` object that is a `langai.Embedder` (text), a
  `langai.ImageEmbedder` (images, so `search.MediaIndex` works unchanged) and a
  `media.VideoEmbedder` (whole segments of a video), all in one space.
- Text-to-image and text-to-video search through the existing `MediaIndex`.
- `provider/yzma` picks and configures acceleration from the hardware: the
  right pinned llama.cpp build is downloaded for the machine, the GPU is used
  when it is there and the CPU when it is not, and the log says which.
- The example gains a model switch (SigLIP 2 frames or Qwen3-VL segments) and a
  text search box; everything else on the page keeps working.

## Non-goals

- Changing `provider/onnx`'s explicit `Provider` setting; it gains the same
  `Auto` later, through the same detection package, as a follow-up.
- Reranking, the 8B model, Matryoshka truncation (the full 2048 dims are stored;
  a `Dims` option can come later since the vectors only need slicing and
  re-normalizing).
- Bit-exact agreement with the Hugging Face reference: the vectors are used
  within one index. A community fork claims upstream's Qwen3-VL vision path sits
  at about 0.97 cosine to the reference; that is fine here.
- Windows, ROCm, SYCL, OpenVINO assets: the detection names them but only the
  assets listed below are pinned. An unpinned backend falls back to the CPU.

## Anti-goals

- A second runtime: everything runs through the llama.cpp libraries yzma already
  loads. No CGO for this provider.
- Shelling out to `llama-server`: the spike used it, the provider does not.
- Changing the `vector.Store` port.

## Architecture

```
langai/                 Embedder, ImageEmbedder (as today)
├── media/              + VideoEmbedder port: path → one vector per segment, with times
├── search/             MediaIndex: + WithVideoEmbedder, + SearchText; IndexVideo picks
│                       segments (VideoEmbedder) or frames (Sampler + ImageEmbedder)
├── provider/yzma/      + Vision: Qwen3-VL-Embedding over libmtmd; + Accel in Config;
│                       libmtmd and the GPU builds among the pinned assets
├── internal/hw/        what the machine has: Vulkan, CUDA, ROCm, Metal, and why
└── examples/           model switch, text search box, segment hits
```

### Acceleration: `internal/hw` and `yzma.Config.Accel`

llama.cpp is built as a CPU library plus optional backend libraries
(`libggml-vulkan.so`, `libggml-cuda.so`, Metal is inside the macOS build), and
`ggml_backend_load_all_from_path`, which yzma calls at load, registers every
backend library in the directory whose driver answers. A GPU build whose driver
is missing simply registers no GPU device. So the generic mechanism is:

1. **Decide which build to install** (managed `LibDir` only). `internal/hw`
   reports, in order of preference per OS: CUDA (an NVIDIA driver: `nvidia-smi`
   on PATH or `/proc/driver/nvidia/version`), ROCm (`/opt/rocm` or
   `rocminfo`), Vulkan (an ICD under `/usr/share/vulkan/icd.d` or
   `/etc/vulkan/icd.d`, or `libvulkan.so.1` loadable), Metal (darwin/arm64,
   always). Each finding carries a one-line reason for the log.
2. **Pin the assets.** `provider/yzma/fetch.go` grows from one asset per
   GOOS/GOARCH to one per (GOOS, GOARCH, backend): linux/amd64 `cpu`,
   `vulkan` (32 MB) and `cuda-13.4` (152 MB, needs the host's CUDA 13 runtime;
   the `cudart-` bundle is not pinned), linux/arm64 `cpu` and `vulkan`,
   darwin/arm64 (Metal built in). The install record gains the backend, so a
   directory installed for the CPU is upgraded when a GPU shows up and the
   other way round is a no-op (the GPU build is a superset).
3. **Use what registered.** After `llama.Load`, the provider enumerates the
   ggml devices (`ggml_backend_dev_count`, `dev_type`, `dev_description`); with
   a GPU device present and `Accel` not `CPU`, `ModelParams.NGpuLayers` is set
   to all layers and the projector is offloaded too. The log line names the
   device (`Vulkan0: AMD Radeon 890M Graphics`) or says `CPU, N threads`.

```go
type Accel string
const (
    AccelAuto   Accel = ""        // pick from the hardware; CPU when nothing is found
    AccelCPU    Accel = "cpu"     // never offload, never install a GPU build
    AccelVulkan Accel = "vulkan"  // require this backend: install it, fail if no device registers
    AccelCUDA   Accel = "cuda"
    AccelMetal  Accel = "metal"
)
```

`Accel` sits in `yzma.Config` next to `Threads`, with `GPULayers int` (0 means
all when a GPU is used) for the few who want a split. The text `Embedder` and
the `Completer` get the acceleration for free.

### `provider/yzma.Vision`: Qwen3-VL-Embedding

```go
v, err := yzma.NewVision(ctx, yzma.Config{...}) // Model: yzma.DefaultVisionModel
v.Embed(ctx, []string{"a walrus on a beach"}, langai.RoleQuery) // langai.Embedder
v.EmbedImages(ctx, images)                                     // langai.ImageEmbedder
v.EmbedVideo(ctx, "/media/clip.mp4")                           // media.VideoEmbedder
v.Dimensions() // 2048
```

- **Model.** `Qwen/Qwen3-VL-Embedding-2B` (Apache-2.0), pinned as the
  DevQuasar Q8_0 GGUF (1.83 GB) plus the f16 projector (0.82 GB), by digest,
  like the other pinned models. The spec records the Hugging Face files, the
  upstream model name and the two digests; `Spec` keeps accepting other
  conversions.
- **Prompt.** The model's own chat template from the GGUF: a system turn with
  the instruction (`Represent the user's input.` by default,
  `VisionConfig.Instruction` overrides it), a user turn holding the media
  marker and/or the text, the assistant prompt, and the embedding is the last
  token's hidden state (`pooling last`), L2-normalized. One format for text,
  image and video, so they share the space.
- **Images.** `EmbedImages` turns each `image.Image` into an RGB bitmap
  (`mtmd.BitmapInit`), tokenizes it with the prompt (`mtmd.TokenizeFromParts`),
  evaluates the chunks (`mtmd.HelperEvalChunks`) in a context with embeddings
  on and last pooling, reads `llama.GetEmbeddingsSeq`, clears the KV cache.
  One image per sequence; `BatchSize` images per decode when they fit.
  Images larger than the model's token budget are scaled down first; the
  budget is `VisionConfig.ImageMaxTokens` (default 1,024, the floor llama.cpp
  warns about for Qwen-VL).
- **Videos.** `EmbedVideo(ctx, path)` cuts the file into segments of
  `VisionConfig.SegmentLength` (default 20 s, about 3,800 tokens at 1 fps) with
  ffmpeg stream copy into a temp directory (no re-encode; `media/ffmpeg` gets
  a `Cut` helper), runs each through `mtmd_helper_video_init` (path based,
  `FPSTarget` 1, timestamps off), tokenizes and evaluates it like an image, and
  returns one `media.Segment{Start, End time.Duration; Vector}` per segment. A
  short clip is one segment. `VisionConfig.FPS` and `MaxSegments` bound the
  cost.
- **Context.** One `llama.Context` sized for the largest segment
  (`SegmentLength × FPS × tokens per frame`, rounded up, 16k by default), KV
  cache cleared between inputs; the embedder serializes calls, as the text
  embedder does.
- **Fetch.** `wantLib` stops skipping `libmtmd.so`. The first real-model test
  downloads about 2.7 GB on top of today's models.

### Ports and `search.MediaIndex`

```go
// media
type Segment struct { Start, End time.Duration; Vector []float32 }
type VideoEmbedder interface {
    EmbedVideo(ctx context.Context, path string) ([]Segment, error)
    Dimensions() int
}
// search
search.WithVideoEmbedder(v media.VideoEmbedder) MediaOption
ix.SearchText(ctx, "a walrus on a beach", 10) ([]MediaMatch, error) // needs an Embedder in the same space
search.WithTextEmbedder(e langai.Embedder) MediaOption
```

- `IndexVideo` with a `VideoEmbedder` stores one chunk per segment, `Seq` =
  the segment's start in milliseconds; without one it samples frames as today.
  A `MediaMatch.At` is then the segment start, and the page says *matched in
  the segment starting at 20 s*.
- `SearchText` embeds the query with the text embedder (`RoleQuery`) and runs
  the same per-chunk search; it fails clearly when no text embedder is set.
  The option is separate from the image embedder on purpose: SigLIP 2 would
  take a text tower later, Qwen passes the same object twice.
- A `MediaIndex` built with a `VideoEmbedder` must not share a collection with
  one built on frames: different vectors for the same video. The example keeps
  one database per model, as today.

### The example

- `-model` accepts `qwen3-vl-embedding-2b` next to the two ONNX models;
  `make run MODEL=qwen3vl`. The wiring switches: Qwen builds one `yzma.Vision`
  and passes it as image, video and text embedder with `Accel` from
  `-accel` (default auto); the ONNX models keep the frame route.
- A text box in the header: *search by text*, same result pane, no preview.
- The info bar says the acceleration in use (`Vulkan0: AMD Radeon 890M` or
  `CPU · 16 threads`) and, for Qwen, the segment length.
- A video item's detail says *segments* instead of *frames the index saw*:
  the storyboard is still the sampled frames, with the segment boundaries
  marked.
- `prepare.sh` writes faststart MP4s (`-movflags +faststart`); harmless for
  the path route, and the files play sooner in the browser.

## Constraints

- The real-model tests (`make test-models`) gain `TestVisionReal`: a photo and
  its copies, a text query, a generated two-segment clip; they are skipped
  without `SEMSEARCH_MODELS=1`. The unit tests fake the yzma API as the
  existing ones do.
- `make verify` stays green; coverage ≥ 70 % per package; no new entries in
  `go.mod` (yzma's `pkg/mtmd` is part of the module already required).
- The CUDA asset is pinned but not run here (no NVIDIA GPU); TODO.md records
  it, like the ONNX CUDA path.

## Left to the builder

- The exact detection probes per OS and their order.
- Segment cutting details (keyframe alignment, the last short segment).
- How much of the ggml device API yzma needs beyond what it binds.

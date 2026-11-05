<!-- todo:guide — managed by todo; this block is rewritten on save. Docs: https://github.com/candy-tools/todo
This file is a todo list managed by "todo", a terminal TODO app:
https://github.com/candy-tools/todo

todo watches this file and reloads it automatically when it changes on disk, so
you — human or agent — can edit it directly in any editor. Keep to this format
so todo can parse what you write:

  # Heading           Headings ("#" to "######") are categories; they nest by
                      heading level.
  - [ ] Open task     A "- [ ]" line is an open task; "- [x]" marks it done.
  - [/] In progress   "- [/]" flags a task in progress, "- [>]" defers it.
  - [x] Done task     Tasks must live under a category heading.
    - [ ] Subtask     Indent by two spaces to nest a subtask under a task.
    Description text  An indented, non-checkbox line is the task's description.

Notes for editors:
- Text above the first heading (this block included) is preserved on save.
- todo rewrites the file into the canonical form above on every change, so any
  other free-form markdown placed between items is not kept.

Prefer the CLI over editing by hand: it writes this file in the exact format
above. `todo add`, `todo done|progress|defer|reopen`, `todo edit`, `todo rm`
and `todo list --json` cover the common actions — run `todo --help`.
-->

# TODO

## tagging

- [ ] judge tags against parts of a page
  A page is one blended embedding of its first ~2 KB, so a second or side topic
  is often missed: stitched pages of two single-tag golden documents get both
  tags 0.59 of the time (60 of 102, TestGoldenTagging), a minor side topic
  rarely.
- [ ] sub-centroids for tags that mix topics
  A tag whose examples cover unrelated topics averages them into one centroid
  that matches neither well. Keep up to 3 sub-centroids per tag, chosen by an
  eval scenario that includes parent-tag merges.
- [ ] use rejections as negative evidence
  A caller can keep a rejected tag off one document, but the Tagger learns
  nothing from it, so a wrong tag can return on similar documents. A document
  that fits no tag gets one at most about 1 time in 4 (0.75 stay untagged on
  the golden set, TestGoldenTagging, with only the document's own tags removed:
  an upper bound, as some of the picks are related tags from its family).
- [ ] real multi-tag documents in the golden set
  TestGoldenTagging's two-topic check stitches pairs of single-tag documents
  into one page (60 of 102 got both tags), which only stands in for a real
  page about two topics. Add real documents that carry several tags and drop
  the stitching.
- [ ] make the "fits no tag" scenario drop the whole tag family
  TestGoldenTagging's untagged scenario removes only a document's own tags, so
  the rest of its tag family (parent, siblings or children) stays in the scope
  and some of the picks are related tags, not unrelated ones. Remove the whole
  family, re-measure, and only then quote the result as a wrong-tag rate and
  set a new floor; until then 0.75 untagged only bounds that rate from above
  (at most about 1 document in 4 gets a wrong tag).

## upstream

- [ ] jupiterrider/ffi: load libffi lazily
  ffi opens libffi.so.8 in init() and panics if it is missing, which makes
  libffi8 a startup requirement of every binary that imports provider/yzma,
  even one that never embeds anything. Ask for a lazy load that returns an
  error on first use instead.
- [ ] yzma: bind ggml_set_abort_callback
  llama.cpp aborts the process on a failed assertion (e.g. a batch larger than
  n_ubatch); the symbol exists in build b11429, and a binding would let
  provider/yzma turn such asserts into errors. Until then the package
  validates every batch itself.

## image/vide

- [x] image to image proximity search
  Done: langai.ImageEmbedder, provider/onnx (ONNX Runtime through
  yalue/onnxruntime_go, cgo; SigLIP 2 base pinned as the default, DINOv2 base
  as the other pinned model) and search.MediaIndex over the existing Store.
  Not for a commercial app: OpenAI CLIP (card puts deployed use out of scope),
  DINOv3 (custom license, needs legal review).
- [x] video to vide proximiuty search
  Done the cheap way: media/ffmpeg samples frames (scene cuts plus an
  interval, or keyframes only), each is embedded with the image model as one
  Chunk (Seq = milliseconds), and an item is scored by its best frame against
  any query frame, so images and videos share one index and find each other.
  Store still has no get-vector-by-docID, so a query that is itself indexed
  comes back first and the caller drops its own id. Video-native models
  (Qwen3-VL-Embedding 2B, V-JEPA 2, X-CLIP, InternVideo2) remain an option
  for whole-clip matching; their Go runtime was not checked. For exact
  near-duplicates DINOv2 or a perceptual hash fits better than SigLIP.
  Not for a commercial app: VideoMAE v2 (CC-BY-NC-4.0).
- [ ] honour EXIF orientation when embedding JPEGs
  Go's image/jpeg ignores the orientation tag, so a phone photo stored
  rotated is embedded rotated and matches its upright video frames less. A
  small APP1/TIFF parser for tag 0x0112 plus a rotate before embedding, in
  search's decodeImage.
- [ ] run provider/onnx on a GPU
  The CUDA (gpu_cuda12 build, device_id) and CoreML paths are wired but were
  only unit-tested with a fake session; the real-model tests ran on a CPU.
  Check the CUDA provider loads, that batches speed up, and the yalue note on
  CUDA overwriting Go's signal handlers (onnxruntime_go issue 140).
- [ ] perceptual hash for exact and near-exact duplicates
  A 64-bit dHash/pHash as a second, model-free ImageEmbedder (±1 per bit, so
  cosine = 1 - 2·hamming/64 and the same Store works) for de-duplication
  jobs that should not pay for a model.
- [x] text to image / video proximity search
  Done with Qwen3-VL-Embedding-2B (Qwen/Qwen3-VL-Embedding-2B, Apache-2.0,
  2048-d) as yzma.Vision over llama.cpp's libmtmd: text, images and video
  segments in one space, MediaIndex.SearchText and WithVideoEmbedder. The
  pinned GGUF is DevQuasar's Q8_0 with the f16 projector (no official GGUF).
  Upstream llama.cpp's Qwen3-VL vision path sits at about 0.97 cosine to the
  Hugging Face reference (a community fork claims 0.9999 with extra patches);
  fine within one index, not for vectors made elsewhere. The SigLIP 2 text
  tower (text_model.onnx, SentencePiece tokenizer in Go) stays an option for a
  lighter text-to-image route on the ONNX side. Adult content: neither the
  paper nor the card mentions filtering; quality on explicit material is
  untested.
- [ ] vision model follow-ups
  Matryoshka: the model's vectors can be cut to 64..2048 and re-normalized;
  a VisionConfig.Dims would shrink the store. The 8B model (MMEB-V2 77.8 vs
  73.2) and the Qwen3-VL-Reranker as a second stage are untried. The
  instruction is one per Vision; per-call instructions (a query instruction
  that differs from the document one) would need a second prompt path.
  ImageMaxTokens did not change the token count in the spike (484 for
  800x600, 1,437 for 1080x1350): check what the projector's bounds do.
- [ ] run provider/yzma on CUDA and on macOS
  Accel auto picks the build from the hardware and offloads to the GPU that
  registers; Vulkan on an AMD 890M iGPU was run (1.5 s per photo vs 7 s on
  the CPU). The CUDA 13.4 build (needs the CUDA runtime on the host) and the
  Metal build are pinned and chosen but were not run. ROCm is detected but
  has no pinned build (AMD cards go through Vulkan); Windows has none.
- [ ] CI for the vision model
  make test-vision downloads 2.7 GB and needs ffmpeg; it is not in
  validate.yml. A cached job like embeddings-real would cover it.
- [ ] iamge Auto Tagging
  Two routes. Caption then tag: a captioner writes text and the existing Tagger
  takes that text. That needs libmtmd, which wantLib (provider/yzma/fetch.go)
  skips today; the pinned llama.cpp build is b11429.
  Moondream 2 (vikhyatk/moondream2): 1.9B, Apache-2.0 on the HF card. The
  vendor page says Moondream model license 1.0: commercial use allowed, no
  general-purpose hosted model service. ggml-org/moondream2-20250414-GGUF is
  F16 only, 2.84 GB + 910 MB mmproj. Likely works on yzma, untested.
  Moondream 2 0.5B: 375 / 479 MiB download and 816 / 996 MiB RAM (int4 / int8),
  license 1.0 (HF card fetch failed, so taken from the vendor page). yzma
  support unknown.
  Qwen3-VL-2B-Instruct (Qwen/Qwen3-VL-2B-Instruct): Apache-2.0, 2B, takes image
  and video. It is the VLM yzma's README benchmarks (its CLI example uses
  Qwen2.5-VL-3B), so the safest fit on yzma. Quantized sizes not checked.
  Florence-2 base / large (microsoft/Florence-2-base, -large): MIT, 0.23B /
  0.77B, three caption lengths plus detection and OCR. Not supported by
  llama.cpp: ONNX or PyTorch only.
  Fixed tag lists skip the text step but return their own vocabulary, not the
  app's tags, so they need a mapping. RAM++ (4,585 tags, Apache-2.0, 873 MB
  INT8 ONNX, ~1.8 GB RAM). WD v3 SwinV2 98M / EVA02-L 315M (Apache-2.0, ONNX
  467 MB / 1.26 GB, trained on Danbooru art, so it fits anime-style images).
  Both need onnxruntime, not yzma.
  Not for a commercial app: Moondream 3 Preview (BSL 1.1, bars selling hosted
  or embedded access), LLaVA 1.6 (card says research use, OpenAI-policy caveat
  on its training data).
- [ ] video auto tagging
  Dedicated video taggers were not researched. These reuse the models above.
  Frames: sample frames (1 FPS or scene cuts), run the image route on each and
  keep the tags that show up in enough frames (RAM++, WD v3, or a caption plus
  the Tagger).
  Whole clip: Qwen3-VL-2B-Instruct takes video (card), and yzma's pkg/mtmd has a
  video path that needs ffmpeg and ffprobe on PATH (read in its source, not
  run). That gives one caption per clip to feed the Tagger.
  Embedding route (untested): embed the clip with Qwen3-VL-Embedding 2B and
  compare it with tag texts or tag centroids built in the same space. X-CLIP
  base can also score a clip against tag texts (its card lists zero-shot video
  classification), but it only sees 8 frames at 224x224.
  Not for a commercial app: VideoMAE v2 (CC-BY-NC-4.0).

## API

- [ ] move towards an api service that can run stuff
- [ ] move to kronk sdk as provider
- [ ] make it possible to use local models or remote subscriptions

# expected fetaures

- [ ] text => existing tags
  - kronk + embedding model (EmbeddingGemma 300M)
  - optional: kronk + reranker model to check the shortlist
- [ ] text => new tags
  - kronk + LLM (Qwen3 4B Instruct) writes candidate tags
  - kronk + embedding model drops candidates too close to existing tags
- [x] image / video => proximity image / vide
  - onnxruntime (cgo) + SigLIP 2 base; kronk can't embed images
  - video: ffmpeg samples frames, each embedded as an image; best frame wins
- [ ] image / video => description
  - kronk + vision LLM + mmproj (Qwen3-VL-2B-Instruct)
  - video: ffmpeg pulls keyframes, sent as images (kronk doesn't decode video)
- [ ] image / video => tags
  - describe it (kronk vision LLM), then reuse the text => tags stacks
  - optional, skip the caption: SigLIP 2 scores the image against tag names
- [ ] text => similar image video
  - onnxruntime + SigLIP 2: the query text lands in the image/frame space
  - kronk only: embedding model over the descriptions (meaning, not looks)

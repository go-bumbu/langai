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
  A caller can keep a rejected tag off one document, but the Tagger learns nothing from it, so a wrong tag can return on similar documents. Re-measured 2026-11-09: with the whole tag family removed no golden document gets a wrong tag (0 of 84); with only its own tags removed 0.64 stay untagged and 0.81 get none but an ancestor. So on the golden set the wrong picks are related tags, and negative evidence would have to be judged on real rejections.
- [ ] real multi-tag documents in the golden set
  TestGoldenTagging's two-topic check stitches pairs of single-tag documents
  into one page (60 of 102 got both tags), which only stands in for a real
  page about two topics. Add real documents that carry several tags and drop
  the stitching.
- [ ] try Generator windows of 8-12 KB
  6 KB windows beat 2 KB ones (F1 0.84 vs 0.74 under the scoring before 2026-11-09) with fewer calls; larger windows need a bigger chat context and were not measured.
- [ ] re-measure NameMinScore at 0.43
  Short essences (names only 0.81 vs 0.79, mixed 0.868 vs 0.860) and long items (mean F1 0.87 vs 0.84 under the 2026-11-09 scoring) both score a little better at 0.43; stitched two-topic pages lose 0.04 (0.83 vs 0.87 with ChunkBytes) and long-member pages 0.03. Weigh it on real data before moving the default.
- [ ] turn ChunkBytes on by default
  Every golden essence is one piece at 2000 bytes, so the short numbers do not move, two-topic pages gain (0.80 to 0.87) and long items find side topics (0.65 to 0.90 under the 2026-11-09 scoring). Costs a re-embed of long items for current users, and a default that depends on the embedder's window.

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

- [ ] run provider/onnx on a GPU
  The CUDA (gpu_cuda12 build, device_id) and CoreML paths are wired but were
  only unit-tested with a fake session; the real-model tests ran on a CPU.
  Check the CUDA provider loads, that batches speed up, and the yalue note on
  CUDA overwriting Go's signal handlers (onnxruntime_go issue 140).
- [ ] vision model follow-ups
  Matryoshka: the model's vectors can be cut to 64..2048 and re-normalized; a VisionOptions setting for the output size would shrink the store (the model id already carries dims). The 8B model (MMEB-V2 77.8 vs 73.2) and the Qwen3-VL-Reranker as a second stage are untried. The instruction is one per Vision; per-call instructions (a query instruction that differs from the document one) would need a second prompt path. ImageMaxTokens did not change the token count in the spike (484 for 800x600, 1,437 for 1080x1350): check what the projector's bounds do. Short videos go to libmtmd's own ffmpeg, which has no protocol whitelist: routing them through the Cutter would close that.
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

## features

- [ ] hybrid search: keywords next to embeddings
  Search is embeddings only, which misses exact names, error codes, version numbers, rare words and one-word queries (ErrModelMismatch, pgx). Both backends have a keyword engine (FTS5 in SQLite, tsvector in Postgres): the store keeps a keyword index of the chunks it already holds, and Index.Search merges the two rankings with reciprocal rank fusion, which needs no score tuning. Covered by vectortest; shown on the text search page.
- [ ] topic discovery: clusters of untagged items as tag suggestions
  The Tagger learns only from tags people made and the Generator works one text at a time; nothing looks across the corpus ("14 untagged notes about sourdough: tag them cooking / bread?"). Cluster the stored vectors of untagged items, let the Completer name each cluster, check the name against the members as the Tagger blends name and members. Works for photos too (trips, events), as the duplicates page groups by hash. Eases the cold start SmoothingK only softens.
- [ ] grounded answers over indexed documents
  Retrieve with Index.Search, have the Completer answer only from those chunks, cite doc ids and heading paths. A capability next to tagging.Generator that takes a Completer and an Index. Keep it small: no agents, no conversation memory.
- [ ] query-time filters on document metadata
  Scoping a search by date, type or folder needs an allow-list of ids today, which does not scale. Small per-document attributes, filtered inside the store's one search ("photos from 2024", "only PDFs"). Wait for a consumer that needs it.

# expected fetaures

- [ ] image / video => description
  - kronk + vision LLM + mmproj (Qwen3-VL-2B-Instruct)
  - video: ffmpeg pulls keyframes, sent as images (kronk doesn't decode video)
- [ ] image / video => tags
  - describe it (kronk vision LLM), then reuse the text => tags stacks
  - optional, skip the caption: SigLIP 2 scores the image against tag names

# review 2026-11-09

## architecture

- [ ] one Chunker for search and tagging
  search.MarkdownChunker{MaxChars} and tagging.Chunk(text, maxBytes) differ in units and live apart; Tagger and Generator take only an int. Left open on 2026-11-09 on purpose: the two do different jobs (structure-aware retrieval chunks with a heading path vs even pieces for scoring), the tagging floors were measured on tagging.Chunk, and a shared port would need a re-measure to prove it changes nothing.

## search, vector, media

## tagging and eval

## providers and fetch

- [ ] phash: measure thresholds on real photos
  The duplicate thresholds come from synthetic pictures. Flat pictures all share the zero hash and a small mark on a near-flat picture flips about half the pHash bits. No ParseHash yet, and the vectors are not a compact bit column in the stores. grey.go has its own compose-on-white code that could use internal/imageutil if it gives identical 8-bit values.

## second pass

## found while building the example

- [ ] EmbeddingCache has no count or size
  The status page can only show the database file's bytes.
- [ ] no lazy VideoEmbedder; Vision.Fingerprint needs a loaded model
  The example keeps the vision model eager for this reason. A LazyVideoEmbedder and a fingerprint computed from spec and options (like ModelID) would let it load on first use.
- [ ] yzma.Completer has no Accel(); LazyCompleter has no Model()
  The status page cannot say what runs the chat model or name it before load.
- [ ] audio-only video fails with ffmpeg's error, not ErrNoFrames
  IndexVideo on an mp4 without a video stream surfaces ffmpeg's own message.
- [ ] phash.Hash does not turn EXIF-rotated JPEGs upright
  MediaIndex applies EXIF orientation in an unexported decode, so a raw phash.Hash of the same file can disagree with the indexed vector. From reading the code, not reproduced.

# change log

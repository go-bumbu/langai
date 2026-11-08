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
- [ ] cachetest: check that Get refreshes the last-used time
  The conformance suite never checks that Get rewrites an entry's last-used time once it is older than CacheRefreshAfter, so a backend that skips it still passes, and Prune then drops vectors still in use and they get embedded again. Needs an injectable clock or a time-shifted Prune cutoff in the suite.
- [x] per-chunk scoring for the Tagger's side topics
  Done 2026-11-07 (docs/superpowers/specs/2026-11-07-tagger-long-items-results.md): with ChunkBytes an item of 3+ pieces scores by the mean of its two best pieces, one of 2 by the best of its halves and their centroid, and a long member enters its tag's blend as its piece closest to the name. Long set: side 0.68 vs 0.58 (mean), precision 0.78 both, mention 0.02; pure max found 0.74 but 3.5 off-label tags a doc vs 2.0, and lost stitched two-topic pages (0.77 vs 0.80). Title-per-chunk added nothing. Cutoffs 0.42/0.58 hold; floors in TestTaggerLongTags.
- [ ] try Generator windows of 8-12 KB
  6 KB windows beat 2 KB ones (F1 0.84 vs 0.74) with fewer calls; larger windows need a bigger chat context and were not measured.
- [ ] eval.MatchTopics: containment sends side tags to the main topic
  A tag that contains a main alias ("housing cooperatives" holds "cooperatives") scores 1 against the main topic and ties go to the first topic, so a side topic picked by its exact label counts as missed: de-genossenschaften and es-cafe score side 0.00 with every side label picked. By exact label the Tagger's side recall is 0.89, not 0.68. Prefer the topic whose own label or alias is the longer containment match; re-measure the Generator and Tagger floors after.
- [ ] re-measure NameMinScore at 0.43
  Short essences (names only 0.81 vs 0.79, mixed 0.868 vs 0.860) and long items (F1 +0.02) both score a little better at 0.43; stitched two-topic pages lose 0.04 (0.83 vs 0.87 with ChunkBytes) and long-member pages 0.03. Weigh it on real data before moving the default.
- [ ] turn ChunkBytes on by default
  Every golden essence is one piece at 2000 bytes, so the short numbers do not move, two-topic pages gain (0.80 to 0.87) and long items find side topics (0.45 to 0.68). Costs a re-embed of long items for current users, and a default that depends on the embedder's window.

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

# review 2026-11-09

## architecture

- [ ] model identity so vector spaces cannot mix
  No port reports which model made a vector and the stores record only dims, so two 768-d models (SigLIP 2, EmbeddingGemma) can share a collection and return garbage with no error; TaggerConfig.Model is typed by hand. Add an optional langai.Identified{Model() string}, implement it in the providers and LazyEmbedder, keep a (model, dims) meta row per index in the stores and fail with vector.ErrModelMismatch.
- [ ] content hash covers the chunker and the model
  search.ContentHash(text) covers only the text, so a changed MaxChars, Chunker or embedder leaves stale chunks in place forever. Hash text + chunker fingerprint + model id.
- [ ] reshape the vector.Store port
  Methods take 4-6 positional strings, Search only wraps SearchScope, and listing doc ids, Count, deleting a whole index, dropping an old generation, batch upsert and a doc-id filter are missing. Introduce a Coord{Index, Partition, Collection}, batch Upsert/Delete, Docs, DeleteIndex, one Search with a scope filter, DropGeneration on the stores.
- [ ] vectortest conformance suite and missing fakes
  sqlitevec and pgvec repeat the same tests by hand and MemStore has its own. Add vector/vectortest.Run for all three, plus langaitest.FakeCompleter (stub chat models are written three times), a map-backed Catalog and a fake VideoEmbedder.
- [ ] read paths must not create tables
  ensure() runs CREATE TABLE from Delete/ContentHash/Chunks/Search in both SQL stores, so a search for an unknown index creates a table set and a read-only Postgres role fails on a lookup. Only Upsert creates the schema; reads return empty when the tables are absent.
- [ ] sentinel errors for errors.Is
  Closed-embedder, dimension-mismatch and empty-id errors are plain strings. Add langai.ErrClosed, vector.ErrDimensions, search.ErrEmptyID and wrap with %w.
- [ ] search API symmetry
  Index.Search returns the store's vector.Match (leaks Partition/Collection) while MediaIndex returns MediaMatch; Index has no SearchIndexed and MediaIndex.SearchIndexed returns the query item itself. Add search.Match, Index.SearchIndexed, and leave the query doc out of both.
- [ ] batch indexing
  Index.Index does one hash read, one Embed and one transaction per document. Add Index.IndexBatch that reads hashes once, embeds changed chunks in one batch and upserts in one call (needs the reshaped Store).
- [ ] ports in the right packages
  vector.Normalize is imported by the providers although vector is the store port, and VideoEmbedder/Segment live in media while the other embedder ports are in the root. Move Normalize, VideoEmbedder and Segment to the root package.
- [ ] LazyEmbedder lifecycle
  No Close (the loaded model can never be released), load takes no ctx, and there is no lazy form of ImageEmbedder, Vision or Completer. load becomes func(ctx), add Close, add LazyImageEmbedder and LazyCompleter over one loader.
- [ ] dedupe provider fetch code
  ensureModelFile, fetchFile and the staged library install are copied between provider/yzma/fetch.go and provider/onnx/fetch.go; yzma.Download is exported only for zarf/eval. Move the shared steps to internal/fetch.
- [ ] consistent provider constructors
  yzma/onnx Config repeat Runtime's fields, OpenVision breaks the Open(ctx, spec, Runtime) pattern, lookup names differ (Spec, LookupCompleterSpec, LookupVisionSpec), yzma has no Models(), and the cache constructors return the interface.
- [ ] one Chunker for search and tagging
  search.MarkdownChunker{MaxChars} and tagging.Chunk(text, maxBytes) differ in units and live apart; Tagger and Generator take only an int. One Chunker port with Markdown and Even implementations. The eval floors depend on tagging.Chunk, so re-measure.
- [ ] doc drift
  internal/tablename says tagging uses ForPrefix (dead outside its test); WithGeneration docs point consumers at an internal package; README's package tree and dependency rule are out of date.

## search, vector, media

- [ ] zero or NaN vector breaks every search in its scope
  A text with no letters or digits embeds to a zero vector with the fake embedder; once stored, every sqlitevec Search in that scope fails scanning a NULL distance, and on pg the NaN score passes the MinScore floor. Add vector.Validate (finite, non-zero), call it on index and query paths and in each store, and treat NaN as below the floor.
- [ ] limit*4 over-fetch returns far fewer items than limit
  One 64-frame near-static video fills all fetched chunks, so SearchIndexed/SearchFrames return one hit where 10 exist. The store needs a per-document best-chunk search (pg DISTINCT ON, sqlite/mem loop with doubling k) and an exclude-doc argument.
- [ ] empty partition or collection is a wildcard in Store.Search
  search.New(e, store, index, "", "notes") searches every partition and returns other tenants' docs. Search matches both coordinates exactly; only the explicit scope filter may leave one open.
- [ ] concurrent sqlitevec writes fail with database is locked
  The transaction starts with a SELECT and upgrades to a write, so 8 concurrent upserts fail 60-95% of calls even with _busy_timeout. Make the first statement a write, document the DSN, add a concurrent-upsert test.
- [ ] concurrent pgvec Upsert of one doc leaves duplicate chunk sets
  Under READ COMMITTED both transactions delete then insert and only serialize at the final docs upsert. Upsert the docs row first so its row lock serializes same-doc writers.
- [ ] LazyEmbedder ignores ctx during load and panics on a nil embedder
  Embed blocks in once.Do past its deadline during a multi-GB load; load returning (nil, nil) or panicking leaves a nil inner that every later Embed dereferences.
- [ ] sqlite caps k at 4096
  Index.Search with limit above 1024 errors on sqlitevec only. Clamp or page inside the store.
- [ ] dimension change accepted at New, fails later with a raw driver error
  New(ctx, db, 16) over an 8-dim store succeeds and every write and search then fails. Compare the existing dims in ensure and return an error naming WithGeneration.
- [ ] ffmpeg: path with a colon read as a protocol, out starting with - read as an option
  Sample(ctx, "clip:final.mkv") fails with Protocol not found. Pass file: URLs for inputs and outputs.
- [ ] ffmpeg Sample buffers every raw frame in one buffer
  Up to 67 MB by default and about 2 GB for unscaled 4K, and each frame aliases the shared buffer. Read frame by frame from the pipe with per-frame allocations.
- [ ] video with zero frames is stored as indexed and never retried
  An empty sampler or EmbedVideo result stores the hash with no chunks, so later IndexVideo calls are no-ops. Error on zero frames or segments.
- [ ] IndexImage reads the whole reader unbounded
  io.ReadAll runs before the pixel guard. Limit the reader.
- [ ] chunker misses paragraph breaks on CRLF and whitespace-only blank lines
  CRLF text is one paragraph and falls to splitWords, which flattens code and lists; the heading prefix is not counted toward MaxChars.
- [ ] NUL or invalid UTF-8 indexes on sqlite and fails on Postgres
  Sanitize before chunking.
- [ ] MemStore differs from the SQL stores
  Scores 0 on a dimension mismatch instead of an error, accepts an empty index, aliases the caller's vectors and orders ties randomly. Covered by the conformance suite: ties by score, then doc id, then seq.

## tagging and eval

- [ ] one failing member text makes a space un-rebuildable
  All single-piece texts go in one Embed call that returns nil on error, so nothing is cached and every Suggest re-embeds everything and returns ErrNotRebuilt. Embed in sub-batches, Put after each, isolate the bad text, and decide a policy for a poison member.
- [ ] a rebuild cannot be cancelled and a panic in it kills the process
  The flight runs under context.WithoutCancel with no recover and Tagger has no Close, so a hung embedder blocks callers and a rebuild can outlive the DB or model.
- [ ] same Model name with another dimension gives zero scores or a panic
  Cached vectors are never checked against Dimensions(); a ragged centroid indexes out of range. Treat a wrong-length hit as a miss, error on length mismatch in scores, refuse ragged input.
- [ ] Prune drops vectors that are still needed
  Only changed tags are rebuilt, so stable tags' entries never refresh their last-used time and Prune(now-30d) removes them. Add a way to retain every key of the current catalog before pruning; document the 24h refresh granularity.
- [ ] parseTags keeps reply scaffolding as tags
  JSON arrays, a Tags: label, think blocks and bold markers come back as tag text. Strip think blocks and fences, try JSON, cut labels, widen the trim sets.
- [ ] cleanPath drops an existing path the model reused
  A level over 4 words or 40 runes voids the whole path, and reused paths are not mapped back to the application's spelling.
- [ ] one failed window discards a whole long Generate
  No retry; an empty reply on the short path looks like no tags. Retry a window once and return partial results with a joined error.
- [ ] best-effort cache writes fail the request
  A failed last-used UPDATE fails Get and so Suggest although the vectors were read; a Put error discards freshly embedded vectors.
- [ ] non-finite vectors are cached forever
  embedBatch checks only lengths; one NaN member makes the tag never match until the cache is wiped.
- [ ] callers at two revisions re-read the catalog on every call
  Flights are keyed by state, so two revisions rebuild concurrently and the later Apply leaves an unrecorded state. Key the flight by space.
- [ ] eval.aliasSim matches inside words
  Tag ai scores 1 against container shipping; only the alias has a length floor. Match on word boundaries. Needs a re-measure of the long-set floors.
- [ ] long-set headline numbers hide conventions
  A zero-tag document counts as precision 0 and the F1 is that of the mean precision and recall, not the mean per-document F1.
- [ ] sqlcache: Postgres and SQLite differ behind the shared code
  Postgres rejects NUL or invalid UTF-8 in index or model; concurrent CREATE TABLE IF NOT EXISTS at startup can fail.

# change log

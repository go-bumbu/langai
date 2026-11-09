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
- [x] make the "fits no tag" scenario drop the whole tag family
  Done 2026-11-09: with the whole family removed 84 of 84 golden documents stay untagged (closest wrong tag 0.016 under its cutoff), floor 0.95. The old own-tags-only scenario is kept as 'own tags gone, none but ancestors' at 0.81, floor 0.75.
- [x] cachetest: check that Get refreshes the last-used time
  The conformance suite never checks that Get rewrites an entry's last-used time once it is older than CacheRefreshAfter, so a backend that skips it still passes, and Prune then drops vectors still in use and they get embedded again. Needs an injectable clock or a time-shifted Prune cutoff in the suite.
- [x] per-chunk scoring for the Tagger's side topics
  Done 2026-11-07 (docs/superpowers/specs/2026-11-07-tagger-long-items-results.md): with ChunkBytes an item of 3+ pieces scores by the mean of its two best pieces, one of 2 by the best of its halves and their centroid, and a long member enters its tag's blend as its piece closest to the name. Long set: side 0.68 vs 0.58 (mean), precision 0.78 both, mention 0.02; pure max found 0.74 but 3.5 off-label tags a doc vs 2.0, and lost stitched two-topic pages (0.77 vs 0.80). Title-per-chunk added nothing. Cutoffs 0.42/0.58 hold; floors in TestTaggerLongTags.
- [ ] try Generator windows of 8-12 KB
  6 KB windows beat 2 KB ones (F1 0.84 vs 0.74 under the scoring before 2026-11-09) with fewer calls; larger windows need a bigger chat context and were not measured.
- [x] eval.MatchTopics: containment sends side tags to the main topic
  Done 2026-11-09: a tag goes to the topic it names, then to the closer whole-word match. Tagger long set: main 1.000, side 0.90 (was 0.68), precision 0.80. Generator: side 0.76 (was 0.71), precision 0.89. Floors reset.
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
- [x] honour EXIF orientation when embedding JPEGs
  Go's image/jpeg ignores the orientation tag, so a phone photo stored
  rotated is embedded rotated and matches its upright video frames less. A
  small APP1/TIFF parser for tag 0x0112 plus a rotate before embedding, in
  search's decodeImage.
- [ ] run provider/onnx on a GPU
  The CUDA (gpu_cuda12 build, device_id) and CoreML paths are wired but were
  only unit-tested with a fake session; the real-model tests ran on a CPU.
  Check the CUDA provider loads, that batches speed up, and the yalue note on
  CUDA overwriting Go's signal handlers (onnxruntime_go issue 140).
- [x] perceptual hash for exact and near-exact duplicates
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

- [x] model identity so vector spaces cannot mix
  No port reports which model made a vector and the stores record only dims, so two 768-d models (SigLIP 2, EmbeddingGemma) can share a collection and return garbage with no error; TaggerConfig.Model is typed by hand. Add an optional langai.Identified{Model() string}, implement it in the providers and LazyEmbedder, keep a (model, dims) meta row per index in the stores and fail with vector.ErrModelMismatch.
- [x] content hash covers the chunker and the model
  search.ContentHash(text) covers only the text, so a changed MaxChars, Chunker or embedder leaves stale chunks in place forever. Hash text + chunker fingerprint + model id.
- [x] reshape the vector.Store port
  Methods take 4-6 positional strings, Search only wraps SearchScope, and listing doc ids, Count, deleting a whole index, dropping an old generation, batch upsert and a doc-id filter are missing. Introduce a Coord{Index, Partition, Collection}, batch Upsert/Delete, Docs, DeleteIndex, one Search with a scope filter, DropGeneration on the stores.
- [x] vectortest conformance suite and missing fakes
  sqlitevec and pgvec repeat the same tests by hand and MemStore has its own. Add vector/vectortest.Run for all three, plus langaitest.FakeCompleter (stub chat models are written three times), a map-backed Catalog and a fake VideoEmbedder.
- [x] read paths must not create tables
  ensure() runs CREATE TABLE from Delete/ContentHash/Chunks/Search in both SQL stores, so a search for an unknown index creates a table set and a read-only Postgres role fails on a lookup. Only Upsert creates the schema; reads return empty when the tables are absent.
- [x] sentinel errors for errors.Is
  Closed-embedder, dimension-mismatch and empty-id errors are plain strings. Add langai.ErrClosed, vector.ErrDimensions, search.ErrEmptyID and wrap with %w.
- [x] search API symmetry
  Index.Search returns the store's vector.Match (leaks Partition/Collection) while MediaIndex returns MediaMatch; Index has no SearchIndexed and MediaIndex.SearchIndexed returns the query item itself. Add search.Match, Index.SearchIndexed, and leave the query doc out of both.
- [x] batch indexing
  Index.Index does one hash read, one Embed and one transaction per document. Add Index.IndexBatch that reads hashes once, embeds changed chunks in one batch and upserts in one call (needs the reshaped Store).
- [x] ports in the right packages
  vector.Normalize is imported by the providers although vector is the store port, and VideoEmbedder/Segment live in media while the other embedder ports are in the root. Move Normalize, VideoEmbedder and Segment to the root package.
- [x] LazyEmbedder lifecycle
  No Close (the loaded model can never be released), load takes no ctx, and there is no lazy form of ImageEmbedder, Vision or Completer. load becomes func(ctx), add Close, add LazyImageEmbedder and LazyCompleter over one loader.
- [x] dedupe provider fetch code
  ensureModelFile, fetchFile and the staged library install are copied between provider/yzma/fetch.go and provider/onnx/fetch.go; yzma.Download is exported only for zarf/eval. Move the shared steps to internal/fetch.
- [x] consistent provider constructors
  yzma/onnx Config repeat Runtime's fields, OpenVision breaks the Open(ctx, spec, Runtime) pattern, lookup names differ (Spec, LookupCompleterSpec, LookupVisionSpec), yzma has no Models(), and the cache constructors return the interface.
- [ ] one Chunker for search and tagging
  search.MarkdownChunker{MaxChars} and tagging.Chunk(text, maxBytes) differ in units and live apart; Tagger and Generator take only an int. Left open on 2026-11-09 on purpose: the two do different jobs (structure-aware retrieval chunks with a heading path vs even pieces for scoring), the tagging floors were measured on tagging.Chunk, and a shared port would need a re-measure to prove it changes nothing.
- [x] doc drift
  internal/tablename says tagging uses ForPrefix (dead outside its test); WithGeneration docs point consumers at an internal package; README's package tree and dependency rule are out of date.

## search, vector, media

- [x] zero or NaN vector breaks every search in its scope
  A text with no letters or digits embeds to a zero vector with the fake embedder; once stored, every sqlitevec Search in that scope fails scanning a NULL distance, and on pg the NaN score passes the MinScore floor. Add vector.Validate (finite, non-zero), call it on index and query paths and in each store, and treat NaN as below the floor.
- [x] limit*4 over-fetch returns far fewer items than limit
  One 64-frame near-static video fills all fetched chunks, so SearchIndexed/SearchFrames return one hit where 10 exist. The store needs a per-document best-chunk search (pg DISTINCT ON, sqlite/mem loop with doubling k) and an exclude-doc argument.
- [x] empty partition or collection is a wildcard in Store.Search
  search.New(e, store, index, "", "notes") searches every partition and returns other tenants' docs. Search matches both coordinates exactly; only the explicit scope filter may leave one open.
- [x] concurrent sqlitevec writes fail with database is locked
  The transaction starts with a SELECT and upgrades to a write, so 8 concurrent upserts fail 60-95% of calls even with _busy_timeout. Make the first statement a write, document the DSN, add a concurrent-upsert test.
- [x] concurrent pgvec Upsert of one doc leaves duplicate chunk sets
  Under READ COMMITTED both transactions delete then insert and only serialize at the final docs upsert. Upsert the docs row first so its row lock serializes same-doc writers.
- [x] LazyEmbedder ignores ctx during load and panics on a nil embedder
  Embed blocks in once.Do past its deadline during a multi-GB load; load returning (nil, nil) or panicking leaves a nil inner that every later Embed dereferences.
- [x] sqlite caps k at 4096
  Index.Search with limit above 1024 errors on sqlitevec only. Clamp or page inside the store.
- [x] dimension change accepted at New, fails later with a raw driver error
  New(ctx, db, 16) over an 8-dim store succeeds and every write and search then fails. Compare the existing dims in ensure and return an error naming WithGeneration.
- [x] ffmpeg: path with a colon read as a protocol, out starting with - read as an option
  Sample(ctx, "clip:final.mkv") fails with Protocol not found. Pass file: URLs for inputs and outputs.
- [x] ffmpeg Sample buffers every raw frame in one buffer
  Up to 67 MB by default and about 2 GB for unscaled 4K, and each frame aliases the shared buffer. Read frame by frame from the pipe with per-frame allocations.
- [x] video with zero frames is stored as indexed and never retried
  An empty sampler or EmbedVideo result stores the hash with no chunks, so later IndexVideo calls are no-ops. Error on zero frames or segments.
- [x] IndexImage reads the whole reader unbounded
  io.ReadAll runs before the pixel guard. Limit the reader.
- [x] chunker misses paragraph breaks on CRLF and whitespace-only blank lines
  CRLF text is one paragraph and falls to splitWords, which flattens code and lists; the heading prefix is not counted toward MaxChars.
- [x] NUL or invalid UTF-8 indexes on sqlite and fails on Postgres
  Sanitize before chunking.
- [x] MemStore differs from the SQL stores
  Scores 0 on a dimension mismatch instead of an error, accepts an empty index, aliases the caller's vectors and orders ties randomly. Covered by the conformance suite: ties by score, then doc id, then seq.
- [x] ffmpeg sampler has no Fingerprint
  media/ffmpeg's sampler does not implement search.Fingerprinter, so changing its config does not re-index videos.
- [x] ffmpeg: restrict protocols
  Consider -protocol_whitelist file,pipe so a crafted playlist cannot make ffmpeg fetch URLs. Cut can leave a partial out file on failure.
- [x] SearchIndexed costs one store search per stored chunk
  Fine for short documents; a very long document or a 64-frame video runs many searches.

## tagging and eval

- [x] one failing member text makes a space un-rebuildable
  All single-piece texts go in one Embed call that returns nil on error, so nothing is cached and every Suggest re-embeds everything and returns ErrNotRebuilt. Embed in sub-batches, Put after each, isolate the bad text, and decide a policy for a poison member.
- [x] a rebuild cannot be cancelled and a panic in it kills the process
  The flight runs under context.WithoutCancel with no recover and Tagger has no Close, so a hung embedder blocks callers and a rebuild can outlive the DB or model.
- [x] same Model name with another dimension gives zero scores or a panic
  Cached vectors are never checked against Dimensions(); a ragged centroid indexes out of range. Treat a wrong-length hit as a miss, error on length mismatch in scores, refuse ragged input.
- [x] Prune drops vectors that are still needed
  Only changed tags are rebuilt, so stable tags' entries never refresh their last-used time and Prune(now-30d) removes them. Add a way to retain every key of the current catalog before pruning; document the 24h refresh granularity.
- [x] parseTags keeps reply scaffolding as tags
  JSON arrays, a Tags: label, think blocks and bold markers come back as tag text. Strip think blocks and fences, try JSON, cut labels, widen the trim sets.
- [x] cleanPath drops an existing path the model reused
  A level over 4 words or 40 runes voids the whole path, and reused paths are not mapped back to the application's spelling.
- [x] one failed window discards a whole long Generate
  No retry; an empty reply on the short path looks like no tags. Retry a window once and return partial results with a joined error.
- [x] best-effort cache writes fail the request
  A failed last-used UPDATE fails Get and so Suggest although the vectors were read; a Put error discards freshly embedded vectors.
- [x] non-finite vectors are cached forever
  embedBatch checks only lengths; one NaN member makes the tag never match until the cache is wiped.
- [x] callers at two revisions re-read the catalog on every call
  Flights are keyed by state, so two revisions rebuild concurrently and the later Apply leaves an unrecorded state. Key the flight by space.
- [x] eval.aliasSim matches inside words
  Tag ai scores 1 against container shipping; only the alias has a length floor. Match on word boundaries. Needs a re-measure of the long-set floors.
- [x] long-set headline numbers hide conventions
  A zero-tag document counts as precision 0 and the F1 is that of the mean precision and recall, not the mean per-document F1.
- [x] sqlcache: Postgres and SQLite differ behind the shared code
  Postgres rejects NUL or invalid UTF-8 in index or model; concurrent CREATE TABLE IF NOT EXISTS at startup can fail.
- [ ] examples: call Tagger.Prune
  ErrPartial is now shown with its tags and the Tagger is closed on shutdown (2026-11-09). The examples still never call Tagger.Prune.
- [ ] stale numbers in tagging comments
  tagging/chunk.go, generator.go and generator_long.go still quote long-set numbers from the scoring before 2026-11-09. The golden top1 floor comment says 0.8929 where the measured value is 0.8810 on main too; cause not found.

## providers and fetch

- [x] Vision.Embed puts raw text through the chat template
  A NUL byte empties the chat message so every such text gets the same vector with no error; a media marker in a text fails the batch and a control token is parsed as one. Tokenize template pieces and user text separately, as the Completer does.
- [x] library install is unsynchronized
  Two goroutines or processes on a fresh managed dir both download and one can remove the directory under the other's load; an Open asking for another build replaces the files under the loaded library. Mutex plus file lock, re-check after acquiring, refuse replacement when loaded.
- [x] NewCompleter drops Config.Accel and GPULayers
  It builds a Runtime by hand instead of cfg.runtime(), so AccelCPU still offloads to a GPU.
- [x] ManagedLibDir deletes whatever LibDir points at
  RemoveAll(libDir) after the download wipes a directory that held other files. Replace only a missing, empty or package-owned directory.
- [x] EmbedVideo treats unknown duration as short
  ffprobe reports 0 for files without a container duration, and the whole file goes to libmtmd as one input; an extensionless path gives ffmpeg no muxer.
- [x] downloaded files are not durable and never re-checked
  No Sync before rename, and any existing file is accepted, so a truncated file after a power loss fails every load. Sync, and compare the size when the spec has one.
- [x] interrupted downloads: fixed timeout, no resume, leaked temp files
  Open overrides the caller's deadline with 15 min / 1 h, each retry starts from zero, and a killed process leaves multi-GB temp files.
- [x] ResizeCrop allocates gigabytes for thin images
  A 1x20000 image allocates 664 MB. Scale only the central source rectangle.
- [x] hw reports CUDA from the driver alone
  A host without the CUDA runtime gets the CUDA build, falls to CPU and stays there although Vulkan would work.
- [x] Vision evaluation ignores ctx
  A video segment or photo cannot be cancelled and Close blocks behind it. Use llama's abort callback.
- [x] Vision.Embed errors where Embedder truncates
  A long text fails the whole batch instead of being truncated to the context.
- [x] embedding length taken from n_embd, not n_embd_out
  A custom GGUF with a smaller output size reads past llama.cpp's buffer.
- [x] llama.cpp log routing fixed by the first opener
  A nil Logger on the first Open silences llama.cpp errors for every later instance.
- [x] transparency handled differently per provider
  onnx premultiplies alpha (transparent becomes black), yzma composes on white. One shared flatten step.
- [x] digest comparison is case-sensitive
  An uppercase SHA256 in a spec fails after the full download.
- [x] model id before load for the lazy loaders
  The README's lazy example passes spec.Name as the model id, but yzma's Model() returns name plus pooling and prefixes, so the load would fail with a mismatch. The providers need a way to compute the id from a spec and config without loading.
- [x] filter llama.cpp ERROR lines after a cancelled evaluation
  A cancelled evaluation logs failed to decode at ERROR.
- [x] Makefile comment for test-vision is stale
  The target now also loads the model a second time on the CPU.
- [ ] phash: measure thresholds on real photos
  The duplicate thresholds come from synthetic pictures. Flat pictures all share the zero hash and a small mark on a near-flat picture flips about half the pHash bits. No ParseHash yet, and the vectors are not a compact bit column in the stores. grey.go has its own compose-on-white code that could use internal/imageutil if it gives identical 8-bit values.

## second pass

- [x] document ids are not sanitized
  An id with invalid UTF-8 fails every Postgres call of its batch, and on SQLite is stored but never found again by Hashes, Delete or Exclude. Reject such ids in vector.CheckWrite with a sentinel; IndexBatch puts the doc in Failed.
- [x] tables-exist cache never invalidates across Store values
  After another store or process deletes an index, every read and write of it fails with no such table until restart. On an undefined-table error drop the cached entry and reopen once.
- [x] pg: concurrent first write reads as an old schema; DeleteIndex not atomic
  open checks meta then docs in two statements, so another store's creation in between looks legacy (2 of 160 first writes); DeleteIndex drops four tables in separate statements.
- [x] a collection's model record outlives its documents
  After deleting every doc, an Upsert with a new model still returns ErrModelMismatch; the only way out is DeleteIndex, which drops every collection. Overwrite the record when the collection is empty.
- [x] sqlstore.open does database I/O under the store-wide mutex
  The first write of a new index holds the mutex while waiting for SQLite's write lock, stalling reads of every other index (a 200 ms deadline returned after 1.9 s).
- [x] SearchIndexed is N exact scans
  Measured on sqlitevec at 20.5k chunks x 768: one search 49 ms, a 10-chunk doc 0.49 s, 64 frames 3.3 s, 500 chunks 24.9 s. Cap the query vectors or rank several vectors in one scan.
- [x] Validate checks the norm in float64, backends compute in float32
  A stored {1e-25, 1e-25, 0, 0} passes Validate and scores +Inf on SQLite, topping every search.
- [x] pg: deadlock between Delete and Upsert of the same ids
  Upsert locks docs rows in id order, DELETE ... IN in scan order: 9 deadlock errors in 6 s with 4 writers and 4 deleters.
- [x] chunker drops headings that have no body
  Sibling headings without text vanish from the index, and an indented # comment in a code block is read as a heading.
- [x] empty-index and driver errors inconsistent
  errEmptyIndex is unexported and some methods return tablename's bare error; search.New accepts an empty index; sqlstore returns raw driver errors without the vector: prefix.
- [x] an embedder outage is recorded as a finished rebuild
  With every Embed failing, members are skipped, the state is recorded and after recovery the tag stays name-only until the space's next revision (score 0 where a healthy build gives -0.24 at another cutoff). Skip a member only on evidence the embedder works; otherwise fail the rebuild and record nothing.
- [x] a rebuild longer than each caller's deadline never completes
  With a 60 ms catalog read and sequential callers on 40 ms deadlines, 0 of 15 succeed and the catalog is read 15 times, because the run is cancelled when its last waiter leaves. Keep an abandoned run alive for a grace period.
- [x] parser keeps scaffolding and glues tags
  Hashtag lists, CJK and Arabic separators and full-width colons glue or keep labels; JSON objects with scores leak numbers; No tags. becomes a tag.
- [x] parser drops or truncates legitimate tags
  (re)insurance, 5" floppy and x < y are dropped; Topic modeling: LDA becomes LDA.
- [x] Prune's cutoff guard does not protect recent reads
  A cutoff of age A only protects reads from the last A minus 24 h, and MemCache and sqlcache differ at exactly 24 h.
- [x] Generator retries a closed Completer on every window
  A closed or down model is called 22 times on a 64 KB text.
- [x] FakeEmbedder and FakeImageEmbedder ignore ctx and cannot fail
  Cancellation and outage paths tested with them pass wrongly.
- [x] three spellings of one coordinate in tagging
  Prune takes Space, Scope takes two strings, vector uses Coord.
- [x] lazy loaders: recovery from a kept load failure is undocumented
  The docs never say to build a new lazy value to retry.
- [x] video segment files pile up in the temp dir
  embedSegments removes cuts only at the end, so a 100-segment video has 100 files on disk at once, plus the full remux for a duration-less file.
- [x] segments are cut into the source's extension
  Only an empty extension falls back to .mkv; .divx, .rmvb, .mp4.part and others fail at cut segment 0.
- [x] container duration is trusted
  A duration near MaxInt64 panics in make, a 1000 h claim plans 180,000 segments, and audio longer than video yields phantom tail segments with duplicate vectors.
- [x] ffmpeg Cut deletes files it did not write
  Cut(ctx, p, 0, 1s, p) removes the source after ffmpeg refuses; a missing binary removes a pre-existing out.
- [x] InstallDir deletes siblings it does not own
  Every Open removes <LibDir>.old and <LibDir>.staging unconditionally and sweeps .partial-* and .download-* files in the parent directory, which can be the home directory.
- [x] metadata can forge a showinfo line
  A video title spelling a showinfo line moves frame timestamps or fails Sample. Anchor the pattern to the filter's log prefix.
- [x] Vision's identity misses settings that change stored output
  Vision has no Fingerprint, so raising MaxSegments never re-indexes truncated videos; ContextTokens now changes long-text vectors but not the model id.
- [x] a symlinked managed LibDir is replaced by a real directory
  swap renames the symlink itself.
- [x] short videos reach libmtmd under the caller's raw path
  The file: hardening covers only the Cutter. Pass an absolute path.
- [x] install lock is taken before the already-usable check
  A process that already loaded the library waits on the lock while another process downloads.
- [x] yzma never compares the install record's tag
  A managed dir left by an older pinned build counts as ready and then fails the version check for the life of the process.
- [x] no cross-process download lock off unix
  Two Windows processes can interleave writes to one .partial file. No Windows build is pinned today.

# change log

- [x] vector: new Store port (breaking)
  Commits 66f4e6d 8cdce0a d2fd28f . vector.Store now takes a Coord{Index, Partition, Collection} and has Upsert(c, model, docs...), Delete(c, ids...), DeleteIndex, Hashes, Docs (paged listing), Chunks and one Search(c, Query). Query has AllPartitions/AllCollections (an empty partition or collection is now an exact value, not a wildcard), Only/Exclude doc-id lists and Model. Search returns one match per document (best chunk), ordered by score, doc id, seq, so limit means documents and one many-frame video no longer crowds out the rest. New sentinels vector.ErrDimensions, ErrInvalidVector, ErrModelMismatch and vector.Validate; zero and NaN vectors are rejected on write and search. Both SQL stores share vector/internal/sqlstore; reads no longer create tables; opening with other dims fails naming WithGeneration; DropGeneration removes an old generation; old-schema tables are refused with a clear error (delete the example DB to rebuild).
- [x] sqlitevec: BLOB vectors instead of the vec0 table
  Vectors are stored as BLOBs and ranked exactly with vec_distance_cosine, which removes the k cap of 4096 (measured on 20k x 768: 19 ms vs 17 ms for vec0 on a full scope, 4x faster on a small scope). Write transactions start with a write, so concurrent upserts no longer fail with database is locked; recommended DSN is _busy_timeout=5000&_journal_mode=WAL. pgvec writes the docs row first so same-doc writers serialize.
- [x] model identity: langai.Identified and a per-collection model record
  Optional port langai.Identified{Model() string}. The stores record the model id per collection on first write and return vector.ErrModelMismatch for another id; search.New and NewMediaIndex pass the embedder's id. The fakes report fixed ids; the real providers and LazyEmbedder follow in later entries.
- [x] vector/vectortest conformance suite
  vectortest.Run runs 23 cases against MemStore, sqlitevec and pgvec (replacing the hand-copied per-backend tests): batch atomicity, exact coordinates, scope filters, allow/exclude lists, crowding, large limits, invalid vectors, dimension and model mismatch, deterministic ties, concurrent upserts, reads on an unknown index.
- [x] langai.Normalize moved from vector to the root package
  Providers no longer import the storage port for it.
- [x] tagging: rebuilds survive bad texts, cancels and panics
  Commits 018142c 133ee49 f6bc9e6. Texts embed in batches of 32 with a cache write after each, so progress survives a failure or cancel; a failed batch is retried text by text, a member text that cannot be embedded (or embeds to NaN/zero) is skipped and logged at Warn through the new TaggerConfig.Logger, and a failing tag name fails the rebuild naming the tag. A rebuild is cancelled when its last waiter leaves, a panic in it becomes an ErrNotRebuilt error, rebuilds of one space serialize, and Tagger.Close() (tagging.ErrClosed) waits for them. A cached vector of another length is re-embedded and mixed lengths are an error, not a zero score or a panic. Open point: an embedder outage during a rebuild skips members and builds tags name-only until the space's next revision.
- [x] tagging: Tagger.Prune and EmbeddingCache.Touch (breaking)
  Commit acda626. Tagger.Prune(ctx, unusedSince, spaces...) first refreshes the last-used time of every name and member key of the given spaces, then prunes, and refuses a cutoff younger than CacheRefreshAfter; before, pruning dropped the vectors of stable tags. EmbeddingCache gained Touch (all backends, MemCache, cachetest); new Space{Index, Partition} and Scope.Touch.
- [x] tagging: cache hardening and a clocked conformance suite (breaking)
  Commits 6abd7c5 cfb3d4b. cachetest.Run now passes an injected clock and checks the last-used refresh, duplicate keys and the prune boundary (NewMemCacheWithClock). sqlcache refuses NUL or invalid UTF-8 in index and model names on both databases and creates its schema under a Postgres advisory lock. A failed last-used write no longer fails Get (ErrCacheWrite is logged) and a failed Put no longer discards fresh vectors.
- [x] tagging: Generator reply parsing, tree paths and partial results
  Commits 9fe889a ab8c4ae. parseTags strips think blocks and code fences, reads JSON arrays and {tags: [...]}, cuts labels, emphasis and trailing parentheticals and drops names with unbalanced markup (so a flat tag like dev > go is now dropped). cleanPath maps a reused tree path back to the application's spelling before the word limits. A failed or empty model call is retried once; a long Generate returns the tags of the windows it read together with ErrPartial, and an empty reply is ErrEmptyReply. make test-gentags was not re-run for the parser change.
- [x] search: own Match type, SearchIndexed, IndexBatch, Docs, multi-id Delete (breaking)
  Commits fce5689 1ef2fac 6b9da6e 2949b40. Index.Search returns search.Match{DocID, Score, Seq, Text} instead of vector.Match. New Index.SearchIndexed (more like this from stored vectors); both SearchIndexed methods now leave the query document out and return ErrNotIndexed for an unknown id. Index.IndexBatch(ctx, []Doc) reads all hashes once, embeds at most 32 texts per call and returns BatchResult{Indexed, Unchanged, Failed}; a document that fails on its own goes to Failed, an embedder or store error stops the call and a second call resumes. Index.Docs / MediaIndex.Docs iterate the indexed ids and hashes for reconciliation; Delete takes several ids. New sentinels ErrEmptyID, ErrNotIndexed, ErrTooLarge, ErrNoFrames; every error carries the search: prefix.
- [x] search: stored hashes cover chunker, sampler and model (breaking)
  Commit 6936987. The stored hash now folds in the chunker's Fingerprint (MarkdownChunker reports its MaxChars; another chunker is described by its Go type), the model id and dimensions, and for media the sampler's or video embedder's Fingerprint when it has one. search.ContentHash is gone. Every collection re-indexes once after this lands.
- [x] search: input hardening and chunker fixes
  Commits 0ebf453 05885a3. IndexImage and SearchImage read at most 64 MiB (WithMaxImageBytes, ErrTooLarge); a video with no frames or segments returns ErrNoFrames and stores nothing; text is cleaned of NUL and invalid UTF-8 before chunking and hashing so SQLite and Postgres agree; a query that embeds to the zero vector returns no results. The chunker reads CRLF, splits on whitespace-only blank lines, counts the heading prefix toward MaxChars (capped at a quarter of it) and splits an over-long paragraph between lines before words, so code blocks keep their shape.
- [x] search: JPEGs are turned upright by EXIF orientation
  Commit 0c440fe. A small bounds-checked APP1/TIFF parser in search/exif.go applies all 8 orientations after decoding; malformed EXIF means no orientation. Includes a fuzz target. Images re-embed once (their hash settings carry a version bump).
- [x] lazy loaders: LazyEmbedder rebuilt, LazyImageEmbedder and LazyCompleter added (breaking)
  Commit cb49c5f. NewLazyEmbedder(model, dims, load func(ctx)) runs the load once on its own goroutine under a context only Close cancels; a caller's ctx bounds only its wait. A panic (LoadPanicError) or nil result is an error, a failed load stays cached (as before), Close releases waiters and closes the loaded value, State() reports LazyNotStarted/Loading/Loaded/Failed/Closed, and Model() makes it langai.Identified; a loaded model with another id or dimension fails the load.
- [x] langaitest.FakeCompleter
  Commit f58839a. A scripted chat model (Replies in sequence or a Reply func, injectable Err) that records its calls; the autotag example's test uses it.
- [x] media/ffmpeg: any local path, streamed frames, reliable process cleanup
  Commit 5b09557. Inputs and outputs are passed as file: URLs, so paths with a colon, a leading dash or a percent sign work. Frames are read one at a time from the pipe (PAM) into buffers of their own instead of one shared buffer of up to gigabytes; ffmpeg is killed and reaped on cancel, read error and at MaxFrames (WaitDelay 2 s); zero frames is an error. Not tested: an ffprobe older than 8 rejecting -pattern_type.
- [x] internal/tablename: unused ForPrefix removed
  Commit e8e2809.
- [x] downloads: shared in internal/fetch, resumable, durable, locked
  Commits 300af05 f475dd2 374a08c. The model download and the staged library install moved from both providers into internal/fetch; yzma.Download is gone. A cut download keeps <file>.partial and resumes with a Range request (digest still verified), files are synced around the rename, a model file of the wrong pinned size is fetched again, the default timeout applies only when ctx has no deadline, and old temp files are swept after 2 h. Library installs run under an in-process lock plus a file lock (<LibDir>.lock now appears next to managed dirs, also in ~/.cache/semsearch-*); a loaded library dir is never replaced and ManagedLibDir only replaces a missing, empty or package-installed directory. Digests compare case-insensitively. Open: another process can still replace a directory a running process loaded from.
- [x] yzma Vision: safe tokenization, truncation, cancellation, duration-less videos
  Commits d6ab874 fc61c1d. Text goes through mtmd.TokenizeFromParts as plain-text parts outside the chat template, so NUL bytes, media markers and control tokens in a text are data (token ids identical to before for ordinary inputs, checked on the real model). Long texts are cut to the context instead of failing the batch. llama.cpp's abort callback stops Vision, Completer and Embedder when ctx ends (a 3 s CPU evaluation returned 25 ms after its deadline; a GPU backend finishes its batch). A video without a container duration is remuxed to .mkv and segmented, and extensionless files are cut into .mkv.
- [x] yzma: Completer honours Accel, n_embd_out, log routing, CUDA detection
  Commits 6aab3a7 e61fb4b fd44142 e1c41fa. NewCompleter passes Config.Accel and GPULayers; vectors are sized by the model's output embedding length; llama.cpp's log goes through a pointer later opens can set; CUDA is reported only when libcudart.so.13 and libcublas.so.13 are on the loader path, with a warning when a GPU build registers no GPU (library names inferred from the asset name, not run on NVIDIA hardware).
- [x] onnx: thin images and transparency
  Commit b27aa24. A 1x20000 picture is prepared in under 16 MiB instead of 664 MB (output byte-identical on ordinary pictures), and both providers compose transparency on white through internal/imageutil.ScaleOnWhite.
- [x] langai.ErrClosed and Model() on the real providers
  Commits 526ad13 537d7e9. Every provider's after-Close path wraps langai.ErrClosed. yzma.Embedder, yzma.Vision and onnx.Embedder report Model() ids that include what changes the vectors (pooling and prefixes; instruction, dims and video settings; preprocessing). Two quantizations of one model need two spec names, since the id leaves out the file digest.
- [x] real-model runs after the provider changes
  make test-models, make test-images (also with DINOv2) and make test-vision all pass with numbers identical to the baseline before the changes. The unit tests no longer reach the network (the fake runtime now pins the CPU build; yzma tests went from 9 s to 0.4 s).
- [x] provider/phash: perceptual hash ImageEmbedder (new)
  Commits 571f5a7 62cc0ee 05b0abe. A model-free langai.ImageEmbedder for exact and near-exact duplicates: pHash (default) or dHash at 64 or 256 bits, as unit vectors so cosine = 1 - 2*hamming/bits and MediaIndex plus any Store work unchanged; Model() is phash-64, dhash-64, phash-256 or dhash-256. Integer arithmetic only (goldens also pass under GOARCH=386), area-average resize read in strips, alpha on white. Embedder.Hash returns the raw Hash with Distance and String. Measured on synthetic pictures (Hamming bits, phash-64): JPEG q60 2, half size 2, double 2, brighter 2, small mark 10; unrelated pairs average 31.5, closest 14. About 1.8 ms per 1920x1080 RGBA picture. Thresholds are not yet measured on real photos; the phash-64 limit for a marked copy has no margin (10 of 10).
- [x] eval: honest long-set scoring and finite metrics (breaking)
  Commits 234444b db3a999. Long-set tags match on whole words (NFC, case-folded, at least 4 runes for a contained string), a tag goes to the topic it names, and MeanLongScores returns LongMean with the mean per-document F1, rates averaged only over documents they exist for, and an Untagged count; the F1-of-means figure is gone. ndcg at k<=0, negative k, repeated ids, NaN scores and an empty cut list no longer give NaN, recall over 1 or a panic. Dataset.Validate refuses repeated tags or ids; LoadLong refuses an empty directory and topics sharing a label.
- [x] quality floors re-measured on the real models
  Commits 5399783 d607b1b 4a438c9. Tagger long set: main recall 1.000 (floor 0.95), side 0.897 (0.84), precision 0.801 (0.75), mean F1 0.838. Generator long set: main 1.000 (0.95), side 0.759 (0.70), precision 0.887 (0.86 kept), mean F1 0.822. Golden: fits no tag, gets none 84 of 84 (floor 0.95). The gains come from the scoring fix, not from the models: the Generator's tags under the old rule reproduce the old numbers exactly, so the parser change moved nothing. make test-models, the Tagger long test and make test-gentags all pass against the new floors. Two comments were found stale and predate this branch: golden top1 is 0.8810 (comment said 0.8929, floor 0.84 passes) and 5 to 10 generated tags reach the cutoff (comment said 6 to 10).
- [x] examples: partial tag results shown, Tagger closed on shutdown
  Commit 709762e. The autotag notebook keeps the tags that come with tagging.ErrPartial and shows a notice (Recommendation.Error); Notebook.Close closes the Tagger in the example's shutdown path.
- [x] one closed sentinel; video ports in the root (breaking)
  Commits 450ff4d ea02ae6. Lazy loaders and the Tagger wrap langai.ErrClosed; tagging.ErrClosed is gone. langai.VideoEmbedder and langai.Segment moved from media to the root; media keeps Sampler, SamplerFunc, Frame and Cutter.
- [x] provider constructors on one convention (breaking)
  Commit 9f73364. yzma and onnx Config embed Runtime (Config{Runtime; ModelsDir, Model}); yzma.NewVision(ctx, cfg, VisionOptions) and OpenVision(ctx, spec, rt, VisionOptions) replace VisionConfig; Spec is renamed LookupSpec in both providers; yzma gained Models(), CompleterModels() and VisionModels(). sqlitecache.New and pgcache.New return concrete *Cache types. Every Open validates its config before any download: negative Threads, GPULayers, BatchSize, DeviceID, Dims, Size, MaxTokens or image tokens, min above max, unknown Accel or Provider are errors.
- [x] model id computed from the spec before loading
  Commit 9989e86. yzma.ModelSpec.ModelID(), yzma.VisionSpec.ModelID(opts) and onnx.ModelSpec.ModelID() return the same id the loaded embedder's Model() reports, so a LazyEmbedder can be given the right id up front; the README's lazy example would have failed its load with the bare spec name. Tests wire a lazy loader around each provider's fake backend.
- [x] ffmpeg: Fingerprint, file protocol only, failed cuts removed
  Commit 5f62c53. Sampler.Fingerprint() covers every setting that changes the frames, so a changed config re-indexes videos. Every ffmpeg and ffprobe input gets -protocol_whitelist file, so a crafted playlist cannot fetch URLs. Cut removes its output on failure or cancel.
- [x] fakes and example tests
  Commits 0c54f46 2b42534 61d9093. langaitest.FakeVideoEmbedder added; the hand-written chat and video stubs in tagging, search and the examples use the fakes. README samples are mirrored as Example tests (lazy loader, search.New, NewMediaIndex, provider constructors, sqlitecache). README, package docs and the Makefile were brought in step with the code. llama.cpp errors caused by a caller's cancellation are logged at Debug; the abort status value and log texts were matched against the pinned binaries, not llama.cpp's source.
- [x] tagging second pass: outages, rebuild grace, Prune guarantee (breaking)
  Commits 659bf96 6e48974 a4a1c48 6634611 9b50dbf. An embedder outage no longer gets recorded as a finished rebuild: a text is skipped as bad only once another Embed has worked (a later text or a canary); otherwise the rebuild fails with ErrNotRebuilt and records nothing, ErrClosed and context errors end the call at once, and a dead embedder costs at most 5 calls. Scope.Incomplete lists the tags built without a member, with no Logger needed. An abandoned rebuild keeps running for TaggerConfig.RebuildGrace (default 30 s, negative cancels at once), so short request deadlines cannot starve a space. Prune refuses a cutoff younger than MinPruneAge (2 x CacheRefreshAfter) and documents its guarantee; both caches refresh at exactly CacheRefreshAfter. Tagger.Scope takes a Space, and Scope.Space returns it. Open: Catalog and TagIndex still take (index, partition) strings; a per-text deadline from the embedder now fails the rebuild instead of skipping the member.
- [x] tagging second pass: parser and Generator
  Commits 3b78f7a 8f8043f. Hashtag runs and CJK/Arabic separators split into several tags, full-width colons and CJK label words are cut, JSON replies are read by their tag members only (tags, keywords, labels, topics; name/tag/label per item) and never also by lines, key: number items are dropped, and a reply that is only No tags / None / N/A (and a few es/de forms) gives no tags. Kept now: (re)insurance, 5" floppy, x < y, dev > go in flat mode, Topic modeling: LDA whole; in tree mode -> separates levels. A closed Completer is asked once (was 22 times on a 64 KB text) and a long text stops after two failed windows with none read.
- [x] langaitest: fakes honour ctx and can fail
  Commit ce5915f. FakeEmbedder gained Err, Fail func(text) error and Calls; FakeImageEmbedder gained Err and Calls; both check ctx first.
- [x] vision second pass: segment planning and temp files (breaking)
  Commits 2689210 01995a0 e0ee71e f4d5884 73ceb80. Segments are planned from the video stream's duration (then Matroska's per-stream tag, then the container's); NaN, non-positive or over-a-week values count as unknown; the count is overflow-safe. MaxSegments 0 now means yzma.DefaultMaxSegments = 720, not all. Cut returns media.ErrPastEnd past the real end, so audio longer than video no longer yields duplicate tail segments. Cuts go into the source's container only where it is known to take the stream, else Matroska; each cut is removed once embedded; VisionOptions.TempDir sets where they go. Videos reach libmtmd under an absolute path. Vision.Fingerprint() covers segment length and cap, and the context cap is now part of the model id (ctx= for Vision, max_tokens= for Embedder), so every yzma identity string changed and stored vectors re-embed once. Open: libmtmd's own ffmpeg has no protocol whitelist for short videos (a local playlist naming local files gets through).
- [x] ffmpeg second pass: Cut and Sample
  Commits a518f98 f56738d. Cut refuses an out that is its input, writes under a temp name next to out and renames on success, and never removes a file it did not write. Frame times are read only from the showinfo filter's own lines, by frame number, so a video title cannot forge them. The sampler fingerprint is ffmpeg/2 because frames now spread over the video stream's length, so sampled videos re-index once.
- [x] fetch second pass: install ownership, symlinks, lock
  Commits 23e9aec 4c76695 80fd6eb e088c44 ee889a5. An install touches only siblings named <dir>.langai-download/-staging/-retired that carry its marker or record (anything else there is an error) and no longer sweeps temp files in the parent directory. A symlinked library directory is installed where it points. A process already running the directory's libraries skips the install lock, and a lock wait that outlasts ctx returns ctx's error (before, it went on unlocked). yzma reinstalls a managed directory whose record names another llama.cpp tag. Off unix each process uses its own partial file; installs there are still not serialized across processes.
- [x] vector second pass: multi-vector search (breaking)
  Commit 4270a7a. vector.Query.Vector is replaced by Vectors [][]float32 and Match gained Query (which vector a hit matched); one scan scores each chunk against its closest query vector, so SearchIndexed no longer runs one search per stored chunk. Measured on 20k chunks x 256 dims for 1 / 16 / 64 vectors: sqlitevec 26 ms / 163 ms / 589 ms (was 28 ms / 444 ms / 1.89 s), pgvec 12 ms / 39 ms / 114 ms (was 13 / 182 / 698), MemStore 9 / 50 / 182 (was 8.5 / 93 / 333). search sends at most DefaultMaxQueryVectors = 64, evenly spaced (WithMaxQueryVectors, WithMediaMaxQueryVectors). Results equal the old merge of single searches on all three stores (conformance case).
- [x] vector second pass: ids, models, schema races (breaking)
  Commits baf6aea 47975e1 70ffe86 872d33f 711f151 ab7184e. Ids, partitions, collections and models with invalid UTF-8 or NUL are refused with vector.ErrInvalidID (search.ErrEmptyID wraps it; in IndexBatch such a doc lands in Failed); vector.ErrEmptyIndex is exported and checked everywhere, also in search.New; driver errors are wrapped as vector: <backend> <op>. An emptied collection gives up its model record, Store.DeleteCollection and Store.Model are new on the port, the search indexes check the model before embedding and gained Reset(ctx) for a model migration. Tables are created and dropped in one locked transaction (no more spurious old-schema error on a concurrent first write), a store reopens an index another store dropped, opening an index does no I/O under the store-wide lock, pgvec locks deleted rows in id order and retries on deadlock, and vectors whose float32 norm underflows or overflows are refused. vectortest gained RunShared for several stores on one database. Open: content hashes and chunk text are not checked for invalid UTF-8 at the Store level (search cleans text first).
- [x] chunker: bodiless headings and code blocks
  Commit f811299. A heading with no text and no subheading becomes a chunk of its heading path instead of vanishing; ATX headings allow at most 3 spaces of indent and are not read inside fenced code. The chunker fingerprint is markdown/2, so text collections re-index once.
- [x] final verification on 2026-11-09 at 8ed7fb3
  All pass: go test -race ./... with the Postgres containers, make verify (test, license-check, lint, benchmark, coverage), make test-models, the Tagger long set, make test-images, make test-vision and make test-gentags. Numbers: golden search recall@5 1.000, mrr 1.000, cross-lingual recall@10 1.000, tagging F1 0.707, top1 0.881; Tagger long set main 1.000, side 0.897, precision 0.801, mean F1 0.838; Generator long set main 1.000, side 0.759, precision 0.887, mean F1 0.822 (unchanged by both parser reworks); images 0.992 / 0.979 / 0.986 / 0.928 / 0.693; vision red vs resized 0.932, red vs blue 0.710. Not run: make semeval, anything on CUDA or macOS, and CI (nothing was pushed).

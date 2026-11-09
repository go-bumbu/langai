# Image and video search

Finding images and videos that look alike with `search.MediaIndex`: with an image model, with no model at all (duplicates), and with a vision model that puts text, images and videos in one space.

## With an image model

```go
emb, err := onnx.New(ctx, onnx.Config{
    ModelsDir: "data/models", // Model "" is onnx.DefaultModel; onnx.Models() lists the others
    Runtime: onnx.Runtime{
        LibDir:        filepath.Join("data", "onnxruntime", onnx.LibVersion), // named after the pinned build
        ManagedLibDir: true,                                                  // download the pinned build into it
        Provider:      onnx.ProviderCPU,                                      // or ProviderCUDA, ProviderCoreML
    },
})
frames, err := ffmpeg.New(ffmpeg.Config{}) // scene cuts, one frame every 5 s, at most 64, 512 px
store, err := sqlitevec.New(ctx, db, emb.Dimensions())

// an image model's vectors get a collection of their own, never a text model's
ix, err := search.NewMediaIndex(emb, store, "tenant-1", "space-7", "media",
    search.WithSampler(frames), search.WithMediaMinScore(0.6))
err = ix.IndexImage(ctx, "photo-42", file)             // JPEG, PNG, GIF, WebP, BMP, TIFF; a no-op while the bytes are unchanged
err = ix.IndexVideo(ctx, "clip-7", "/media/clip-7.mp4") // one vector per sampled frame; a no-op while the file is unchanged
hits, err := ix.SearchImage(ctx, file, 10)              // []search.MediaMatch: DocID, Score, At (the matching frame's time)
hits, err = ix.SearchVideo(ctx, "/media/query.mp4", 10)  // and QueryAt, the query frame that matched
hits, err = ix.SearchIndexed(ctx, "clip-7", 10)          // "more like this" with the stored vectors: no model runs
err = ix.Delete(ctx, "photo-42", "clip-7")               // Docs lists what is indexed, as for Index
```

An image is one frame at time zero and a video the frames its `Sampler` picked, each
embedded as an image into one vector space, so an image query finds videos and a video
query finds images. A search embeds the query frames, asks the store in one search for
the best frame of each of the closest items and scores an item by its best frame
against any query frame, so a video is found by the one shot that resembles the query
most, and its other frames take no places from other items. A query of more than 64
frames or segments (`WithMediaMaxQueryVectors` changes it) searches with 64 of them,
evenly spaced, and `SearchFrames` picks them before embedding; `At` says where in the video that is, and for a video
query `QueryAt` which of its frames matched. `SearchIndexed` leaves the item it starts
from out of its hits; a search by a file that is also indexed finds that item first, with
a score near 1. `IndexFrames` and `SearchFrames` take
frames you extracted yourself, with your own content hash; more image formats register
through `image.RegisterFormat`.

`IndexImage` and `SearchImage` read at most 64 MiB from their reader
(`WithMaxImageBytes`) and refuse a longer stream with `search.ErrTooLarge`, and refuse a
picture of more than 256 megapixels before decoding it. A video the `Sampler` takes no
frames from, or the `VideoEmbedder` no segments, is `search.ErrNoFrames` and nothing is
stored, so the next run tries it again rather than keep an item that is never found.

- **Models.** The default, `google/siglip2-base-patch16-224` (Apache-2.0, 768
  dimensions), was trained with a text tower, so its vectors can later be matched by
  text too; with it a resized or recompressed copy scores above 0.9 and an unrelated
  picture well below. `onnx.DINOv2Model` (`facebook/dinov2-base`, Apache-2.0, 768
  dimensions) has no text side and keeps instance-level detail, so it suits
  near-duplicate matching: it puts unrelated pictures further apart. The two live in
  different spaces: one model per collection, as `tagging` keys every embedding by model.
  `Open` runs any other ONNX vision model from a `ModelSpec` (tensor names, pooling,
  input size, resize and normalization). Finding the copies of a picture (the same one
  re-encoded, resized or lightly edited) needs no model at all: see
  [Duplicates without a model](#duplicates-without-a-model).
- **Videos.** `ffmpeg.Config` sets the scene-change threshold (0.4), the shortest and
  longest gap between frames (1 s and 5 s), the cap per video (64 frames, with the
  shortest gap raised to `duration/64` so a cut-heavy video still yields about that
  many) and the size frames are scaled to (512 px). Scene detection decodes every frame;
  `KeyframesOnly` decodes only the encoder's keyframes, which sit at cuts and at a fixed
  interval anyway, and makes hours of video cheap at the price of a coarser sample.
  The `Sampler`'s `Fingerprint` covers every one of these settings, so changing one
  indexes the videos again. A path is always a local file, relative or absolute: a colon,
  a leading dash or a `%d` in a name is never read as a protocol, an option or an image
  sequence, and ffmpeg and ffprobe open their input with only the `file` protocol on
  their whitelist, so a crafted playlist or concat list that names an `http://` (or
  `rtp://`) input fails at once instead of reaching the network. A frame's time comes
  from the line ffmpeg's own filter logs for it, by the frame's number, so a title or a
  chapter name that spells such a line moves nothing. Frames are read
  from ffmpeg one at a time, each with pixels of its own, so memory holds the frames
  taken (at most 64 × 512 × 512 × 4 bytes, 67 MB, with the defaults) and a frame you
  keep does not keep the others alive. A video no frame comes out of is an error, and
  cancelling the context kills ffmpeg and ffprobe.
- **Cost.** On a CPU with the default four threads the SigLIP 2 model embeds a
  1024-pixel picture in about 100 ms, eight per batch (an AMD Ryzen AI 9 HX PRO 370,
  `BenchmarkEmbedImagesReal`), and the process holds about 1.3 GB with the model
  loaded; `Runtime.Threads` and `BatchSize` tune it, and CUDA takes batches. Indexing a
  video costs its decode plus one embedding per frame. Embedding, the ffmpeg run and the
  hash of a video all run per call; the background job that indexes decides how many run
  at once.
- **Limits.** A JPEG is turned upright by its EXIF orientation (all eight, read by a
  small parser of its own that takes a broken block for none) before it is embedded;
  other formats are embedded as decoded. Transparent parts are composed on white, here
  and in the vision model below, so a logo on a transparent background is the same
  picture to every model. A picture of an extreme shape (one pixel wide, twenty thousand
  tall) costs no more memory to prepare than any other. The SigLIP 2 text tower is not
  wired, so text queries need the vision model below. The CUDA and CoreML paths are wired
  but were not run on a GPU in this release; the CPU path is what the real-model tests
  cover.

## Duplicates without a model

```go
dup, err := phash.New(phash.Config{})                  // the 64-bit pHash; or Algorithm: phash.AlgorithmDHash, Bits: 256
store, err := sqlitevec.New(ctx, db, dup.Dimensions()) // 64 dimensions, one per bit
ix, err := search.NewMediaIndex(dup, store, "tenant-1", "space-7", "duplicates",
    search.WithMediaMinScore(0.875))                   // at most 4 of the 64 bits differ
err = ix.IndexImage(ctx, "photo-42", file)
hits, err := ix.SearchIndexed(ctx, "photo-42", 10)     // its copies; Score is 1 - 2·hamming/64

h, err := dup.Hash(img)          // or without a store: the bits themselves
same := h == other               // comparable, so also a map key
near := h.Distance(other) <= 4   // Hamming distance; h.String() is the hex
```

`provider/phash` is an `ImageEmbedder` that runs no model: it reduces a picture to a
perceptual hash and returns the bits as a unit vector of ±1/√bits entries, so the cosine
of two pictures is exactly `1 - 2·hamming/bits` and `MediaIndex` and every store work on
it as on a model's vectors. It is pure Go, downloads nothing and hashes a 1920×1080
picture in about 2 ms (5 ms for a decoded JPEG; `BenchmarkHash`, one core of an AMD Ryzen
AI 9 HX PRO 370), so a de-duplication job does not pay for a model.

- **What it finds.** The same picture again: saved as a JPEG, scaled up or down,
  slightly brightened, or with a small mark stamped on it. Not a crop, a rotated or
  mirrored copy, or a picture that only looks similar; those need a model
  (`onnx.DINOv2Model` for near-duplicates, SigLIP 2 for "looks like"). A picture with
  almost no structure has almost nothing to hash: every flat picture has the same hash.
- **Algorithms.** `AlgorithmPHash` (the default) thresholds the 8×8 lowest frequencies of
  the DCT of a 32×32 grey grid at their median; `AlgorithmDHash` compares neighbouring
  cells of a 9×8 grid. pHash keeps unrelated pictures furthest apart; dHash moves least
  under a local edit. `Bits: 256` widens the gap between duplicates and the rest.
- **Scores.** With the 64-bit pHash a re-encoded, resized or brightened copy stayed
  within 2 bits on the package's test pictures and one with a mark within 10, while the
  closest of 79,800 unrelated pairs was 14 bits apart: a floor of 0.875 (4 bits) keeps
  the plain copies, 0.6875 (10 bits) the marked ones too. The package doc has the table
  for all four hashes.
- **One hash per collection.** `Model()` names the algorithm and size (`phash-64`,
  `dhash-64`, `phash-256`, `dhash-256`), so a store refuses to mix two of them, or a
  hash with a model's vectors.
- **Deterministic.** Integer arithmetic only: the same pixels give the same bits on
  every OS and architecture. Transparency is composed on white, and any `image.Image`
  is accepted (paletted, 16-bit, CMYK, YCbCr) up to 256 megapixels.

## Text, images and videos in one space

```go
v, err := yzma.NewVision(ctx, cfg, yzma.VisionOptions{ // cfg as for yzma.New: the Runtime is shared
    Cutter: frames, // the ffmpeg Sampler cuts videos into segments too
})
ix, err := search.NewMediaIndex(v, store, "tenant-1", "space-7", "media",
    search.WithVideoEmbedder(v), search.WithTextEmbedder(v))
err = ix.IndexImage(ctx, "photo-42", file)              // as above
err = ix.IndexVideo(ctx, "clip-7", "/media/clip-7.mp4") // one vector per 20 s segment
hits, err := ix.SearchText(ctx, "a walrus on a beach", 10) // []search.MediaMatch, At = the segment's start
vecs, err := v.Embed(ctx, sense.RoleQuery, []string{"..."})  // the same object is the text Embedder
```

`yzma.Vision` runs `Qwen/Qwen3-VL-Embedding-2B` (Apache-2.0, 2048 dimensions) through
llama.cpp and its libmtmd: the model's chat template wraps every input in an
instruction (`Represent the user's input.` unless `VisionOptions.Instruction` says
otherwise), the projector turns pictures and frames into tokens, and the vector is the
last token's hidden state. Texts, images and videos land in one space, so a text finds
pictures and clips, and an image finds the stretch of a video it appears in.

- **Videos as segments.** A video goes through the model in segments of
  `SegmentLength` (20 s), each sampled at `FPS` (1) and embedded as one input: about 190
  tokens per second of video, so a segment is a few thousand tokens and a feature film
  never overflows the context (`ContextTokens`, 16384). The `Cutter` (the ffmpeg
  `Sampler`) cuts by stream copy, so a segment starts at the last keyframe before its
  time. A cut is written under a name of its own and renamed into place when it is
  whole: one that fails or is cancelled leaves nothing, and never removes a file that
  was already there, and a cut onto its own input is refused. `MaxSegments` caps
  the work per video: 720 segments unless you set it (`DefaultMaxSegments`, four hours
  at 20 s), and what lies past the cap is not embedded. A segment is one vector: the
  model sees the whole stretch, which is what makes a clip match a description, and
  what loses the "matched at 4.00 s" precision of the frame route.
- **A video's length is read, not believed.** The length is the video stream's (a
  sound track that runs on after the picture does not add segments), and a header can
  still be wrong. A length no video has (none, or more than a week, `media.MaxDuration`)
  counts as unknown, and a walk through a video that ends before its header says stops
  at the `Cutter`'s `media.ErrPastEnd` instead of embedding the last pictures again.
  A file without a length (a recording that was never finalized, a stream dump) is
  first copied into a Matroska file in the temp directory, at most `MaxSegments`
  segments of it, so that it has one, and is then cut like any other.
- **Disk.** Segment files go into a directory of the call's own inside
  `VisionOptions.TempDir` (the system's temp directory unless set), one at a time: each
  is removed once it is embedded. A video takes the room of one segment there; one
  without a length also that of its copy, up to `MaxSegments` segments long, so point
  `TempDir` at a disk when the system's is kept in memory.
- **Paths.** libmtmd reads a video through an ffmpeg of its own and gives it the path
  bare, so every video reaches it under an absolute path, a segment file as well as a
  video short enough to go uncut: a relative `-x.mp4` or `clip:final.mkv` is then no
  option and no protocol. That ffmpeg has no protocol whitelist; the `Cutter` probes
  every video under its own first, and refuses one that names remote inputs.
- **Segment containers.** ffmpeg picks a segment's container by its file extension, so
  a segment keeps the video's extension only where that names a container known to take
  such a file's codecs by stream copy: `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.ts`,
  the MPEG program streams and raw H.264/HEVC streams (`.m4v` is cut into `.mp4`).
  Anything else is cut into `.mkv`: a name without an extension, one ffmpeg has no
  container for (`.divx`, `.rmvb`), a backup or a half-renamed download (`.mp4.part`).
- **Texts are plain text.** The chat template is laid out once, around stand-ins, and
  the instruction and every input are tokenized as parts of their own: a text that
  spells one of the model's control tokens (`<|im_end|>`), libmtmd's media marker or a
  NUL byte is read as those characters and nothing else. A text that does not fit the
  context next to the instruction is cut at its end, as the text `Embedder` cuts,
  instead of failing the batch it came in; a picture or a video segment that does not
  fit is still an error (`ImageMaxTokens`, `SegmentLength`).
- **Cancelling.** `ctx` is passed to llama.cpp as an abort callback in all three
  (`Embedder`, `Completer`, `Vision`): on the CPU a cancelled call stops within one
  step of the computation and returns the context's error, and the model stays usable.
  A GPU backend does not ask the callback, so there the batch that is running (up to
  2048 tokens for `Vision`, 512 for the `Completer`) finishes first.
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

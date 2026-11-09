# Upgrading

What changes for stored data and behaviour between versions. Everything the library stores is derived from your own documents, so an upgrade costs indexing time, never data.

## From v0.1.1

- **Search tables of v0.1.1 are refused.** The SQL stores have a new table layout, and
  every read and write of an index written before it fails with `vector.ErrLegacySchema`
  (`vector: index has tables of the schema before this version: index "…" (sqlite): …`). Drop
  the index with `Store.DeleteIndex`, or open the store `WithGeneration` for a table set
  of its own, and index again. That is all a v0.1.1 search index needs; the next three
  points are for one written, with the new tables, by a version in between.
- **A collection records its model, and a model's name says more.** `Model()` of the
  `provider/yzma` and `provider/onnx` embedders is the identity of their vectors: the
  name followed by the settings that decide a vector
  (`google/embeddinggemma-300m;pooling=mean;prefixes=…;max_tokens=512`; for `Vision` the
  context size too, for ONNX the picture preparation). A collection that holds vectors
  under another string answers `vector.ErrModelMismatch` until it is emptied
  (`Index.Reset`, `MediaIndex.Reset`).
- **A stored hash covers what shapes the vectors, not only the content:** the chunker's
  settings (`markdown/2 max=1200`), the `Sampler`'s (`ffmpeg/2 …`) or the video model's
  segmenting (`yzma-vision/2 …`), the model and its dimensions. A hash stored without
  them matches none of today's, so each document and video is embedded once more the
  next time it is indexed, and after that only when one of these changes.
- **Pictures are embedded once more** as well: a JPEG is now turned upright by its EXIF
  orientation first, and every image's hash says so (`image/2`).
- **The Tagger's cache keeps its table**, and its entries stay good under the
  `TaggerConfig.Model` they were written with. The samples now pass `emb.Model()`, the
  identity above, where they passed `yzma.DefaultModel`: to the cache that is another
  model, so a Tagger changed to it embeds what it needs again, as it goes, and
  `Tagger.Prune` later drops the old rows.
- **Videos through `yzma.Vision`** are cut by the video stream's length and capped:
  `MaxSegments` 0 used to mean every segment and now means 720 (`DefaultMaxSegments`), so
  set it for videos of more than four hours.

## Tagger: long items

With `ChunkBytes` set, a long item used to be cached as the mean
of its pieces under the model marked with the chunk size (`<model>+chunks=<n>`); now each
piece is cached under the model itself. The first call after the upgrade embeds the pieces
of every long item once and rebuilds the tags that have long members; `Prune` later drops
the old rows. Without `ChunkBytes` nothing changes.

## Tagger: from the Store to the Catalog

The Tagger used to keep its own copy of your tags, items and
links in a `tagging.Store`; now it reads them from your `Catalog`, so the application's
writes no longer go through it. `NewTagger(emb, store, cfg)` became
`NewTagger(emb, catalog, cache, cfg)`, and `SetTag`, `SetItem`, `TagByHand` and their
kin are gone; `Suggest` and `Scores` now take `(ctx, rev, text)`. No data needs to move:
the cache starts empty and every embedding is recomputed on first use. The old tables are
left behind for your application to drop, as nothing reads them any more:
`autotag_<first 16 hex of sha256(index)>_{tags,items,links}`, and
`autotag_<generation>_<hash>_{tags,items,links}` for generations. The example service's
old `data/autotag-<model>.sqlite` files can be deleted.

## Generator: long texts

The defaults changed: `MaxBytes` went from 2 KB to 64 KB, so a long text
that took one call now takes up to 11, and yzma's default chat context went from 2048
to 4096 tokens. `GeneratorConfig{MaxBytes: 2048}` restores the earlier behaviour: one call
on the first 2 KB. With a chat context under 4096 tokens keep `MaxBytes` at or under
`ChunkBytes`: a 6 KB window, in German about 2,500 tokens, may not fit, and then that
window fails (twice, see "A bad answer is asked for again" in
[Generating tags](tagging.md#generating-tags)). If you never tag long texts, set
`CompleterSpec.ContextTokens` to 2048 to keep the smaller KV cache.

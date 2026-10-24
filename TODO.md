# TODO

## autotag

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
  libffi8 a startup requirement of every binary that imports semsearch/yzma,
  even one that never embeds anything. Ask for a lazy load that returns an
  error on first use instead.
- [ ] yzma: bind ggml_set_abort_callback
  llama.cpp aborts the process on a failed assertion (e.g. a batch larger than
  n_ubatch); the symbol exists in build b11429, and a binding would let
  semsearch/yzma turn such asserts into errors. Until then the package
  validates every batch itself.

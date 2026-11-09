# Semantic search

Indexing documents and searching them by meaning with `search.Index`, the vector stores, model identity, and loading models lazily.

```go
cfg := yzma.Config{
    ModelsDir: "data/models",
    Runtime: yzma.Runtime{
        LibDir:        filepath.Join("data", "llama-cpp", yzma.LibTag), // named after the pinned build
        ManagedLibDir: true,                                            // download the pinned build into it
    },
}
emb, err := yzma.New(ctx, cfg) // cfg.Model "" is yzma.DefaultModel; yzma.Models() lists the others
store, err := sqlitevec.New(ctx, db, emb.Dimensions())

// index (e.g. a tenant) is the hard isolation boundary; partition and collection
// locate the documents within it
ix, err := search.New(emb, store, "tenant-1", "space-7", "notes", search.WithMinScore(0.3))
err = ix.Index(ctx, "note-42", markdownText) // a no-op while the text is unchanged
res, err := ix.IndexBatch(ctx, docs)         // many at once: res.Indexed, res.Unchanged, res.Failed
hits, err := ix.Search(ctx, "how long to boil spaghetti", 10) // []search.Match: DocID, Score, Seq and Text of the best chunk
more, err := ix.SearchIndexed(ctx, "note-42", 10)            // "more like this" with the stored chunks: no model runs
for d, err := range ix.Docs(ctx) { /* d.ID, d.Hash: what is indexed, by id */ }
err = ix.Delete(ctx, "note-7", "note-9")                     // e.g. what was deleted while the app was offline
err = ix.Reset(ctx)                                          // empty the collection, e.g. to rebuild it with a new model
```

`Search` and `SearchIndexed` return a `search.Match` per document: its best chunk, by
position and text (a snippet), and its score. `SearchIndexed` searches with the stored
chunks of the document, in one store search, and scores each other document by its best
chunk against any of them; the document itself is never among the hits, and an id with
nothing stored is `search.ErrNotIndexed`. A document of more than 64 chunks
(`search.DefaultMaxQueryVectors`; `WithMaxQueryVectors` changes it) searches with 64 of
them, evenly spaced from the first, so the cost stays bounded: the store scores every
chunk in scope against every query vector in one pass. Over 20,000 chunks of 256
dimensions (`BenchmarkSearchVectors`) a search of 64 vectors takes about 0.6 s on
SQLite and 0.12 s on Postgres, against 0.03 s and 0.013 s for one vector. Every error the package returns starts with `search:`, and an
empty document id anywhere is `search.ErrEmptyID`, refused before a model or the store
is asked anything. An id that is not valid UTF-8 or holds a NUL byte (a Latin-1 file
path, say) is refused the same way with `vector.ErrInvalidID`, which `ErrEmptyID`
wraps: no backend can keep such an id alike (Postgres refuses it, SQLite's JSON id
lists mangle it), so the stores refuse it everywhere it enters, and in `IndexBatch` it
fails only its own document. `search.New` and `search.NewMediaIndex` refuse an empty
index (`vector.ErrEmptyIndex`) and a partition or collection of such bytes up front.

`IndexBatch` indexes many documents (`[]search.Doc{ID, Text}`) the way `Index` does
one, in bounded batches: one hash read for all of them, at most 32 chunk texts per
embedder call, one store write per such batch. Unchanged documents cost only their share
of the hash read, so it suits a full re-index at startup. The `BatchResult` lists what
was `Indexed`, what was `Unchanged` and what `Failed` for a reason of its own (a chunk
that embeds to the zero vector), by id; such a failure does not stop the others. An
embedder or store error does: it is returned, `Indexed` is exactly what was stored before
it, and the same call again picks up where it stopped.

Text is cleaned before it is chunked and hashed: invalid UTF-8 becomes U+FFFD and NUL
bytes are dropped, which a Postgres `TEXT` column refuses and SQLite takes, so both
backends store the same. A blank query finds nothing, and so does one that embeds to the
zero vector, which has no direction to rank by; a document chunk that embeds to it fails
that document, by name.

"Unchanged" means the hash stored with the document matches, and that hash covers
everything that shapes its vectors: the text, the chunker's settings and the embedding
model (its name, when it implements `sense.Identified`, and its dimensions; a change of
a named model is refused before anything is re-indexed, see below). A chunker
describes its settings through the optional `search.Fingerprinter`; `MarkdownChunker`
does (`markdown/2 max=1200`), and one without it is described by its Go type name, so
swapping chunkers re-indexes and changing a custom one's settings does not. The
`MediaIndex` hashes cover the model in the same way, and a video's also the `Sampler`'s or
the `VideoEmbedder`'s `Fingerprint`, when it has one. `yzma.Vision` has: its segment
length, its cap on segments and the version of its segmenting
(`yzma-vision/2 segment=20s max_segments=720`), so a video indexed under one cap is
indexed again under another.

`MarkdownChunker`, the default (`WithChunker` replaces it), cuts a document at its
headings and packs its paragraphs into chunks of at most `MaxChars` bytes (1200), each
prefixed with its heading path ("Title › Section"), which counts towards that size and is
cut to a quarter of it. A heading with no text under it and no subheading (a list of
`## Milk`, `## Eggs` headings) is a chunk of its heading path alone, so its words stay
searchable without being folded into a section they do not describe. A heading is an
ATX `#` line indented by at most three spaces and outside fenced code, so a `# comment`
in a code block is never one. CRLF text and blank lines holding spaces chunk like plain
ones, and a paragraph too long for one chunk is split between lines, so code and lists
keep their shape, and only a line too long for one chunk between words.

Every `Store` backend keeps one physical table set per index, so one index's vectors are
never queryable from another's. Within an index a search sees exactly one collection,
the one at its partition and collection: an empty partition or collection is a value
like any other, not a wildcard. A `vector.Query` opens a coordinate only when asked
(`AllPartitions`, `AllCollections`), and can keep or leave out given document ids.
`Store.Search` ranks documents, not chunks: it returns the best chunk of each of the
best documents, so one long document or one many-framed video never crowds out the
rest, in score order, then by document id, then by chunk. A query carries one or more
vectors (`Query.Vectors`): a chunk scores its best cosine against any of them, in one
pass over the chunks, which ranks exactly what one search per vector merged by best
match would, and `Match.Query` says which vector a hit matched. The SQL stores hand all
the vectors over as one parameter (a BLOB of packed float32s on SQLite, a `vector[]` on
Postgres), whatever their number. `DeleteIndex` removes an
index with everything in it, e.g. when a tenant leaves, in one transaction (as an
index's tables are also created in one), so no Store ever sees half an index; another Store on the same
database (another process, a maintenance CLI) notices: its reads of that index find
nothing and its next write creates it again. The first write of a new index never
stalls the store's calls on other indexes, and a call waiting for an index another
call is creating gives up when its context does. Every method checks its
arguments first, alike in every backend, and an error of the database underneath comes
back wrapped with `vector:`, the backend and the operation (`vector: postgres upsert:
…`).

Vectors of different models never mix. A store refuses a vector of the wrong length
(`vector.ErrDimensions`) and a zero, NaN or infinite one, or one whose norm underflows or
overflows in float32, the precision the backends compute in (`vector.ErrInvalidVector`); a
SQL store opened with other dimensions than an index's tables hold refuses that index.
When the embedder names its model (`sense.Identified`), `search.New` and
`search.NewMediaIndex` pass that name on: a write to a collection records it, and while
the collection holds documents a write or search with another model fails with
`vector.ErrModelMismatch`. An embedder that names no model is not checked.

A model change on the same generation therefore goes through emptying the collection:
`IndexBatch`, `IndexImage`, `IndexVideo` and `IndexFrames` ask the store for the
collection's model (`Store.Model`) before they embed anything, so a mismatch costs no
model time, and fail with `vector.ErrModelMismatch`; `Index.Reset` / `MediaIndex.Reset`
(`Store.DeleteCollection`) empty the collection, and the next indexing run rebuilds it
with the new model. A collection that holds no documents holds no model, whichever way
it was emptied. Every partition of a collection shares its model, so a collection with
documents left in other partitions keeps its model until those are emptied too. To keep
serving the old vectors while the new ones are built, use a generation instead (below).

A store opened with `WithGeneration` keeps further table sets of its own, so a migration
can build a second model's vectors next to the current ones before switching over; the
new generation's store then retires the old one with `DropGeneration`. The SQL stores
refuse the tables of the schema before this version with `vector.ErrLegacySchema`:
delete the index (`DeleteIndex`) or build a new generation, and index again.

The first `yzma.New` downloads the llama.cpp libraries (~18 MB) and the default model,
`google/embeddinggemma-300m` (~329 MB, 768 dimensions, multilingual); both are pinned by
digest. `ctx` bounds the downloads: with no deadline of its own it gets a default one
(15 minutes here, an hour for the chat and vision models, 30 minutes in `provider/onnx`),
and a deadline you set stands, shorter or longer. A download that is cut short keeps what
it got in `<file>.partial` and the next call asks the server for the rest; the whole is
verified against the digest either way. (That needs file locks, which keep processes
after one file apart: on a system without them, none of which has a pinned build, each
process writes a `<file>.partial-<pid>` of its own and a failed download starts over.) A model file that is not the pinned size (a
crash, a full disk) is fetched again instead of being loaded. To keep all that off your
startup path, wrap the loader in a `LazyEmbedder`, which reports its model and
dimensions from the model's spec without loading anything:

```go
spec, err := yzma.LookupSpec(cfg.Model, cfg.ModelsDir) // the spec yzma.New(ctx, cfg) loads
emb := sense.NewLazyEmbedder(spec.ModelID(), spec.Dims, func(ctx context.Context) (sense.Embedder, error) {
    return yzma.New(ctx, cfg) // ctx is the LazyEmbedder's own: only Close cancels it
})
defer emb.Close()

err = emb.Load(ctx)        // optional: load now, e.g. in the background at startup
state, err := emb.State() // not started, loading, loaded, failed or closed, for a health endpoint
```

`LazyImageEmbedder` and `LazyCompleter` do the same for an image model and a chat model.
All three behave alike:

- **One load, shared.** The first call (or `Load`) starts it on a goroutine of its own
  and every caller waits for it under its own context: a request that times out
  returns, and the load goes on for the others. The load's context belongs to the lazy
  value and is cancelled by `Close` alone.
- **A failed load is kept.** The error is returned by every later call and nothing
  loads again, so a model that cannot be loaded costs one attempt, not one per request.
  A loader that panics (`*sense.LoadPanicError`, with the stack) or returns nil fails
  the same way. To try again once the cause is fixed, build a new lazy value
  (`sense.NewLazyEmbedder(...)` again), use it in place of the failed one and `Close`
  that one: a lazy value never retries on its own.
- **The model is checked.** The identity and dimensions given at construction are what
  the stores record before anything loads (an empty identity is unknown and unchecked).
  A loaded model with other dimensions, or whose `Model()` is another string, fails the
  load and is closed instead of mixing two vector spaces. So pass the identity the
  loaded model will report, which each provider computes from its spec without loading
  (`yzma.ModelSpec.ModelID()`, `yzma.VisionSpec.ModelID(opts)`,
  `onnx.ModelSpec.ModelID()`), not the spec's bare `Name`.
- **`Close` releases the model.** It cancels a load in progress and waits for it, closes
  the loaded model when it has a `Close`, and makes every waiting and later call fail.

`Model()` on `yzma.Embedder`, `yzma.Vision` and `onnx.Embedder` is the identity of the
vectors they produce, the string to store next to them: the model's name followed by
what else decides a vector (the pooling, the prefixes and the token cap for a text
model; the instruction, the image token bounds, the video sampling and the context size
for `Vision`; the tensor and the picture preparation for an ONNX model). The rule is
one: a setting that can change the vector of some input is in the identity, so
`ModelSpec.MaxTokens` and `VisionOptions.ContextTokens` are, because a text longer than
them is cut there, and changing either means embedding again. A setting that changes
which segments a video yields without changing any of them (`VisionOptions.MaxSegments`)
is in `Vision.Fingerprint()` instead, which a `MediaIndex` hashes with every video; what
changes neither (threads, the accelerator, directories) is in nothing. The identity is
the same on every run and changes
whenever a vector would, so vectors stored under another identity are stale; the
specs' `ModelID` gives the same string before the model loads. A call on
a closed provider, lazy loader or `Tagger` returns an error that wraps `sense.ErrClosed`.

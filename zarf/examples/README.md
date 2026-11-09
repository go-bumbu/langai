# langai examples

A single-user web service to exercise langai's features by hand, one page per
feature: **media search** (images and videos that look like an image, a video
or a text), **text search** (documents that answer a question, in any
language, with the chunk that matched), **duplicates** (exact and near-exact
copies of pictures, by perceptual hash and no model), **auto tagging** (tags
for notes, picked among your own tags or written new by a chat model), and a
**status** page that says what is loaded and how. Every page is a debug view:
it shows the numbers behind every result.

## Run it

```
cd zarf/examples
make prepare   # sample photos, copies of some and two clips into data/media, sample documents into data/docs
make run       # index both, serve the pages on http://127.0.0.1:8085/
```

From the repository root, `make example-prepare` and `make example` do the same.

Needs Go, ffmpeg and ffprobe on PATH (the video sampler), curl (for `prepare`)
and, as `vector/sqlitevec` is CGO, a C compiler and `libsqlite3-dev`, plus the
libraries `provider/yzma` loads (`libffi8 libgomp1 libstdc++6`). The first
`make run` downloads the ONNX Runtime build (~11 MB) and the SigLIP 2 model
(~370 MB) into `~/.cache/semsearch-onnx`, and the llama.cpp build (~18 MB) and
EmbeddingGemma (~330 MB) into `~/.cache/semsearch-yzma`: the directories
`make test-images` and `make test-models` use too, so nothing downloads twice.

`make prepare` fetches twenty photos from picsum.photos, makes copies of three
of them with ffmpeg (re-encoded, half size, brightened, with a mark stamped on,
as a PNG, and one file once more) for the duplicates page, renders two
slideshow clips, and writes the repository's own golden texts
(`zarf/eval/testdata`) as Markdown files: 84 short documents in English,
Spanish and German and 24 long ones with sections. Drop your own images (JPEG,
PNG, GIF, WebP, BMP, TIFF) and videos (MP4, MOV, MKV, WebM, AVI, M4V) anywhere
under `data/media`, your own `.md` and `.txt` files anywhere under `data/docs`,
and click **rescan** on the page, or restart: a file whose bytes are unchanged
is a no-op, a changed one is re-embedded, a removed one leaves the index.

The service is up at once. The models sit behind the library's lazy loaders
and the first scans run behind the server, so a restart over unchanged
directories serves every page within milliseconds and loads no model at all:
see [Models: what is lazy and why](#models-what-is-lazy-and-why).

Settings, as environment variables or `make run MODEL=facebook/dinov2-base`:

| Setting | Default | What it does |
|---|---|---|
| `MEDIA_DIR` | `data/media` | the directory the media search and duplicates pages scan, recursively; dot-directories are skipped |
| `DOCS_DIR` | `data/docs` | the directory the text search page scans for `.md`, `.markdown` and `.txt` files |
| `DATA_DIR` | `data` | the index databases, the catalog, the notebook and the thumbnail cache |
| `MODELS_DIR` | `~/.cache/semsearch-onnx` | the ONNX models and runtime |
| `MODEL` | `google/siglip2-base-patch16-224` | the media page's model: `facebook/dinov2-base` (also ONNX, videos as frames) or `qwen3vl` (Qwen3-VL-Embedding-2B through llama.cpp: text, images and videos as segments in one space); each model gets a database of its own |
| `PROVIDER` | `cpu` | `cuda` or `coreml` to run an ONNX model on a GPU |
| `ACCEL` | auto | what runs the GGUF model: empty picks a GPU when the machine has one (Vulkan, CUDA, Metal), `cpu`, `vulkan`, `cuda` or `metal` require it |
| `SEGMENT` | `20s` | how much of a video the vision model sees at once |
| `LLAMA_DIR` | `~/.cache/semsearch-yzma` | the GGUF files and the llama.cpp build: the text embedder, the chat model and the vision model, shared with `make test-models` and `make test-vision` |
| `TAG_MIN_SCORE` | `0.45` | the cutoff of the recommended tags on the auto tagging page; the ones under it show greyed |
| `MIN_SCORE` | `0` | media hits below it are dropped; 0 shows everything, which is what a debug view wants |
| `TEXT_MIN_SCORE` | `0` | the same for the text search page |
| `MAX_IMAGE_BYTES` | `0` | the most the media index reads of an image (`search.WithMaxImageBytes`); 0 is the library's 64 MiB. Set it low (`60000`) to see files listed as *too large* |
| `LIMIT` | `12` | hits per search |
| `POSTGRES` | empty | the DSN of a Postgres database with pgvector: the vectors and the tagging cache go there instead of into SQLite files, see [Postgres](#postgres) |
| `ADDR` | `127.0.0.1:8085` | listen address |

`make clean` removes the SQLite databases, the catalogs and the thumbnails;
the media, the documents, the notebook of the auto tagging page and the models
stay.

## Status: what to click

`http://127.0.0.1:8085/` is the home page, and says what is loaded and how.

- **Pages** links to the four feature pages with a line on what each shows.
- **Models** has a row per model: its name and what uses it, the identity the
  stores record with its vectors (`ModelID()` of its spec, e.g.
  `google/embeddinggemma-300m;pooling=mean;prefixes=…;max_tokens=512`), its
  dimensions, the provider that runs it and what it runs on, which a lazy
  model knows once it is loaded (`Vulkan0: AMD Radeon 890M Graphics`). The
  pill is the state of its lazy loader (`LazyEmbedder.State` and its kin):
  *not started*, *loading*, *loaded*, *failed* or *closed*. **load now** calls
  `Load`, and the row then says how long that took (0.4 s for EmbeddingGemma,
  0.7 s for SigLIP 2 on the test laptop). The row also says why the model is
  lazy, or why not.
- **A load that fails** turns its row red with the error the loader keeps and
  what to do about it. A lazy loader never retries: every later call returns
  the same error, so a model that cannot load costs one attempt. Fix the cause
  and restart; an application that must stay up builds a new lazy value and
  closes the failed one. Try it with `make run LLAMA_DIR=/dev/null/llama`: the
  service starts, the pages serve what is stored, and the first search turns
  the text model's row to *failed: yzma: Runtime.LibDir: not a directory*.
- **Collections** has a row per index a page keeps: the store and where it is,
  its coordinate (index, partition, collection), how many documents it holds,
  the model the store **recorded** for it (`vector.Store.Model`) and the model
  the page runs. When the two differ the row turns amber: the index refuses to
  write or search (`vector.ErrModelMismatch`) until **reset** empties the
  collection (`Index.Reset`, `MediaIndex.Reset`), after a confirmation; the
  page's next scan builds it again. The duplicates page's collections appear
  once a search has opened them: one per hash, in a `vector.MemStore`.
- **Embedding cache** is the Tagger's `tagging.EmbeddingCache`: where it
  lives, the size of the database that holds it (a cache has no count of its
  own), and what the last `Tagger.Prune` removed. **prune now** runs one.

## Media search: what to click

- **The grid** is the catalog: every file the scan found, a `video` badge on
  clips and a red `failed` badge (hover for the error) on files that could not
  be indexed; those are retried on every rescan. A file the library refuses
  for a reason it names shows that instead, in amber: *too large* for an image
  longer than `MAX_IMAGE_BYTES` (`search.ErrTooLarge`), *no frames* for a
  video the sampler takes no frame from (`search.ErrNoFrames`; a file ffmpeg
  cannot read at all, a sound file named `.mp4` say, is `failed` with
  ffmpeg's own error).
- **Click a thumbnail** to see the item large and, under *looks like*, the
  items closest to it. Each hit shows its cosine **score** with a bar; the
  item itself comes first, greyed and marked *self*, because its score says
  what "identical" scores like with this model (about 1.0). With SigLIP 2 a
  resized or greyscale copy scores above 0.9 and an unrelated picture well
  below 0.8.
- **Videos** are the frames the sampler picked (scene cuts plus one frame every
  few seconds, see `media/ffmpeg`), each embedded like an image. *Frames the
  index saw* is that sample, with each frame's time; click a frame to seek the
  player. A video hit says *matched at* the time of its best frame; when the
  open item is a video, every hit says which *query frame* found it. Click the
  time to seek the player there. The storyboard is the sampler run on the file
  as it is now; it equals what was indexed as long as the file is unchanged.
- **Search with a file** runs the same search for an image or a video that is
  not in the catalog: an uploaded copy of a catalog item is an ordinary hit
  scoring near 1.
- The footer of every search says how many query frames were embedded, how
  many hits came back and how long the search took, and the header line says
  which model, how many dimensions, which provider, where the model's lazy
  loader stands, the floor and what the last scan did.
- **The model loads late.** With an ONNX model, a start over an unchanged
  directory says *model not started* and stays that way while you click
  thumbnails: the scan's no-ops and the stored vectors need no model. Tick
  **re-embed on click**, upload a file or add one and rescan, and it loads
  (0.7 s) and the header says *model loaded*.
- **A database built by another model** is refused, not mixed: the scan stops
  at its first file and a banner says which model the store holds and which
  the page runs, with **reset the index and rescan**
  (`MediaIndex.Reset`). Each model name has its own database, so this takes a
  change of the model's identity under one name: index with
  `make run MODEL=qwen3vl`, then start with `MODEL=qwen3vl SEGMENT=10s`.

### With the vision model

`make run MODEL=qwen3vl` switches to Qwen3-VL-Embedding-2B (a 2.7 GB download on first
run, into `LLAMA_DIR`). The header shows what runs it (`Vulkan0: AMD Radeon 890M
Graphics` or `cpu`) and the segment length, a **search by text** box appears, and the
scores read differently: unrelated items sit near 0.1 instead of 0.7, a description
against its picture around 0.65. Videos are indexed as segments, so a hit says
*matched in the segment at 20.00 s* and the storyboard labels each sampled frame with
its segment; the clip is one vector per segment, which is what lets "a black dog" find
the clip and what makes a clip's score against one of its photos lower (about 0.5 for
a four-shot slideshow) than the frame route's. Expect about 1.5 s per photo on the
laptop's iGPU and 7 s on the CPU, so a large library takes a while the first time;
later runs skip unchanged files.

## Text search: what to click

`http://127.0.0.1:8085/textsearch/` is `search.Index` over the files of
`data/docs`, embedded by EmbeddingGemma 300M.

- **The scan** is two calls. `Index.IndexBatch` gets every document and embeds
  only the changed ones; the line over the list is its `BatchResult`: how many
  it *indexed*, how many were *unchanged*, and how many *failed* for a reason
  of their own, each named with that reason under it (a file whose name is not
  valid UTF-8 fails with `vector.ErrInvalidID`, and the others go on). Then
  the ids `Index.Docs` lists and the directory no longer holds leave in one
  `Index.Delete`: the *removed* count. The 108 samples are 636 chunks and take
  14 s the first time on the test laptop's iGPU; **rescan** over an unchanged
  directory takes 7 ms and never reaches the model.
- **Search** in any language: a hit is a document, scored by its best chunk,
  with that chunk's text under it and, in blue, its heading path (`The honey
  bee › The colony and its castes`), which the chunker puts in front of every
  chunk so that it is embedded with it. The chips under *try* are questions of
  the golden set that cross a language: `my sourdough starter is not rising`
  finds the English text at 0.75 and the German and Spanish bread recipes at
  0.50 and 0.47; a German question about hut-to-hut hiking finds the English
  guide at 0.68 ahead of the German one. Unrelated texts sit around 0.3 to
  0.4: `TEXT_MIN_SCORE` drops them.
- **Click a document** (in the list, or a hit's title) to see it as the index
  does: cut into the chunks of `search.MarkdownChunker`, one vector each, with
  each chunk's heading path and size. From a hit, the chunk that matched is
  framed. Above the chunks, **more like this** is `Index.SearchIndexed`: the
  document's stored chunk vectors are the query, so no model runs, and the
  line says so. A document of 33 chunks searches with all of them in 40 ms.
- **The header** has the embedder's identity (what the store records), its
  dimensions, where its lazy loader stands, the chunker's fingerprint
  (`markdown/2 max=1200`, hashed with every document, so a change of chunker
  indexes them again), the counts and the store.
- **Another model's index** shows as on the media page: a banner with what
  the store holds, and **reset the index and rescan** (`Index.Reset`).

## Duplicates: what to click

`http://127.0.0.1:8085/duplicates/` finds the copies among the pictures of
`data/media` with `provider/phash`: no model, nothing to download. It shows
both faces of that package.

- **The groups on the left** come from the raw hashes. The scan decodes every
  picture once and takes all four hashes from it (`Embedder.Hash`; 40 pictures
  in half a second, nearly all of it decoding). Every pair is compared with
  `Hash.Distance`, the Hamming distance, and pictures at most **threshold**
  bits apart are linked into a group: 780 pairs in about 0.1 ms, with no store
  and no vectors. A group whose hashes are all equal is marked *exact
  duplicates*, the others *near-exact*, and the table under each group is the
  distance between every two members. The summary line also gives the closest
  pair that is *not* in a group, which says how far the threshold is from
  linking unrelated pictures.
- **Try two thresholds.** At 4 bits of the 64-bit pHash the samples make six
  groups: the re-encoded, half-size, brightened, PNG and greyscale copies sit
  0 to 2 bits from their originals. The copies with a mark stamped on are 2, 8
  and 10 bits away, so 4 bits leaves two of them out and 10 takes them all,
  while the closest unrelated pair is 22 bits apart.
- **hash** and **bits** switch between pHash and dHash, 64 and 256 bits; each
  starts at its own threshold. dHash moves least under a local edit: the
  marked copies are within 2 bits of 64. 256 bits put more room between
  copies and the rest: with the 256-bit dHash every copy, marked or not, is
  within 6 bits and the closest unrelated pair 90 bits apart.
- **Click a picture**, or **find copies of a file** of your own, to look the
  copies up the other way: in a `search.MediaIndex` over the same hash as an
  `ImageEmbedder`, exactly as a model's vectors would be searched
  (`SearchIndexed` for a picture of the catalog, `SearchImage` for an upload).
  Every hit shows the relation: `cosine 0.6875 = 1 − 2·10/64 → 10 bits`, and
  next to it the distance of the two raw hashes, which agrees (✓). Hits
  further than the threshold are greyed: they are what a floor
  (`search.WithMediaMinScore`) would drop.
- Each hash is a vector space of its own (`phash-64`, `dhash-256`…), so each
  gets its own collection, opened by the first search that asks for it, in a
  `vector.MemStore`: hashing is fast enough that nothing is worth keeping
  between runs. The status page lists them.
- A JPEG with an EXIF orientation is the one case where the two faces differ
  (≠ instead of ✓): the index turns it upright before hashing, as it does for
  a model, and a raw `Hash` takes the pixels as decoded.

## Auto tagging: what to click

`http://127.0.0.1:8085/autotag/` is a notebook of notes and tags, and shows
both ways `tagging` tags a text:

- **Existing tags** (`tagging.Tagger`, EmbeddingGemma 300M, ~330 MB, loaded on
  the first note the cache does not hold; the header says *not started* or
  *loaded*). Every note shows, under *auto*, the notebook's tags the Tagger picks
  for it, as dashed chips; **+** puts one on by hand. Tags put on by hand show
  solid, with a ✓ when the Tagger picks them too. A hand tag teaches its tag:
  the tag's vector moves from its name toward the notes it was put on (their
  mean), so rename a tag to a full path (`cooking / baking`) to help a tag
  with no examples yet. Tag names are read as a tree: of `dev` and `dev / go`,
  both over their cutoffs, only the one the note clears by more is picked, and
  a child has to beat its parent to be picked at all, so a Go note gets
  `dev / go` and a general programming note `dev`, not both. Every chip shows
  the note's score against the tag, and **scores** lists the note against
  every tag, each bar marked at that tag's cutoff, so a tag that was not
  picked says by how much it missed, or which tag on its path fits better. The left
  column counts, per tag, the notes tagged by hand and the ones it is picked
  for; click a tag to filter. Picking for every note reads cached vectors
  only, so the whole list takes milliseconds.
- **Long notes.** EmbeddingGemma reads 512 tokens, about 2.2 KB of English, and
  drops the rest. A note over 2,000 bytes is embedded in parts
  (`TaggerConfig.ChunkBytes`), and its header says *read in N parts*. A note of
  three parts or more scores each tag by its two best parts, so a section about
  a tag counts in full and a passing paragraph does not; a note of two parts by
  the better half or the two together. A long note tagged by hand teaches the
  tag with its part closest to the tag's name.
- **Recommended tags** (`tagging.Generator`). **recommend new tags** has Qwen3 4B
  Instruct (2.5 GB, loaded on the first click, about 4 s per note on the laptop's
  iGPU) write up to ten tags, each scored by the embedder against the note:
  every one comes back with its score and a bar marking the cutoff
  (`TAG_MIN_SCORE`), the ones under it greyed. A tag that matches one of yours
  by name or leaf (`baking` and `cooking / baking`) says so and **+ tag** uses
  it; **+ create** makes it a new tag and puts it on the note. Two controls in
  the header shape them: **recommend in** writes the tags in the note's own
  language (the page's default), or in English, Spanish, German, French or
  Italian (`tagging.WithLanguageTags`); the page remembers the choice. Asked
  for English, the 55 German and Spanish samples got English tags for all but
  about three notes (the tortilla and train-travel notes stayed largely in
  their own language), with place and dish names kept as they are. With **as a
  tree** and English tags in the notebook, the tags come back English even in
  the note's language: pick the notebook's language there. **as a tree** has the model write paths
  that extend your tags (`tagging.WithTreeTags`, with the notebook's tags as
  the tree): the levels that are your tags already show grey and the new ones
  bold, and **+ create** makes the missing levels too, taking the spelling of
  the ones that exist. With the golden tags as the tree, a Spanish note about Go
  errors got `dev / go / error wrapping` and siblings.
- **warm** calls `Scope.Warm` with every note: the Tagger comes up to the
  notebook's revision and embeds what its cache lacks. Over a full cache that
  is a millisecond and no model call, which is what the cache is for; after a
  switch of embedding models it embeds everything once. The service does the
  same at startup.
- **A tag built without one of its notes** shows *⚠ incomplete* in the tag
  list (`Scope.Incomplete`): the embedder could not take that note's text, so
  the tag is built from its name and its other notes until the notebook next
  changes, and the note itself shows the error.
- **load samples** imports the eval golden set (`../eval/testdata/golden.yaml`):
  15 tags and 84 documents in English, Spanish and German, with no hand tags,
  so every tag is judged by its name. Each note shows its *expected* tags,
  green when the note has them and amber when not.
- **load samples, partly tagged** imports the same and puts each tag by hand
  on the first two documents that expect it (30 hand tags, all on English
  documents, as the file lists them first), so the tags learn from examples.
  On samples loaded already it only adds those hand tags, which turns the
  baseline into the trained state in place.

  On the 54 notes left untagged, names alone find 46 of their 56 expected
  tags with 14 extra picks; two examples per tag find 51 with 6 extra, and on
  the 49 Spanish and German ones among them, taught by English examples, 46
  of 51 with 5 extra. Before the Tagger scored a note by its mean similarity
  to a tag's examples and read the tags as a tree, the same two examples per
  tag found 34 of 56: the cosine to the examples' centroid rises with every
  example added, so the fixed cutoff it was judged by suited five examples
  and was too strict for two. Nearly every extra pick is a related tag,
  `cooking / baking` for a recipe, not an unrelated one.

The notebook (`data/notebook.json`) owns the notes, tags and hand tags. The
application provides its tags and hand-made links through a `Catalog` (here
the notebook itself, read when it changes); the Tagger keeps only an embedding cache (`MemCache`,
`tagging/sqlitecache`, `tagging/pgcache`) and an in-memory index of tag
vectors. Here that cache is `data/autotag-cache.sqlite`, one for every
embedding model (the model is part of every key), so `make clean` or another
model only costs embedding the notes again. The service prunes it with
`Tagger.Prune` at startup and once a day (`Notebook.KeepCachePruned`): what no
note or tag used for 30 days goes, the texts of edited and deleted notes and
another model's rows; the log and the status page say how many embeddings
that was.

## Models: what is lazy and why

The service uses the library's lazy loaders (`langai.LazyEmbedder`,
`LazyImageEmbedder`, `LazyCompleter`) and not a wrapper of its own. Each is
built from the model's spec: its `ModelID()` and dimensions are what the
stores and the Tagger's cache are set up with before anything loads, and what
the loader checks the loaded model against.

| Model | Lazy | Why |
|---|---|---|
| text embedder (EmbeddingGemma, llama.cpp) | `LazyEmbedder` | One instance serves the text search page, the Tagger and the Generator's scoring. A scan of unchanged documents is one hash read and cached notes are read from the cache, so it loads on the first new text or query. |
| chat model (Qwen3 4B Instruct) | `LazyCompleter` | 2.5 GB that only **recommend new tags** needs. |
| image model (SigLIP 2, DINOv2, ONNX) | `LazyImageEmbedder` | The media page scans at startup, but a file whose bytes are unchanged never reaches the model and "looks like" searches with stored vectors, so a restart over an unchanged library serves the whole page without it. A new file, an upload or a re-embed loads it. |
| vision model (Qwen3-VL) | no, loaded before the service starts | It is the index's image, video and text embedder in one object. The library has no lazy loader for a `VideoEmbedder`, and the fingerprint the index hashes with every video (`Vision.Fingerprint`) exists on the loaded model only. |
| perceptual hashes | nothing to load | A hash is a few milliseconds of integer arithmetic. |

A load failure is kept by the loader and shown on the status page with the
way out; the service keeps running on what is stored.

## Postgres

```
docker run -d --name langai-pg -e POSTGRES_PASSWORD=langai -e POSTGRES_DB=langai \
    -p 127.0.0.1:5432:5432 pgvector/pgvector:pg18
make run POSTGRES='postgres://postgres:langai@127.0.0.1:5432/langai?sslmode=disable'
```

With `POSTGRES` set (or `-postgres`), the vector stores are `vector/pgvec` and
the Tagger's cache is `tagging/pgcache`, all in that one database; without it
they are `vector/sqlitevec` and `tagging/sqlitecache` in files under
`DATA_DIR`. The pages are the same and so are the scores: only the store
differs. What is a database file per model with SQLite is a generation of
tables per model with Postgres (`pgvec.WithGeneration`, named after a digest
of the model's name), so two models never share tables and switching back to
a model finds its index as it was. The first run embeds everything again, as
the two backends share nothing; `make clean` does not touch the database: use
**reset** on the status page, or drop it. The duplicates page stays in memory
either way.

## How it is built

```
zarf/examples/
├── main.go           flags, wiring of the pages (ffmpeg, search.Index, search.MediaIndex), the server
├── models.go         the models behind the library's lazy loaders (yzma, onnx)
├── stores.go         the vector stores and the tagging cache: SQLite files or Postgres
├── tagging.go        wiring of the auto tagging page
├── status.go         wiring of the status page
├── status/           Service: models, collections, the cache; Handler: the API and the home page
│   └── ui/           index.html, app.js, style.css, embedded in the binary
├── mediasearch/      Library: scan, catalog, search, thumbnails, storyboards; Handler: the API and the page
├── textsearch/       Library: scan, catalog, search, a document's chunks, the samples; Handler
├── duplicates/       Finder: hashes, groups, the MediaIndex over a hash; Handler
├── autotag/          Notebook: notes, tags, hand tags, the Tagger's Catalog; Handler
├── assets/           what the pages share: base.css, and ui.js with the navigation and DOM helpers
├── prepare.sh        the sample data
└── data/             git-ignored: media/, docs/, index-<model>.sqlite, index-<model>.catalog.json,
                      thumbs/, notebook.json, autotag-cache.sqlite
```

Every page package has a plain Go type that does the work and a `Handler`
that serves it with its embedded page; `main.go` only wires. They import the
library's ports, `search` and `tagging`, never a provider, so their tests run
on `langaitest`'s fakes and `vector.MemStore`; the real models are wired in
`models.go` alone. A new example is a new package with its own `Handler` and
page, mounted in `main.go` next to the others, with a line in `assets/ui/ui.js`
for the navigation.

The status API, under `/`: `GET api/status` (everything the page shows),
`POST api/models/<key>/load` (the model's status after the load; a failed load
is a state, not a failed request), `POST api/collections/<key>/reset`,
`POST api/cache/prune`.

The text search API, under `/textsearch/`: `GET api/info`, `GET api/docs`,
`GET api/search?q=&limit=`, `GET api/similar/<id>?limit=` (more like this),
`GET api/doc/<id>` (the text and its chunks), `POST api/rescan` (409 while a
scan runs; the answer is the scan's stats), `POST api/reset`. Ids are paths
relative to the documents directory.

The duplicates API, under `/duplicates/`: `GET api/info`,
`GET api/groups?kind=&threshold=` (`kind` one of `phash-64`, `dhash-64`,
`phash-256`, `dhash-256`; the threshold in bits), `POST api/search?kind=&threshold=&limit=`
(multipart `file`), `GET api/similar/<id>?kind=&threshold=&limit=`,
`POST api/rescan`, `GET file/<id>`.

`autotag` takes a `tagging.EmbeddingCache` and builds its own Tagger and
`tagging.Generator`, so its tests run on a `tagging.MemCache`,
`langaitest.FakeEmbedder` and a canned chat model. Its
API, under `/autotag/`: `GET api/info`, `GET api/notebook` (every tag and note
with its automatic tags), `POST api/tags` (`{"name"}`; an existing name returns
that tag), `PUT|DELETE api/tags/<id>`, `POST api/notes` (`{"title","text"}`),
`DELETE api/notes/<id>`, `PUT|DELETE api/notes/<id>/tags/<tag>` (hand tags),
`GET api/notes/<id>/recommend?lang=<BCP 47>&tree=1`, `POST api/tags` takes
`"parents": true` to create a path's missing levels, `POST api/samples?hand=<n>` (n hand tags
per tag, 0 by default), `POST api/warm`.

`mediasearch` imports only the library's ports and `search`, so its tests run
on `langaitest.FakeImageEmbedder`, a `vector.MemStore` and a fake sampler; the
real model and ffmpeg are wired in `main.go` and `models.go` alone.

The API, under `/mediasearch/`: `GET api/info`, `GET api/items`,
`GET api/similar/<id>?limit=&embed=` (by default the item's stored vectors
are the query and no model runs; `embed=1` embeds the file again, the
page's **re-embed on click** box), `POST api/search` (multipart `file`),
`GET api/text?q=` (with the vision model), `POST api/rescan` (409 while a
scan runs, or when the index holds another model's vectors, then with
`"mismatch": true`), `POST api/reset`, `GET api/frames/<id>`,
`GET thumb/<id>`, `GET storyboard/<id>`, `GET file/<id>`. Ids are paths
relative to the media directory and only catalog keys are served.

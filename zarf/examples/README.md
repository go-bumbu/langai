# langai examples

A single-user web service to exercise langai's features by hand, one page per
feature: **media search** (images and videos that look like an image or a
video, with the numbers behind every hit) and **auto tagging** (tags for notes,
picked among your own tags or written new by a chat model).

## Run it

```
cd zarf/examples
make prepare   # sample photos from picsum.photos and two slideshow clips into data/media
make run       # index data/media, serve the pages on http://127.0.0.1:8085/
```

From the repository root, `make example-prepare` and `make example` do the same.

Needs Go, ffmpeg and ffprobe on PATH (the video sampler), curl (for `prepare`)
and, as `vector/sqlitevec` is CGO, a C compiler and `libsqlite3-dev`. The first
`make run` downloads the ONNX Runtime build (~11 MB) and the SigLIP 2 model
(~370 MB) into `~/.cache/semsearch-onnx`, the directory `make test-images` uses
too, so nothing downloads twice. Drop your own images (JPEG, PNG, GIF, WebP, BMP,
TIFF) and videos (MP4, MOV, MKV, WebM, AVI, M4V) anywhere under `data/media`
and click **rescan**, or restart: a file whose bytes are unchanged is a no-op,
a changed one is re-embedded, a removed one leaves the index.

Settings, as environment variables or `make run MODEL=facebook/dinov2-base`:

| Setting | Default | What it does |
|---|---|---|
| `MEDIA_DIR` | `data/media` | the directory scanned, recursively; dot-directories are skipped |
| `DATA_DIR` | `data` | the index database, the catalog and the thumbnail cache |
| `MODELS_DIR` | `~/.cache/semsearch-onnx` | models and runtime |
| `MODEL` | `google/siglip2-base-patch16-224` | `facebook/dinov2-base` (also ONNX, videos as frames) or `qwen3vl` (Qwen3-VL-Embedding-2B through llama.cpp: text, images and videos as segments in one space); each model gets a database of its own |
| `PROVIDER` | `cpu` | `cuda` or `coreml` to run an ONNX model on a GPU |
| `ACCEL` | auto | what runs the GGUF model: empty picks a GPU when the machine has one (Vulkan, CUDA, Metal), `cpu`, `vulkan`, `cuda` or `metal` require it |
| `SEGMENT` | `20s` | how much of a video the vision model sees at once |
| `LLAMA_DIR` | `~/.cache/semsearch-yzma` | the GGUF files and the llama.cpp build, shared with `make test-vision` and the auto tagging page |
| `TAG_MIN_SCORE` | `0.45` | the cutoff of the recommended tags on the auto tagging page; the ones under it show greyed |
| `MIN_SCORE` | `0` | hits below it are dropped; 0 shows everything, which is what a debug view wants |
| `LIMIT` | `12` | hits per search |
| `ADDR` | `127.0.0.1:8085` | listen address |

`make clean` removes the databases, catalogs and thumbnails; the media, the
notebook of the auto tagging page and the models stay.

## Media search: what to click

- **The grid** is the catalog: every file the scan found, a `video` badge on
  clips and a red `failed` badge (hover for the error) on files that could not
  be indexed; those are retried on every rescan.
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
  which model, how many dimensions, which provider, the floor and what the
  last scan did.

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

## Auto tagging: what to click

`http://127.0.0.1:8085/autotag/` is a notebook of notes and tags, and shows
both ways `tagging` tags a text:

- **Existing tags** (`tagging.Tagger`, EmbeddingGemma 300M, ~330 MB, loaded at
  start). Every note shows, under *auto*, the notebook's tags the Tagger picks
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
model only costs embedding the notes again.

## How it is built

```
zarf/examples/
├── main.go           flags, wiring (onnx + ffmpeg + sqlitevec + search.MediaIndex), the home page
├── tagging.go        wiring of the auto tagging page (yzma Embedder, lazy Completer, sqlitecache)
├── mediasearch/      Library: scan, catalog, search, thumbnails, storyboards; Handler: the API and the page
│   └── ui/           index.html, app.js, style.css, embedded in the binary
├── autotag/          Notebook: notes, tags, hand tags, the Tagger's Catalog; Handler: the API and the page
│   └── ui/
├── prepare.sh        the sample data
└── data/             git-ignored: media/, index-<model>.sqlite, index-<model>.catalog.json, thumbs/,
                      notebook.json, autotag-cache.sqlite
```

`autotag` takes a `tagging.EmbeddingCache` and builds its own Tagger and
`tagging.Generator`, so its tests run on a `tagging.MemCache`,
`langaitest.FakeEmbedder` and a canned chat model. Its
API, under `/autotag/`: `GET api/info`, `GET api/notebook` (every tag and note
with its automatic tags), `POST api/tags` (`{"name"}`; an existing name returns
that tag), `PUT|DELETE api/tags/<id>`, `POST api/notes` (`{"title","text"}`),
`DELETE api/notes/<id>`, `PUT|DELETE api/notes/<id>/tags/<tag>` (hand tags),
`GET api/notes/<id>/recommend?lang=<BCP 47>&tree=1`, `POST api/tags` takes
`"parents": true` to create a path's missing levels, `POST api/samples?hand=<n>` (n hand tags
per tag, 0 by default).

`mediasearch` imports only the library's ports and `search`, so its tests run
on `langaitest.FakeImageEmbedder`, a `vector.MemStore` and a fake sampler; the
real model and ffmpeg are wired in `main.go` alone. A new example is a new
package with its own `Handler` and page, mounted in `main.go` next to
`/mediasearch/`.

The API, under `/mediasearch/`: `GET api/info`, `GET api/items`,
`GET api/similar/<id>?limit=&embed=` (by default the item's stored vectors
are the query and no model runs; `embed=1` embeds the file again, the
page's **re-embed on click** box), `POST api/search` (multipart `file`),
`POST api/rescan` (409 while a scan runs), `GET api/frames/<id>`,
`GET thumb/<id>`, `GET storyboard/<id>`, `GET file/<id>`. Ids are paths
relative to the media directory and only catalog keys are served.

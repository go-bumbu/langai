# langai examples

A single-user web service to exercise langai's features by hand, one page per
feature. The first is **media search**: images and videos that look like an
image or a video, with the numbers behind every hit.

## Run it

```
cd examples
make prepare   # sample photos from picsum.photos and two slideshow clips into data/media
make run       # index data/media, serve http://127.0.0.1:8085/mediasearch/
```

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
| `MODEL` | `google/siglip2-base-patch16-224` | or `facebook/dinov2-base`; each model gets a database of its own |
| `PROVIDER` | `cpu` | `cuda` or `coreml` to run the model on a GPU |
| `MIN_SCORE` | `0` | hits below it are dropped; 0 shows everything, which is what a debug view wants |
| `LIMIT` | `12` | hits per search |
| `ADDR` | `127.0.0.1:8085` | listen address |

`make clean` removes the databases, catalogs and thumbnails; the media and the
models stay.

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

## How it is built

```
examples/
├── main.go           flags, wiring (onnx + ffmpeg + sqlitevec + search.MediaIndex), the home page
├── mediasearch/      Library: scan, catalog, search, thumbnails, storyboards; Handler: the API and the page
│   └── ui/           index.html, app.js, style.css, embedded in the binary
├── prepare.sh        the sample data
└── data/             git-ignored: media/, index-<model>.sqlite, index-<model>.catalog.json, thumbs/
```

`mediasearch` imports only the library's ports and `search`, so its tests run
on `langaitest.FakeImageEmbedder`, a `vector.MemStore` and a fake sampler; the
real model and ffmpeg are wired in `main.go` alone. A new example is a new
package with its own `Handler` and page, mounted in `main.go` next to
`/mediasearch/`.

The API, under `/mediasearch/`: `GET api/info`, `GET api/items`,
`GET api/similar/<id>?limit=`, `POST api/search` (multipart `file`),
`POST api/rescan` (409 while a scan runs), `GET api/frames/<id>`,
`GET thumb/<id>`, `GET storyboard/<id>`, `GET file/<id>`. Ids are paths
relative to the media directory and only catalog keys are served.

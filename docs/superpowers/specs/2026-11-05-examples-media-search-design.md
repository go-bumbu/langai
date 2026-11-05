# Example service: media similarity search

Date: 2026-11-05. Status: approved by default (the author asked for an unattended
build and left the choices to the builder; every choice below is the builder's
unless marked *given*).

## Intent

*Given:* a single-user web service in a new `examples/` folder that lets the
author exercise langai's features by hand, starting with image-to-image, image-to-
video and video-to-image similarity search. `make prepare` fetches sample images
(and the author can drop in more images and videos); `make run` scans and indexes
them and serves a page that shows a thumbnail per media file; clicking one shows
the items that look like it, with debug information such as the proximity score.
More examples (automatic tagging, ...) will be added later, so the service is
built to grow a page per feature. The server takes inspiration from codex
(the author's application that consumes langai) but is simplified as far as it
goes, since it is only an example.

## Goals

- One command to get sample data, one to run; nothing to install beyond Go, ffmpeg
  and curl. The model and the ONNX Runtime download on first run, as in the tests.
- Every feature of `search.MediaIndex` reachable from the browser: image query,
  video query (frames, `At`, `QueryAt`), the no-op re-index of unchanged files, a
  query that is not in the corpus (an uploaded file).
- Debug information next to every result: score, which frame matched and when,
  how long the search took, the model and its settings, what the index saw of a
  video (its sampled frames).
- Room for the next examples: a feature is one Go package with its own page,
  mounted by a tiny `main`.

## Non-goals

- Users, auth, multi-tenancy, configuration files: it is one process, one
  directory, bound to localhost.
- A frontend toolchain: plain HTML, CSS and JavaScript embedded in the binary.
- Background jobs, watch mode: a scan runs at start and on a button.
- Thumbnails and storyboards of production quality (EXIF orientation, HEIC,
  animated WebP...). The library's own limits apply.

## Anti-goals

- New dependencies in `go.mod`: the example uses the stdlib, the module's own
  packages and `go-sqlite3`, which the module already has.
- Any change to the library's API to suit the example. If something is missing it
  goes to TODO.md.
- A second `go.mod`: the example is a package of the module so `make verify`
  builds, lints and tests it like everything else.

## Architecture

```
examples/
├── Makefile              prepare, run, clean
├── README.md             how to run it, what to click, what the numbers mean
├── prepare.sh            sample images from picsum.photos, sample videos via ffmpeg
├── main.go               package main: flags, wiring, the home page, one mount per feature
├── mediasearch/          package mediasearch: the Library (scan, index, catalog,
│   │                     thumbnails, storyboards) and its HTTP handler
│   └── ui/               index.html, app.js, style.css (go:embed)
└── data/                 git-ignored: media/, index-<model>.sqlite, thumbs/
```

Dependencies point one way: `main` imports `mediasearch`, the providers and the
stores; `mediasearch` imports `search`, `media` and `langai` only (ports and the
capability), so its tests run with `langaitest.FakeImageEmbedder`, a
`vector.MemStore` and a `media.SamplerFunc`, without a model or ffmpeg.

### `main`

Flags, with defaults for `make run` from `examples/`:

| Flag | Default | Meaning |
|---|---|---|
| `-addr` | `127.0.0.1:8085` | listen address |
| `-media` | `data/media` | directory scanned for images and videos (recursive) |
| `-data` | `data` | index database and thumbnail cache |
| `-models` | `$HOME/.cache/semsearch-onnx` | ONNX models and runtime, shared with `make test-images` so nothing downloads twice |
| `-model` | `onnx.DefaultModel` | `google/siglip2-base-patch16-224` or `facebook/dinov2-base` |
| `-provider` | `cpu` | `cpu`, `cuda` or `coreml` |
| `-limit` | `12` | hits per search |
| `-min-score` | `0` | `WithMediaMinScore`; 0 shows everything, which is what a debug view wants |

Wiring: `onnx.New` (ModelsDir `<models>/models`, LibDir
`<models>/onnxruntime/<onnx.LibVersion>`, managed), `ffmpeg.New(ffmpeg.Config{})`
(the library defaults), `sqlitevec.New` over `<data>/index-<model slug>.sqlite`
(one database per model: the two pinned models are both 768-d, but a database
per model keeps the spaces apart whatever model comes next),
`search.NewMediaIndex(emb, store, "examples", "local", "media", WithSampler,
WithMediaMinScore)`. Then `mediasearch.New`, one `Scan` before listening (logged
per file), the home page at `/` listing the features, the feature at
`/mediasearch/`. Logs go to stderr through `slog`, the onnx logger included, so
the first-run downloads are visible.

### `mediasearch.Library`

```go
type Config struct {
    Dir      string              // the media directory
    Index    *search.MediaIndex
    Sampler  media.Sampler       // the same one the index uses: storyboards show what it saw
    CacheDir string              // thumbnails and storyboards
    Limit    int
    Info     Info                // model name, dimensions, provider, min score: shown as is
}
type Item struct {
    ID      string // path relative to Dir, slash-separated
    Name    string
    Kind    string // "image" | "video"
    Size    int64
    ModTime time.Time
    Indexed time.Duration // what IndexImage/IndexVideo took; 0 when it was a no-op
    Skipped bool          // the no-op: bytes unchanged since the last run
    Error   string        // indexing failed (undecodable file, ffmpeg error); the item stays listed
}
type Hit struct {
    Item    Item
    Score   float64
    At      time.Duration // the matching frame of a video hit
    QueryAt time.Duration // the matching frame of a video query
    Self    bool          // the hit is the query item itself: kept and marked, not dropped
}
type Result struct {
    Query   *Item         // nil for an uploaded file
    Frames  int           // query frames embedded (1 for an image)
    Elapsed time.Duration
    Hits    []Hit
}
```

- `Scan(ctx)`: walk `Dir`; a file is an image or a video by extension (images:
  jpg, jpeg, png, gif, webp, bmp, tif, tiff; videos: mp4, m4v, mov, mkv, webm,
  avi); everything else is ignored. Each file is indexed in turn, images with
  `IndexImage`, videos with `IndexVideo`, and the time is recorded. The index
  hides whether a call was a no-op, so the Library keeps its own SHA-256 per
  file in `<CacheDir>/catalog.json`: a file whose hash equals the saved one is
  marked `Skipped` (the index call is still made, and is the cheap no-op). Files
  that disappeared since the last scan are `Delete`d. Sequential on purpose: the
  embedder batches and ffmpeg is heavy; the log says where it is. Returns stats
  (indexed, skipped, failed, removed, elapsed) and keeps the catalog in memory.
- `Items()`: the catalog, sorted by id.
- `Similar(ctx, id, limit)`: for an image, `SearchImage` over the file; for a
  video, `Sampler.Sample` then `SearchFrames`, so `Frames` is known. Every hit is
  joined with its `Item`; the hit whose id is the query's is marked `Self`.
- `Search(ctx, name, r, limit)`: an uploaded file; by extension an image
  (`SearchImage` over the bytes) or a video (written to a temp file, sampled,
  `SearchFrames`).
- `Thumb(ctx, id)`: a JPEG, longest side 320, cached in `CacheDir` under the
  SHA-256 of id, size and mtime. An image is decoded (the formats `search`
  registers) and scaled with `golang.org/x/image/draw` (already a dependency);
  a video's thumbnail is its first sampled frame, and that same `Sample` run
  also writes the storyboard, so a video is sampled once per change.
- `Storyboard(ctx, id)`: the sampled frames of a video as one JPEG strip of equal
  cells (height 120) plus `{CellW, CellH, Times []int}` in milliseconds, cached
  like the thumbnail. The page lays the times over the cells.

### HTTP

`mediasearch.Handler(lib)` returns an `http.Handler` for a mount point:

| Route | Returns |
|---|---|
| `GET /` and the static files | the page |
| `GET /api/info` | model, dimensions, provider, min score, limit, media dir, item counts, last scan stats |
| `GET /api/items` | `[]Item` |
| `GET /api/similar/{id...}?limit=` | `Result` |
| `POST /api/search` (multipart `file`) | `Result` |
| `POST /api/rescan` | scan stats |
| `GET /api/frames/{id...}` | the storyboard's `{CellW, CellH, Times}` |
| `GET /thumb/{id...}` | JPEG |
| `GET /storyboard/{id...}` | JPEG |
| `GET /file/{id...}` | the original (`http.ServeFile`, so `<video>` can seek) |

Ids are catalog keys: an unknown id is 404, so nothing outside the media
directory is ever served. Errors are `{"error": "..."}` with a 4xx/5xx. A scan
and a search may run at once; the catalog is behind a mutex.

### The page

One page, two panes. Left: a grid of thumbnails, a badge on videos, the name
under each, a red badge on items whose indexing failed (hover shows the error).
Top: the info bar (model, dims, provider, min score, N images, M videos, scan
time), a "search with a file" input and a "rescan" button. Right, after a click:
the item large (`<img>` or `<video controls>`), its facts (path, kind, size,
indexed in, skipped), for a video its storyboard with the frame times, and the
hits: thumbnail, name, score with a bar, `At` for video hits (clicking it seeks
the preview when the hit is the open item, otherwise opens the hit), `QueryAt`
for video queries, the self hit greyed at the top, and a footer line with
"N frames embedded, K hits in X ms". Uploading a file shows the same result
pane with the upload's preview.

### `make prepare`

`prepare.sh <dir>` downloads about twenty photos from picsum.photos by stable
id (`https://picsum.photos/id/<id>/800/600.jpg`, Unsplash licence) into
`<dir>/photos/`, plus for a few of them a 400×300 copy and a greyscale copy, so
near-duplicates and their scores are visible. It then renders two slideshow
videos into `<dir>/clips/` with ffmpeg (four photos each, two seconds per
photo, 640 px, the native mpeg4 encoder so no libx264 is needed), so an image
query finds the clip and the second at which it appears, and the clip finds
its photos. Existing files are kept, so the author's own files survive a
re-run; `make clean` removes only the index and the cache.

## Constraints

- `make verify` stays green: `examples/mediasearch` is tested to the 70 %
  coverage gate with fakes; `examples` (package main) has no tests and the gate
  tolerates a package without tests.
- `make lint` passes with the repository's golangci-lint settings (gosec included:
  file paths come from the catalog, the server binds to localhost by default).
- `examples/data/` is git-ignored at the root.
- The root README gets an "Examples" section and the layout tree stops calling
  `examples/` planned.

## Builder's calls after the review

- The catalog file belongs to the index: `main` names it after the model
  (`<data>/index-<model slug>.catalog.json`, next to the database) and
  `make clean` removes the databases, the catalogs and the thumbnail cache
  together, so a `Skipped` mark can never outlive the vectors it refers to. The
  hash of a file that failed to index is not saved, so it is retried on every
  scan and shown in the grid with an error badge meanwhile.
- The page shows `At`/`QueryAt` by the item's kind, never by the value: an
  image has no time and a video's first frame is at 0.
- Storyboards and thumbnails are cached by id, size and mtime and made by the
  same sampler configuration the index uses; the README says that a storyboard
  is the sampler run on the file as it is now.
- An uploaded copy of a corpus item is an ordinary hit scoring near 1; nothing
  is marked `Self` for an upload.
- Ids go in the path as `{id...}` wildcards (Go 1.22 mux); the page encodes
  each segment with `encodeURIComponent` and the tests use a name with spaces.
- A rescan while a scan runs answers 409.
- Uploads are capped at 256 MB (`http.MaxBytesReader`); a video upload's temp
  file is removed when the search returns.

## Left to the builder

- Exact styling of the page; the JavaScript structure.
- The picsum ids and which get variants.
- Thumbnail and storyboard sizes, JPEG quality.
- How progress is logged during the scan.

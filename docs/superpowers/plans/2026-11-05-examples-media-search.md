# Example service: media similarity search — implementation plan

> **For agentic workers:** built inline, task by task, test first. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `examples/`: a localhost web service that scans a media directory into a `search.MediaIndex`, shows a thumbnail grid and, per item or uploaded file, the items that look alike with their scores and frame times; `make prepare` fetches sample data, `make run` serves it.

**Spec:** `docs/superpowers/specs/2026-11-05-examples-media-search-design.md`.

**Tech Stack:** Go 1.26, stdlib `net/http` (1.22 patterns), `go:embed`, `golang.org/x/image/draw`, `go-sqlite3` (already in go.mod), ffmpeg and curl on PATH for `prepare.sh`.

## Global constraints

- One-line commit messages; `git add` as a separate step; never push.
- `make lint` and `make test` green after every task; `make coverage` ≥ 70 % for `examples/mediasearch` at the end.
- No new entries in `go.mod`.

## Tasks

### Task 1: `examples/mediasearch` — catalog and scan

- [ ] `library.go`: `Config`, `Item`, `Library`, `New`, `Scan`, `Items`, `Stats`; kind by extension; relative slash ids; SHA-256 per file; `catalog.json` in `CacheDir`; `Delete` of vanished files; sequential indexing with timings and an `Error` per failed item.
- [ ] `library_test.go` with `FakeImageEmbedder`, `MemStore`, a `SamplerFunc` that returns two frames: first scan indexes images and videos; second scan marks them skipped; a changed file is re-indexed; a removed file is deleted; an undecodable file keeps an `Error`; unsupported extensions are ignored.
- [ ] Commit.

### Task 2: search

- [ ] `search.go`: `Hit`, `Result`, `Similar` (image → `SearchImage`; video → `Sample` + `SearchFrames`), `Search` (uploaded bytes; video via a temp file), `Self` marking, joining hits with items.
- [ ] Tests: the query itself comes first and is `Self`; an image finds the video that contains it with `At`; a video finds its images with `QueryAt`; an uploaded copy of an image finds it; unknown id and unsupported upload are errors.
- [ ] Commit.

### Task 3: thumbnails and storyboards

- [ ] `thumbs.go`: `Thumb`, `Storyboard`, `Frames` info; the cache keyed by id, size and mtime; `fit` scaling; the strip composer.
- [ ] Tests: an image thumbnail is at most 320 px and cached (second call reads the file); a video's thumbnail is its first sampled frame and its storyboard has one cell per frame with the frames' times; an unknown id is an error.
- [ ] Commit.

### Task 4: HTTP handler and page

- [ ] `handler.go`: the routes of the spec; JSON encoding; multipart upload; `http.ServeFile` for originals; embedded `ui/`.
- [ ] `ui/index.html`, `ui/app.js`, `ui/style.css`: grid, info bar, upload, rescan, detail pane, storyboard, hits with score bars.
- [ ] `handler_test.go` with `httptest`: info, items, similar, search upload, rescan, thumb, storyboard, file, 404s.
- [ ] Commit.

### Task 5: `main`, Makefile, prepare, docs

- [ ] `examples/main.go`: flags, onnx + ffmpeg + sqlitevec wiring, scan, home page, mount.
- [ ] `examples/Makefile`: `prepare`, `run`, `clean`, `help`; `examples/prepare.sh`.
- [ ] `examples/README.md`; root README "Examples" section and layout line; `.gitignore` entry for `examples/data/`.
- [ ] Run it for real against the sample data and the cached model; fix what the browser shows.
- [ ] `make verify`; commit.

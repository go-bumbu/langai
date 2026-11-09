// Package textsearch is the example service's semantic text search: it scans
// a directory of Markdown and text files into a search.Index with one
// IndexBatch call, reconciles the index with the directory through Docs and
// Delete, keeps a catalog of what it found, and answers a free-text query or
// "more like this document" with the chunk that matched and its score. It
// also shows a document cut into the chunks that were embedded. Handler
// serves it over HTTP. It knows the library's ports only, so its tests run
// on langaitest's fakes.
package textsearch

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
)

// States of a document after a scan.
const (
	// StateIndexed means the scan chunked, embedded and stored it.
	StateIndexed = "indexed"
	// StateUnchanged means it was stored already with the same text, chunker
	// and model: the scan did nothing for it.
	StateUnchanged = "unchanged"
	// StateFailed means it was refused for a reason of its own (Doc.Error).
	StateFailed = "failed"
	// StatePending means the index does not hold it as it is: the scan
	// stopped before it, or the index was reset since.
	StatePending = "not indexed"
)

// maxDocBytes is the longest file the scan reads; a longer one is listed as
// failed.
const maxDocBytes = 4 << 20

// Errors the handler maps to statuses.
var (
	// ErrScanning is returned by Scan and Reset while a scan runs.
	ErrScanning = errors.New("textsearch: a scan is already running")
	// ErrNotFound is returned for an id that is not in the catalog.
	ErrNotFound = errors.New("textsearch: no such document")
)

// Config wires a Library.
type Config struct {
	// Dir is the directory scanned for .md, .markdown and .txt files,
	// recursively.
	Dir string
	// Index is where the documents go and what answers the searches.
	Index *search.Index
	// Chunker is the chunker Index cuts documents with, so that a document's
	// view shows the chunks that were embedded; nil means the Index's
	// default, a search.MarkdownChunker.
	Chunker search.Chunker
	// Limit is the default number of hits per search; <= 0 means 10.
	Limit int
	// Info describes the setup; it is shown on the page as is.
	Info Info
	// ModelState says where the embedding model's lazy loader stands ("not
	// started", "loaded"...); nil leaves it out.
	ModelState func() string
	// Stored returns the model the store recorded for the index's collection
	// (vector.Store.Model), to name it when it is not the one Index writes
	// with; nil leaves the name out.
	Stored func(ctx context.Context) (string, error)
	// Examples are queries the page offers to try.
	Examples []string
	// Logger receives the scan's progress; nil discards it.
	Logger *slog.Logger
}

// Info is what the page shows about the setup.
type Info struct {
	// Model is the embedding model's identity, the string the store records
	// with its vectors.
	Model      string `json:"model"`
	Dimensions int    `json:"dimensions"`
	// Chunker is the chunker's fingerprint (search.Fingerprinter), which the
	// index hashes with every document; New fills it in.
	Chunker  string  `json:"chunker"`
	MinScore float64 `json:"minScore"`
	// Store names the vector store.
	Store string `json:"store"`
}

// Doc is one file of the documents directory.
type Doc struct {
	// ID is the path relative to the directory, slash-separated.
	ID   string `json:"id"`
	Name string `json:"name"`
	// Title is the first heading or line of the text.
	Title   string    `json:"title"`
	Bytes   int64     `json:"bytes"`
	ModTime time.Time `json:"modTime"`
	// Chunks is how many chunks the chunker cut the text into at the last
	// scan: the number of vectors the document has.
	Chunks int `json:"chunks"`
	// State is what the last scan did with it, see the State constants.
	State string `json:"state"`
	// Error is why it failed.
	Error string `json:"error,omitempty"`
}

// ScanStats sums up one scan: the BatchResult of its IndexBatch call, and
// what the reconciliation removed.
type ScanStats struct {
	// Indexed, Unchanged and Failed count the documents by what IndexBatch
	// did with them; Failed includes the files the scan could not read.
	Indexed   int `json:"indexed"`
	Unchanged int `json:"unchanged"`
	Failed    int `json:"failed"`
	// Pending counts the documents the batch did not reach, because an
	// embedder or store error stopped it (Error).
	Pending int `json:"pending"`
	// Removed counts the documents deleted from the index because their
	// files are gone.
	Removed int `json:"removed"`
	// Failures gives the reason of every failed document, by id.
	Failures map[string]string `json:"failures,omitempty"`
	// Error is the embedder or store error that stopped the batch, and
	// Mismatch says it was vector.ErrModelMismatch: the collection holds
	// another model's vectors and has to be reset.
	Error    string `json:"error,omitempty"`
	Mismatch bool   `json:"mismatch,omitempty"`
	// IndexMs is what the IndexBatch call took, ElapsedMs the whole scan.
	IndexMs   int64     `json:"indexMs"`
	ElapsedMs int64     `json:"elapsedMs"`
	At        time.Time `json:"at"`
}

// Library is the catalog over the index.
type Library struct {
	cfg     Config
	chunker search.Chunker
	logger  *slog.Logger

	mu   sync.Mutex
	docs map[string]*Doc
	// heads holds each document's headings, to tell a chunk's heading path
	// from its text.
	heads    map[string]map[string]bool
	last     ScanStats
	scanning atomic.Bool
}

// New returns a Library; nothing is read until Scan.
func New(cfg Config) (*Library, error) {
	if cfg.Dir == "" {
		return nil, errors.New("textsearch: empty documents directory")
	}
	if cfg.Index == nil {
		return nil, errors.New("textsearch: nil index")
	}
	if cfg.Limit <= 0 {
		cfg.Limit = 10
	}
	l := &Library{cfg: cfg, chunker: cfg.Chunker, logger: cfg.Logger, docs: map[string]*Doc{}, heads: map[string]map[string]bool{}}
	if l.chunker == nil {
		l.chunker = &search.MarkdownChunker{}
	}
	if l.logger == nil {
		l.logger = slog.New(slog.DiscardHandler)
	}
	// what the index hashes with every document: a chunker without a
	// fingerprint is described by its Go type
	if f, ok := l.chunker.(search.Fingerprinter); ok {
		l.cfg.Info.Chunker = f.Fingerprint()
	} else {
		l.cfg.Info.Chunker = fmt.Sprintf("%T", l.chunker)
	}
	return l, nil
}

// Info returns the setup the Library was built with.
func (l *Library) Info() Info { return l.cfg.Info }

// Limit returns the default hits per search.
func (l *Library) Limit() int { return l.cfg.Limit }

// Dir returns the documents directory.
func (l *Library) Dir() string { return l.cfg.Dir }

// Examples returns the queries the page offers to try.
func (l *Library) Examples() []string { return l.cfg.Examples }

// Scanning reports whether a scan runs now.
func (l *Library) Scanning() bool { return l.scanning.Load() }

// ModelState says where the embedding model's loader stands; "" when the
// Library was not told.
func (l *Library) ModelState() string {
	if l.cfg.ModelState == nil {
		return ""
	}
	return l.cfg.ModelState()
}

// StoredModel returns the model the store recorded for the collection; ""
// when the collection is empty or the Library was not given a way to ask.
func (l *Library) StoredModel(ctx context.Context) string {
	if l.cfg.Stored == nil {
		return ""
	}
	model, err := l.cfg.Stored(ctx)
	if err != nil {
		l.logger.Warn("stored model not read", "err", err)
		return ""
	}
	return model
}

// LastScan returns the stats of the last scan; zero before the first.
func (l *Library) LastScan() ScanStats {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.last
}

// Docs returns the catalog sorted by id.
func (l *Library) Docs() []Doc {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]Doc, 0, len(l.docs))
	for _, d := range l.docs {
		out = append(out, *d)
	}
	slices.SortFunc(out, func(a, b Doc) int { return strings.Compare(a.ID, b.ID) })
	return out
}

// Doc returns one document by id.
func (l *Library) Doc(id string) (Doc, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	d, ok := l.docs[id]
	if !ok {
		return Doc{}, false
	}
	return *d, true
}

// isText reports whether a file is a document by its extension.
func isText(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".md", ".markdown", ".txt":
		return true
	}
	return false
}

// found is one file of a walk: its catalog entry and, unless reading it
// failed, its text.
type found struct {
	doc  Doc
	text string
}

// walk lists and reads the documents under dir, skipping dot-directories. A
// file that cannot be read, or is too long, comes back failed with the
// reason, and without text. A directory that does not exist holds nothing.
func (l *Library) walk() ([]found, error) {
	var out []found
	err := filepath.WalkDir(l.cfg.Dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if path == l.cfg.Dir && errors.Is(err, fs.ErrNotExist) {
				return filepath.SkipAll
			}
			return err
		}
		if d.IsDir() {
			if path != l.cfg.Dir && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if !isText(d.Name()) {
			return nil
		}
		rel, err := filepath.Rel(l.cfg.Dir, path)
		if err != nil {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		f := found{doc: Doc{ID: filepath.ToSlash(rel), Name: d.Name(), Title: d.Name(), Bytes: info.Size(), ModTime: info.ModTime(), State: StatePending}}
		switch {
		case info.Size() > maxDocBytes:
			f.doc.State, f.doc.Error = StateFailed, fmt.Sprintf("longer than the %d MiB the scan reads", maxDocBytes>>20)
		default:
			data, err := os.ReadFile(path) //nolint:gosec // a file the walk found under the documents directory, only read
			if err != nil {
				f.doc.State, f.doc.Error = StateFailed, err.Error()
				break
			}
			f.text = clean(string(data))
			f.doc.Title = title(f.text, d.Name())
			f.doc.Chunks = len(l.chunker.Chunk(f.text))
		}
		out = append(out, f)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("textsearch: walk %s: %w", l.cfg.Dir, err)
	}
	return out, nil
}

// clean makes a text what the index stores of it: invalid UTF-8 becomes
// U+FFFD and NUL bytes are dropped, so the chunks shown are the chunks
// embedded.
func clean(text string) string {
	return strings.ReplaceAll(strings.ToValidUTF8(text, "�"), "\x00", "")
}

// title is the text of a document's first heading, or its first line that
// is not blank, cut to a line's length; fallback when the text is blank.
func title(text, fallback string) string {
	for line := range strings.SplitSeq(text, "\n") {
		line = strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(line), "#"))
		if line == "" {
			continue
		}
		if r := []rune(line); len(r) > 100 {
			return string(r[:100]) + "…"
		}
		return line
	}
	return fallback
}

// headings returns the texts of a document's ATX headings ("## Section"),
// as a set.
func headings(text string) map[string]bool {
	set := map[string]bool{}
	for line := range strings.SplitSeq(text, "\n") {
		line = strings.TrimSpace(line)
		if hashes := len(line) - len(strings.TrimLeft(line, "#")); hashes == 0 || hashes > 6 || len(line) == hashes || line[hashes] != ' ' {
			continue
		}
		set[strings.TrimSpace(strings.Trim(line, "# "))] = true
	}
	return set
}

// Scan reads the documents directory and brings the index in step with it,
// in two calls: IndexBatch with every document (it embeds only the changed
// ones, and its BatchResult says which those were), then, for the files
// that are gone, a Delete of the ids Docs lists and the directory no longer
// holds. An embedder or store error stops the batch; what was indexed
// before it stays, the rest is pending, and the stats carry the error. Only
// one scan runs at a time: a second one gets ErrScanning.
func (l *Library) Scan(ctx context.Context) (ScanStats, error) {
	if !l.scanning.CompareAndSwap(false, true) {
		return ScanStats{}, ErrScanning
	}
	defer l.scanning.Store(false)
	start := time.Now()
	files, err := l.walk()
	if err != nil {
		return ScanStats{}, err
	}
	stats := ScanStats{At: start, Failures: map[string]string{}}
	batch := make([]search.Doc, 0, len(files))
	readable := make(map[string]bool, len(files))
	docs := make(map[string]*Doc, len(files))
	heads := make(map[string]map[string]bool, len(files))
	for i := range files {
		d := &files[i].doc
		docs[d.ID] = d
		if d.State == StateFailed {
			stats.Failures[d.ID] = d.Error
			continue
		}
		readable[d.ID] = true
		heads[d.ID] = headings(files[i].text)
		batch = append(batch, search.Doc{ID: d.ID, Text: files[i].text})
	}

	indexStart := time.Now()
	res, err := l.cfg.Index.IndexBatch(ctx, batch)
	stats.IndexMs = time.Since(indexStart).Milliseconds()
	for _, id := range res.Indexed {
		docs[id].State = StateIndexed
	}
	for _, id := range res.Unchanged {
		docs[id].State = StateUnchanged
	}
	for id, reason := range res.Failed {
		docs[id].State, docs[id].Error = StateFailed, reason.Error()
		stats.Failures[id] = reason.Error()
	}
	stats.Indexed, stats.Unchanged, stats.Failed = len(res.Indexed), len(res.Unchanged), len(stats.Failures)
	stats.Pending = len(files) - stats.Indexed - stats.Unchanged - stats.Failed
	if err != nil {
		stats.Error, stats.Mismatch = err.Error(), errors.Is(err, vector.ErrModelMismatch)
		l.logger.Warn("indexing stopped", "err", err, "indexed", stats.Indexed, "pending", stats.Pending)
	}
	if ctx.Err() == nil {
		stats.Removed = l.removeGone(ctx, readable)
	}
	stats.ElapsedMs = time.Since(start).Milliseconds()

	l.mu.Lock()
	l.docs, l.heads, l.last = docs, heads, stats
	l.mu.Unlock()
	l.logger.Info("documents scanned", "indexed", stats.Indexed, "unchanged", stats.Unchanged, "failed", stats.Failed,
		"pending", stats.Pending, "removed", stats.Removed, "elapsed", time.Since(start).Round(time.Millisecond))
	return stats, ctx.Err()
}

// removeGone deletes from the index, in one Delete, every document Docs
// lists that is not among the files the scan read (keep): the ones whose
// files are gone or can no longer be read. It returns how many.
func (l *Library) removeGone(ctx context.Context, keep map[string]bool) int {
	var gone []string
	for info, err := range l.cfg.Index.Docs(ctx) {
		if err != nil {
			l.logger.Warn("index not listed", "err", err)
			return 0
		}
		if !keep[info.ID] {
			gone = append(gone, info.ID)
		}
	}
	if len(gone) == 0 {
		return 0
	}
	if err := l.cfg.Index.Delete(ctx, gone...); err != nil {
		l.logger.Warn("delete failed", "ids", len(gone), "err", err)
		return 0
	}
	l.logger.Info("removed", "ids", gone)
	return len(gone)
}

// Reset empties the index's collection (search.Index.Reset), which is the
// way to another embedding model: the store refuses a model's vectors while
// the collection holds another's. Every document is pending afterwards and
// the next Scan embeds them all. It fails with ErrScanning while a scan
// runs.
func (l *Library) Reset(ctx context.Context) error {
	if !l.scanning.CompareAndSwap(false, true) {
		return ErrScanning
	}
	defer l.scanning.Store(false)
	if err := l.cfg.Index.Reset(ctx); err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, d := range l.docs {
		if d.State != StateFailed {
			d.State = StatePending
		}
	}
	l.last = ScanStats{}
	l.logger.Info("index reset")
	return nil
}

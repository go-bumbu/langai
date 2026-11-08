// Package mediasearch is the example service's media similarity search: it
// scans a directory of images and videos into a search.MediaIndex, keeps a
// catalog of what it found, answers "what looks like this" for an item or an
// uploaded file, and makes the thumbnails and storyboards the page shows.
// Handler serves it over HTTP. It knows the library's ports only, so its
// tests run on langaitest's fakes.
package mediasearch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-bumbu/langai/media"
	"github.com/go-bumbu/langai/search"
)

// Kinds of items.
const (
	KindImage = "image"
	KindVideo = "video"
)

// ErrScanning is returned by Scan while another scan runs.
var ErrScanning = errors.New("mediasearch: a scan is already running")

// Config wires a Library.
type Config struct {
	// Dir is the directory scanned for images and videos, recursively.
	Dir string
	// Index is where the items go and what answers the searches.
	Index *search.MediaIndex
	// Sampler turns a video into frames; it should be the one Index uses,
	// so that a storyboard shows what the index saw.
	Sampler media.Sampler
	// CatalogFile keeps the content hash of every indexed file between
	// runs, so a scan can tell a no-op from a real embedding. It belongs
	// with the index database: remove the two together.
	CatalogFile string
	// CacheDir holds the thumbnails and storyboards.
	CacheDir string
	// Limit is the default number of hits per search; <= 0 means 12.
	Limit int
	// Info describes the setup; it is shown on the page as is.
	Info Info
	// Logger receives the scan's progress; nil discards it.
	Logger *slog.Logger
}

// Info is what the page shows about the setup.
type Info struct {
	Model      string  `json:"model"`
	Dimensions int     `json:"dimensions"`
	Provider   string  `json:"provider"` // what runs the model: an ONNX provider, or "cpu" or a GPU device for a GGUF model
	MinScore   float64 `json:"minScore"`
	// Mode says how videos are indexed: "frames" (sampled frames through an
	// image model) or "segments" (whole stretches through a video model).
	Mode string `json:"mode"`
	// SegmentMs is the segment length in milliseconds in the segments mode.
	SegmentMs int64 `json:"segmentMs"`
	// Text says whether text queries work (the model has a text side).
	Text bool `json:"text"`
}

// Item is one file of the media directory.
type Item struct {
	// ID is the path relative to the media directory, slash-separated.
	ID      string    `json:"id"`
	Name    string    `json:"name"`
	Kind    string    `json:"kind"`
	Size    int64     `json:"size"`
	ModTime time.Time `json:"modTime"`
	// IndexedMs is what the index call took in the last scan, hashing
	// included.
	IndexedMs int64 `json:"indexedMs"`
	// Skipped says the file's bytes were unchanged since the previous scan,
	// so the index call was its cheap no-op.
	Skipped bool `json:"skipped"`
	// Error is why indexing failed; the item stays listed and is retried on
	// the next scan.
	Error string `json:"error,omitempty"`
}

// ScanStats sums up one scan.
type ScanStats struct {
	Indexed int           `json:"indexed"`
	Skipped int           `json:"skipped"`
	Failed  int           `json:"failed"`
	Removed int           `json:"removed"`
	Elapsed time.Duration `json:"-"`
	// ElapsedMs is Elapsed for the page.
	ElapsedMs int64     `json:"elapsedMs"`
	At        time.Time `json:"at"`
}

// Library is the catalog over the index.
type Library struct {
	cfg    Config
	logger *slog.Logger

	mu       sync.Mutex
	items    map[string]*Item
	hashes   map[string]string // saved content hash per id, the catalog file
	last     ScanStats
	scanning atomic.Bool

	genMu sync.Mutex // one thumbnail or storyboard is made at a time
}

// New returns a Library and loads the catalog file when there is one.
func New(cfg Config) (*Library, error) {
	if cfg.Dir == "" {
		return nil, errors.New("mediasearch: empty media directory")
	}
	if cfg.Index == nil {
		return nil, errors.New("mediasearch: nil index")
	}
	if cfg.Limit <= 0 {
		cfg.Limit = 12
	}
	l := &Library{cfg: cfg, logger: cfg.Logger, items: map[string]*Item{}, hashes: map[string]string{}}
	if l.logger == nil {
		l.logger = slog.New(slog.DiscardHandler)
	}
	if cfg.CatalogFile != "" {
		data, err := os.ReadFile(cfg.CatalogFile)
		switch {
		case errors.Is(err, fs.ErrNotExist):
		case err != nil:
			return nil, fmt.Errorf("mediasearch: read catalog: %w", err)
		default:
			if err := json.Unmarshal(data, &l.hashes); err != nil {
				return nil, fmt.Errorf("mediasearch: parse catalog %s: %w", cfg.CatalogFile, err)
			}
		}
	}
	return l, nil
}

// Info returns the setup the Library was built with.
func (l *Library) Info() Info { return l.cfg.Info }

// Limit returns the default hits per search.
func (l *Library) Limit() int { return l.cfg.Limit }

// Dir returns the media directory.
func (l *Library) Dir() string { return l.cfg.Dir }

// LastScan returns the stats of the last scan; zero before the first.
func (l *Library) LastScan() ScanStats {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.last
}

// Items returns the catalog sorted by id.
func (l *Library) Items() []Item {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]Item, 0, len(l.items))
	for _, it := range l.items {
		out = append(out, *it)
	}
	slices.SortFunc(out, func(a, b Item) int { return strings.Compare(a.ID, b.ID) })
	return out
}

// Item returns one item by id.
func (l *Library) Item(id string) (Item, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	it, ok := l.items[id]
	if !ok {
		return Item{}, false
	}
	return *it, true
}

// Path returns the file behind an id, which must be a catalog key.
func (l *Library) Path(id string) (string, bool) {
	if _, ok := l.Item(id); !ok {
		return "", false
	}
	return filepath.Join(l.cfg.Dir, filepath.FromSlash(id)), true
}

// kindOf returns the kind of a file by its extension, "" for one that is
// neither an image the index decodes nor a video.
func kindOf(name string) string {
	switch strings.ToLower(strings.TrimPrefix(filepath.Ext(name), ".")) {
	case "jpg", "jpeg", "png", "gif", "webp", "bmp", "tif", "tiff":
		return KindImage
	case "mp4", "m4v", "mov", "mkv", "webm", "avi":
		return KindVideo
	}
	return ""
}

// walk lists the media files under dir, skipping dot-directories, as items
// without index information.
func walk(dir string) ([]Item, error) {
	var found []Item
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != dir && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		kind := kindOf(d.Name())
		if kind == "" {
			return nil
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		found = append(found, Item{ID: filepath.ToSlash(rel), Name: d.Name(), Kind: kind, Size: info.Size(), ModTime: info.ModTime()})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("mediasearch: walk %s: %w", dir, err)
	}
	return found, nil
}

// Scan walks the media directory and indexes every image and video in turn:
// a file whose bytes are unchanged since the previous scan is marked
// Skipped, one that fails keeps its error, and files that disappeared are
// deleted from the index. The catalog fills as the scan goes, so the page
// can show it meanwhile. Only one scan runs at a time: a second one gets
// ErrScanning.
func (l *Library) Scan(ctx context.Context) (ScanStats, error) {
	if !l.scanning.CompareAndSwap(false, true) {
		return ScanStats{}, ErrScanning
	}
	defer l.scanning.Store(false)
	start := time.Now()
	found, err := walk(l.cfg.Dir)
	if err != nil {
		return ScanStats{}, err
	}
	stats := ScanStats{At: start}
	seen := make(map[string]bool, len(found))
	for i := range found {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		it := found[i]
		seen[it.ID] = true
		l.indexOne(ctx, &it, &stats, len(found)-i-1)
		l.mu.Lock()
		l.items[it.ID] = &it
		l.mu.Unlock()
	}
	stats.Removed = l.removeGone(ctx, seen)
	stats.Elapsed = time.Since(start)
	stats.ElapsedMs = stats.Elapsed.Milliseconds()
	l.mu.Lock()
	l.last = stats
	l.mu.Unlock()
	if err := l.saveCatalog(); err != nil {
		return stats, err
	}
	l.logger.Info("scan done", "indexed", stats.Indexed, "skipped", stats.Skipped, "failed", stats.Failed, "removed", stats.Removed, "elapsed", stats.Elapsed)
	return stats, nil
}

// indexOne hashes and indexes one file and fills in the item's index fields;
// remaining is how many items of the scan still wait after this one, for the log.
func (l *Library) indexOne(ctx context.Context, it *Item, stats *ScanStats, remaining int) {
	path := filepath.Join(l.cfg.Dir, filepath.FromSlash(it.ID))
	start := time.Now()
	hash, err := fileHash(path)
	if err == nil {
		err = l.index(ctx, it, path)
	}
	it.IndexedMs = time.Since(start).Milliseconds()
	l.mu.Lock()
	defer l.mu.Unlock()
	if err != nil {
		it.Error = err.Error()
		delete(l.hashes, it.ID)
		stats.Failed++
		l.logger.Warn("index failed", "id", it.ID, "err", err, "remaining", remaining)
		return
	}
	it.Skipped = l.hashes[it.ID] == hash
	l.hashes[it.ID] = hash
	if it.Skipped {
		stats.Skipped++
		l.logger.Info("unchanged", "id", it.ID, "remaining", remaining)
		return
	}
	stats.Indexed++
	l.logger.Info("indexed", "id", it.ID, "kind", it.Kind, "took", time.Duration(it.IndexedMs)*time.Millisecond, "remaining", remaining)
}

func (l *Library) index(ctx context.Context, it *Item, path string) error {
	if it.Kind == KindVideo {
		return l.cfg.Index.IndexVideo(ctx, it.ID, path)
	}
	f, err := os.Open(path) //nolint:gosec // a file the walk found under the media directory, only read
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return l.cfg.Index.IndexImage(ctx, it.ID, f)
}

// removeGone deletes from the index and the catalog every item not in seen.
func (l *Library) removeGone(ctx context.Context, seen map[string]bool) int {
	l.mu.Lock()
	var gone []string
	for id := range l.items {
		if !seen[id] {
			gone = append(gone, id)
		}
	}
	for id := range l.hashes {
		if !seen[id] && l.items[id] == nil {
			gone = append(gone, id)
		}
	}
	l.mu.Unlock()
	removed := 0
	for _, id := range gone {
		if err := l.cfg.Index.Delete(ctx, id); err != nil {
			l.logger.Warn("delete failed", "id", id, "err", err)
			continue
		}
		l.mu.Lock()
		delete(l.items, id)
		delete(l.hashes, id)
		l.mu.Unlock()
		removed++
		l.logger.Info("removed", "id", id)
	}
	return removed
}

// saveCatalog writes the hashes to the catalog file, if there is one.
func (l *Library) saveCatalog() error {
	if l.cfg.CatalogFile == "" {
		return nil
	}
	l.mu.Lock()
	data, err := json.MarshalIndent(l.hashes, "", "  ")
	l.mu.Unlock()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(l.cfg.CatalogFile), 0o750); err != nil {
		return fmt.Errorf("mediasearch: save catalog: %w", err)
	}
	if err := os.WriteFile(l.cfg.CatalogFile, data, 0o600); err != nil {
		return fmt.Errorf("mediasearch: save catalog: %w", err)
	}
	return nil
}

// fileHash is the SHA-256 of a file, hex encoded.
func fileHash(path string) (string, error) {
	f, err := os.Open(path) //nolint:gosec // a file the walk found under the media directory, only hashed
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

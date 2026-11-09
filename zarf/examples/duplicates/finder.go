// Package duplicates is the example service's duplicate finder: it hashes
// the pictures of a directory with provider/phash, which runs no model, and
// shows both faces of that package. The raw Hash values group the pictures
// into exact and near-exact duplicates by their Hamming distance, with no
// store; a search.MediaIndex over the same hash, as an ImageEmbedder,
// answers "find the copies of this uploaded picture" as it would with a
// model's vectors, and the page shows that its cosine score is the Hamming
// distance read another way: 1 - 2·distance/bits. Handler serves it over
// HTTP.
package duplicates

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-bumbu/langai/provider/phash"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
)

// DefaultKind is the hash the page starts with: phash's own default.
const DefaultKind = "phash-64"

// The coordinates of the MediaIndexes: one collection per hash, since each
// hash is a vector space of its own.
const (
	indexName      = "examples"
	indexPartition = "local"
	collectionBase = "duplicates-"
)

// maxPixels is the largest picture the scan decodes, phash's own limit.
const maxPixels = 256 << 20

// Errors the handler maps to statuses.
var (
	// ErrScanning is returned by Scan while another scan runs.
	ErrScanning = errors.New("duplicates: a scan is already running")
	// ErrNotFound is returned for an id that is not in the catalog.
	ErrNotFound = errors.New("duplicates: no such picture")
	// ErrBadInput is returned for a hash kind or a threshold the Finder
	// does not have.
	ErrBadInput = errors.New("duplicates: bad input")
)

// Config wires a Finder.
type Config struct {
	// Dir is the directory scanned for pictures, recursively.
	Dir string
	// Limit is the default number of hits per search; <= 0 means 12.
	Limit int
	// Logger receives the scan's progress; nil discards it.
	Logger *slog.Logger
}

// Kind is one of the four hashes: an algorithm at a size.
type Kind struct {
	// Name is the hash's identity, phash's Model: "phash-64", "dhash-256"...
	Name      string `json:"name"`
	Algorithm string `json:"algorithm"`
	Bits      int    `json:"bits"`
	// Threshold is the Hamming distance the page starts with for this hash:
	// the bits a re-encoded, resized or brightened copy stays within.
	Threshold int `json:"threshold"`
}

// kind is a Kind with what computes it and, once a search asked for it, the
// MediaIndex over it.
type kind struct {
	Kind
	hasher *phash.Embedder

	// store keeps the vectors of the hash's MediaIndex: a vector.MemStore of
	// its own, since hashing takes milliseconds a picture and leaves nothing
	// worth keeping between runs, and a store holds vectors of one length.
	store *vector.MemStore

	// mu guards index and synced: one sync of a kind's index runs at a time.
	mu    sync.Mutex
	index *search.MediaIndex
	// synced holds, per picture in the index, the stamp it was indexed at.
	synced map[string]stamp
}

// stamp tells a file from an edited one without reading it.
type stamp struct {
	size int64
	mod  time.Time
}

// Picture is one image file of the directory.
type Picture struct {
	// ID is the path relative to the directory, slash-separated.
	ID      string    `json:"id"`
	Name    string    `json:"name"`
	Size    int64     `json:"size"`
	ModTime time.Time `json:"modTime"`
	Width   int       `json:"width"`
	Height  int       `json:"height"`
	// Error is why the picture could not be hashed.
	Error string `json:"error,omitempty"`

	// hashes holds the picture's hash of every kind, by the kind's name.
	hashes map[string]phash.Hash
}

// ScanStats sums up one scan.
type ScanStats struct {
	// Hashed counts the pictures decoded and hashed, Unchanged the ones
	// whose size and time are those of the previous scan, so their hashes
	// were kept.
	Hashed    int `json:"hashed"`
	Unchanged int `json:"unchanged"`
	Failed    int `json:"failed"`
	Removed   int `json:"removed"`
	// HashUs is the time spent decoding and hashing, in microseconds, all
	// four hashes of every hashed picture together: nearly all of it is the
	// decode.
	HashUs    int64     `json:"hashUs"`
	ElapsedMs int64     `json:"elapsedMs"`
	At        time.Time `json:"at"`
}

// Finder is the catalog of hashed pictures.
type Finder struct {
	cfg    Config
	logger *slog.Logger
	kinds  []*kind

	mu       sync.Mutex
	pictures map[string]*Picture
	last     ScanStats
	scanning atomic.Bool
}

// New returns a Finder; nothing is read until Scan.
func New(cfg Config) (*Finder, error) {
	if cfg.Dir == "" {
		return nil, errors.New("duplicates: empty directory")
	}
	if cfg.Limit <= 0 {
		cfg.Limit = 12
	}
	f := &Finder{cfg: cfg, logger: cfg.Logger, pictures: map[string]*Picture{}}
	if f.logger == nil {
		f.logger = slog.New(slog.DiscardHandler)
	}
	// plain copies stayed within 2 bits of 64 and 9 of 256 on phash's test
	// pictures; photos move a little more, so the page starts a bit wider
	for _, k := range []Kind{
		{Algorithm: string(phash.AlgorithmPHash), Bits: 64, Threshold: 4},
		{Algorithm: string(phash.AlgorithmDHash), Bits: 64, Threshold: 4},
		{Algorithm: string(phash.AlgorithmPHash), Bits: 256, Threshold: 16},
		{Algorithm: string(phash.AlgorithmDHash), Bits: 256, Threshold: 16},
	} {
		h, err := phash.New(phash.Config{Algorithm: phash.Algorithm(k.Algorithm), Bits: k.Bits})
		if err != nil {
			return nil, fmt.Errorf("duplicates: %w", err)
		}
		k.Name = h.Model()
		f.kinds = append(f.kinds, &kind{Kind: k, hasher: h, store: vector.NewMemStore(), synced: map[string]stamp{}})
	}
	return f, nil
}

// Dir returns the directory scanned.
func (f *Finder) Dir() string { return f.cfg.Dir }

// Limit returns the default hits per search.
func (f *Finder) Limit() int { return f.cfg.Limit }

// Scanning reports whether a scan runs now.
func (f *Finder) Scanning() bool { return f.scanning.Load() }

// Kinds returns the four hashes.
func (f *Finder) Kinds() []Kind {
	out := make([]Kind, len(f.kinds))
	for i, k := range f.kinds {
		out[i] = k.Kind
	}
	return out
}

// kind finds a hash by name; "" is DefaultKind.
func (f *Finder) kind(name string) (*kind, error) {
	if name == "" {
		name = DefaultKind
	}
	for _, k := range f.kinds {
		if k.Name == name {
			return k, nil
		}
	}
	return nil, fmt.Errorf("%w: no hash %q", ErrBadInput, name)
}

// LastScan returns the stats of the last scan; zero before the first.
func (f *Finder) LastScan() ScanStats {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.last
}

// Pictures returns the catalog sorted by id.
func (f *Finder) Pictures() []Picture {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]Picture, 0, len(f.pictures))
	for _, p := range f.pictures {
		out = append(out, *p)
	}
	slices.SortFunc(out, func(a, b Picture) int { return strings.Compare(a.ID, b.ID) })
	return out
}

// Path returns the file behind an id, which must be a catalog key.
func (f *Finder) Path(id string) (string, bool) {
	f.mu.Lock()
	_, ok := f.pictures[id]
	f.mu.Unlock()
	if !ok {
		return "", false
	}
	return filepath.Join(f.cfg.Dir, filepath.FromSlash(id)), true
}

// isImage reports whether a file is a picture the scan decodes, by its
// extension.
func isImage(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff":
		return true
	}
	return false
}

// walk lists the pictures under dir, skipping dot-directories, without
// hashes. A directory that does not exist holds nothing.
func walk(dir string) ([]Picture, error) {
	var out []Picture
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if path == dir && errors.Is(err, fs.ErrNotExist) {
				return filepath.SkipAll
			}
			return err
		}
		if d.IsDir() {
			if path != dir && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if !isImage(d.Name()) {
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
		out = append(out, Picture{ID: filepath.ToSlash(rel), Name: d.Name(), Size: info.Size(), ModTime: info.ModTime()})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("duplicates: walk %s: %w", dir, err)
	}
	return out, nil
}

// Scan walks the directory and hashes every picture with all four hashes,
// from one decode: the hashing itself is a fraction of reading the pixels.
// A picture whose size and time are those of the previous scan keeps its
// hashes, one that cannot be decoded keeps its error, and the ones that
// disappeared leave the catalog. The MediaIndexes that searches opened are
// brought in step afterwards. Only one scan runs at a time: a second one
// gets ErrScanning.
func (f *Finder) Scan(ctx context.Context) (ScanStats, error) {
	if !f.scanning.CompareAndSwap(false, true) {
		return ScanStats{}, ErrScanning
	}
	defer f.scanning.Store(false)
	start := time.Now()
	found, err := walk(f.cfg.Dir)
	if err != nil {
		return ScanStats{}, err
	}
	f.mu.Lock()
	previous := f.pictures
	f.mu.Unlock()

	stats := ScanStats{At: start}
	pictures := make(map[string]*Picture, len(found))
	for i := range found {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		p := &found[i]
		pictures[p.ID] = p
		if old, ok := previous[p.ID]; ok && old.Error == "" && old.Size == p.Size && old.ModTime.Equal(p.ModTime) {
			p.Width, p.Height, p.hashes = old.Width, old.Height, old.hashes
			stats.Unchanged++
			continue
		}
		hashStart := time.Now()
		err := f.hash(p)
		stats.HashUs += time.Since(hashStart).Microseconds()
		if err != nil {
			p.Error = err.Error()
			stats.Failed++
			f.logger.Warn("not hashed", "id", p.ID, "err", err)
			continue
		}
		stats.Hashed++
	}
	for id := range previous {
		if pictures[id] == nil {
			stats.Removed++
		}
	}
	f.mu.Lock()
	f.pictures = pictures
	f.mu.Unlock()

	// the indexes a search has opened follow the catalog
	for _, k := range f.kinds {
		if err := f.sync(ctx, k, false); err != nil {
			f.logger.Warn("index not brought in step", "hash", k.Name, "err", err)
		}
	}
	stats.ElapsedMs = time.Since(start).Milliseconds()
	f.mu.Lock()
	f.last = stats
	f.mu.Unlock()
	f.logger.Info("pictures hashed", "hashed", stats.Hashed, "unchanged", stats.Unchanged, "failed", stats.Failed,
		"removed", stats.Removed, "elapsed", time.Since(start).Round(time.Millisecond))
	return stats, nil
}

// hash decodes a picture once and fills in its size and its four hashes.
func (f *Finder) hash(p *Picture) error {
	img, err := decode(filepath.Join(f.cfg.Dir, filepath.FromSlash(p.ID)))
	if err != nil {
		return err
	}
	p.Width, p.Height = img.Bounds().Dx(), img.Bounds().Dy()
	p.hashes = make(map[string]phash.Hash, len(f.kinds))
	for _, k := range f.kinds {
		h, err := k.hasher.Hash(img)
		if err != nil {
			return err
		}
		p.hashes[k.Name] = h
	}
	return nil
}

// decode reads a picture file, refusing one of more pixels than phash
// hashes before its pixels are read.
func decode(path string) (image.Image, error) {
	file, err := os.Open(path) //nolint:gosec // a file the walk found under the scanned directory, only decoded
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	cfg, _, err := image.DecodeConfig(file)
	if err != nil {
		return nil, fmt.Errorf("not a picture: %w", err)
	}
	if int64(cfg.Width)*int64(cfg.Height) > maxPixels {
		return nil, fmt.Errorf("%d×%d is more than %d megapixels", cfg.Width, cfg.Height, maxPixels>>20)
	}
	if _, err := file.Seek(0, 0); err != nil {
		return nil, err
	}
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("not a picture: %w", err)
	}
	return img, nil
}

// Collection describes the MediaIndex of one hash, for the status page: the
// store it keeps its vectors in and its place there.
type Collection struct {
	Kind  string
	Store vector.Store
	Coord vector.Coord
}

// Collections lists the MediaIndexes searches have opened so far.
func (f *Finder) Collections() []Collection {
	var out []Collection
	for _, k := range f.kinds {
		k.mu.Lock()
		if k.index != nil {
			out = append(out, Collection{Kind: k.Name, Store: k.store, Coord: coordOf(k.Name)})
		}
		k.mu.Unlock()
	}
	return out
}

func coordOf(kindName string) vector.Coord {
	return vector.Coord{Index: indexName, Partition: indexPartition, Collection: collectionBase + kindName}
}

// Reset empties the MediaIndex of the named hash (search.MediaIndex.Reset);
// the next search with that hash indexes the pictures again.
func (f *Finder) Reset(ctx context.Context, kindName string) error {
	k, err := f.kind(kindName)
	if err != nil {
		return err
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.index == nil {
		return nil
	}
	if err := k.index.Reset(ctx); err != nil {
		return err
	}
	k.synced = map[string]stamp{}
	return nil
}

// sync brings the MediaIndex of a hash in step with the catalog: pictures
// that are new or changed are indexed with IndexImage, the ones that are
// gone deleted. With open false it does nothing for a hash no search has
// asked for yet; with open true it builds that hash's index first.
func (f *Finder) sync(ctx context.Context, k *kind, open bool) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.index == nil {
		if !open {
			return nil
		}
		c := coordOf(k.Name)
		ix, err := search.NewMediaIndex(k.hasher, k.store, c.Index, c.Partition, c.Collection)
		if err != nil {
			return err
		}
		k.index = ix
	}
	f.mu.Lock()
	want := make(map[string]stamp, len(f.pictures))
	for id, p := range f.pictures {
		if p.Error == "" {
			want[id] = stamp{size: p.Size, mod: p.ModTime}
		}
	}
	f.mu.Unlock()

	var gone []string
	for id := range k.synced {
		if _, ok := want[id]; !ok {
			gone = append(gone, id)
		}
	}
	if len(gone) > 0 {
		if err := k.index.Delete(ctx, gone...); err != nil {
			return err
		}
		for _, id := range gone {
			delete(k.synced, id)
		}
	}
	for id, st := range want {
		if have, ok := k.synced[id]; ok && have.size == st.size && have.mod.Equal(st.mod) {
			continue
		}
		if err := f.indexOne(ctx, k, id); err != nil {
			f.logger.Warn("not indexed", "hash", k.Name, "id", id, "err", err)
			continue
		}
		k.synced[id] = st
	}
	return ctx.Err()
}

func (f *Finder) indexOne(ctx context.Context, k *kind, id string) error {
	file, err := os.Open(filepath.Join(f.cfg.Dir, filepath.FromSlash(id)))
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()
	return k.index.IndexImage(ctx, id, file)
}

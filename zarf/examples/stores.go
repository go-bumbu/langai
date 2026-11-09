package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/tagging/sqlitecache"
	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/vector/sqlitevec"
)

// stores opens what the pages keep their derived data in: a vector store
// per model, and the Tagger's embedding cache. Each model gets a table set
// of its own, so two models' vectors never share one and switching back to
// a model finds its index as it was: with SQLite that is a database file
// per model in the data directory.
type stores struct {
	dataDir string
	// backend names the databases for the pages: "SQLite".
	backend string
	dbs     []*sql.DB

	// cache is the Tagger's embedding cache, cacheAt where it lives, and
	// cacheBytes the size of the database that holds it.
	cache      tagging.EmbeddingCache
	cacheAt    string
	cacheBytes func(ctx context.Context) (int64, error)
}

// openStores opens the embedding cache; the vector stores open as the pages
// ask for them, since each needs its model's dimensions.
func openStores(ctx context.Context, o options) (*stores, error) {
	if err := os.MkdirAll(o.dataDir, 0o750); err != nil {
		return nil, err
	}
	s := &stores{dataDir: o.dataDir, backend: "SQLite"}
	path := filepath.Join(o.dataDir, "autotag-cache.sqlite")
	db, err := s.open("sqlite3", path+"?_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		return nil, fmt.Errorf("open tag cache database: %w", err)
	}
	cache, err := sqlitecache.New(ctx, db)
	if err != nil {
		s.Close()
		return nil, fmt.Errorf("tag cache: %w", err)
	}
	s.cache, s.cacheAt = cache, "tagging/sqlitecache · "+path
	s.cacheBytes = func(context.Context) (int64, error) { return fileSizes(path, path+"-wal") }
	return s, nil
}

func (s *stores) open(driver, dsn string) (*sql.DB, error) {
	db, err := sql.Open(driver, dsn)
	if err != nil {
		return nil, err
	}
	s.dbs = append(s.dbs, db)
	return db, nil
}

// vectors opens the vector store of the model named name, whose vectors
// have dims dimensions, and returns it with a description of where it is
// and the path its files start with (a page keeps its catalog next to
// them).
func (s *stores) vectors(ctx context.Context, name string, dims int) (vector.Store, string, string, error) {
	base := filepath.Join(s.dataDir, "index-"+slug(name))
	db, err := s.open("sqlite3", base+".sqlite?_busy_timeout=5000&_journal_mode=WAL")
	if err != nil {
		return nil, "", "", fmt.Errorf("open database: %w", err)
	}
	store, err := sqlitevec.New(ctx, db, dims)
	if err != nil {
		return nil, "", "", fmt.Errorf("vector store: %w", err)
	}
	return store, "vector/sqlitevec · " + base + ".sqlite", base, nil
}

// Close closes every database the stores opened.
func (s *stores) Close() {
	for _, db := range s.dbs {
		_ = db.Close()
	}
}

// fileSizes sums the sizes of the files that exist among paths.
func fileSizes(paths ...string) (int64, error) {
	var total int64
	for _, p := range paths {
		info, err := os.Stat(p)
		switch {
		case errors.Is(err, os.ErrNotExist):
		case err != nil:
			return 0, err
		default:
			total += info.Size()
		}
	}
	return total, nil
}

// slug turns a model name into a file name: "google/siglip2-base-patch16-224"
// becomes "google-siglip2-base-patch16-224".
func slug(name string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '-' {
			return r
		}
		return '-'
	}, name)
}

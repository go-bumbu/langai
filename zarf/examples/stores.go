package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	_ "github.com/jackc/pgx/v5/stdlib"
	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/sense/tagging"
	"github.com/go-bumbu/sense/tagging/pgcache"
	"github.com/go-bumbu/sense/tagging/sqlitecache"
	"github.com/go-bumbu/sense/vector"
	"github.com/go-bumbu/sense/vector/pgvec"
	"github.com/go-bumbu/sense/vector/sqlitevec"
)

// stores opens what the pages keep their derived data in: a vector store
// per model, and the Tagger's embedding cache. Each model gets a table set
// of its own, so two models' vectors never share one and switching back to
// a model finds its index as it was. With SQLite, the default, that is a
// database file per model in the data directory (vector/sqlitevec,
// tagging/sqlitecache); with -postgres it is one Postgres database, with a
// generation of tables per model (vector/pgvec, tagging/pgcache).
type stores struct {
	dataDir string
	// backend names the databases for the pages.
	backend string
	// pg is the Postgres database, and pgName what to call it; nil with
	// SQLite.
	pg     *sql.DB
	pgName string
	dbs    []*sql.DB

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
	s := &stores{dataDir: o.dataDir}
	var err error
	if o.postgres != "" {
		err = s.openPostgres(ctx, o.postgres)
	} else {
		err = s.openSQLite(ctx)
	}
	if err != nil {
		s.Close()
		return nil, err
	}
	return s, nil
}

func (s *stores) openSQLite(ctx context.Context) error {
	s.backend = "SQLite, a file per model in " + s.dataDir
	path := filepath.Join(s.dataDir, "autotag-cache.sqlite")
	db, err := s.open("sqlite3", path+"?_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		return fmt.Errorf("open tag cache database: %w", err)
	}
	cache, err := sqlitecache.New(ctx, db)
	if err != nil {
		return fmt.Errorf("tag cache: %w", err)
	}
	s.cache, s.cacheAt = cache, "tagging/sqlitecache · "+path
	s.cacheBytes = func(context.Context) (int64, error) { return fileSizes(path, path+"-wal") }
	return nil
}

func (s *stores) openPostgres(ctx context.Context, dsn string) error {
	s.pgName = "postgres"
	if u, err := url.Parse(dsn); err == nil && u.Host != "" {
		s.pgName = "postgres " + u.Host + u.Path // never the password
	}
	s.backend = "Postgres with pgvector (" + s.pgName + "), a generation of tables per model"
	db, err := s.open("pgx", dsn)
	if err != nil {
		return fmt.Errorf("open postgres: %w", err)
	}
	cache, err := pgcache.New(ctx, db)
	if err != nil {
		return fmt.Errorf("tag cache: %w", err)
	}
	s.pg, s.cache, s.cacheAt = db, cache, "tagging/pgcache · "+s.pgName
	s.cacheBytes = func(ctx context.Context) (int64, error) {
		var n int64
		err := db.QueryRowContext(ctx, `SELECT pg_database_size(current_database())`).Scan(&n)
		return n, err
	}
	return nil
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
// and the path its page keeps its own files under (the media page's
// catalog), which is the store's own with SQLite.
func (s *stores) vectors(ctx context.Context, name string, dims int) (store vector.Store, where, base string, err error) {
	base = filepath.Join(s.dataDir, "index-"+slug(name))
	if s.pg != nil {
		// a generation is 1 to 12 lowercase letters and digits: a digest of
		// the model's name
		gen := generation(name)
		store, err = pgvec.New(ctx, s.pg, dims, pgvec.WithGeneration(gen))
		if err != nil {
			return nil, "", "", fmt.Errorf("vector store: %w", err)
		}
		return store, "vector/pgvec · " + s.pgName + " · generation " + gen, base + "-postgres", nil
	}
	db, err := s.open("sqlite3", base+".sqlite?_busy_timeout=5000&_journal_mode=WAL")
	if err != nil {
		return nil, "", "", fmt.Errorf("open database: %w", err)
	}
	store, err = sqlitevec.New(ctx, db, dims)
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

// generation names a model's table set in Postgres: "m" and the first
// eleven hex digits of the SHA-256 of its name.
func generation(model string) string {
	sum := sha256.Sum256([]byte(model))
	return "m" + hex.EncodeToString(sum[:])[:11]
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

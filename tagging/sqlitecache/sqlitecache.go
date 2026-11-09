// Package sqlitecache is a tagging.EmbeddingCache backed by SQLite. It needs no
// extension: the embeddings of every index live in one plain table. It uses
// SQLite's upserts, so SQLite 3.24 or newer.
//
// The cache uses the *sql.DB it is given, so the application imports a SQLite
// driver and opens the database. SQLite allows one writer at a time. When
// several goroutines write, open the database so that a writer waits for its
// turn rather than failing with "database is locked", for example with
// github.com/mattn/go-sqlite3:
//
//	db, err := sql.Open("sqlite3", "autotag.db?_txlock=immediate&_busy_timeout=5000")
//
// _txlock=immediate makes every transaction take the write lock as it begins,
// and waiting for that lock is what the busy timeout covers.
package sqlitecache

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/go-bumbu/sense/tagging"
	"github.com/go-bumbu/sense/tagging/internal/sqlcache"
)

// Cache is a tagging.EmbeddingCache on SQLite, in one table of db: Get,
// Put, Touch, DeleteIndex and Prune are the interface's. It is safe for
// concurrent use.
type Cache struct {
	*cache
}

// cache names the shared SQL implementation, so that Cache embeds it under
// an unexported field.
type cache = sqlcache.Cache

// verify the interface is actually satisfied
var _ tagging.EmbeddingCache = (*Cache)(nil)

// New returns a Cache using db, creating its table if needed, and checks
// that db is a working SQLite database.
func New(ctx context.Context, db *sql.DB) (*Cache, error) {
	return open(ctx, db, time.Now)
}

// open is New on the clock now, which the tests set.
func open(ctx context.Context, db *sql.DB, now func() time.Time) (*Cache, error) {
	if db == nil {
		return nil, errors.New("nil database handle")
	}
	var v string
	if err := db.QueryRowContext(ctx, `SELECT sqlite_version()`).Scan(&v); err != nil {
		return nil, fmt.Errorf("sqlite not available: %w", err)
	}
	c, err := sqlcache.New(ctx, db, sqlcache.SQLite, now)
	if err != nil {
		return nil, err
	}
	return &Cache{c}, nil
}

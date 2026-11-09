// Package pgcache is a tagging.EmbeddingCache backed by PostgreSQL. It needs
// plain Postgres, with no extension: the embeddings of every index live in one
// ordinary table. It uses upserts (INSERT ... ON CONFLICT), so PostgreSQL 9.5
// or newer.
//
// The cache uses the *sql.DB it is given, so the application imports a Postgres
// driver and opens the database, for example with github.com/jackc/pgx/v5/stdlib:
//
//	db, err := sql.Open("pgx", "postgres://user:password@localhost:5432/app")
//
// Several goroutines, or processes, can write at once with no special setup: a
// Put is one transaction of upserts made in one fixed order of the keys (model,
// role, hash), so two Puts of overlapping keys lock their rows in the same
// order and the second waits for the first instead of deadlocking with it.
package pgcache

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/go-bumbu/sense/tagging"
	"github.com/go-bumbu/sense/tagging/internal/sqlcache"
)

// Cache is a tagging.EmbeddingCache on Postgres, in one table of db: Get,
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
// that db is a working Postgres database.
func New(ctx context.Context, db *sql.DB) (*Cache, error) {
	return open(ctx, db, time.Now)
}

// open is New on the clock now, which the tests set.
func open(ctx context.Context, db *sql.DB, now func() time.Time) (*Cache, error) {
	if db == nil {
		return nil, errors.New("nil database handle")
	}
	var v string
	if err := db.QueryRowContext(ctx, `SELECT current_setting('server_version_num')`).Scan(&v); err != nil {
		return nil, fmt.Errorf("postgres not available: %w", err)
	}
	c, err := sqlcache.New(ctx, db, sqlcache.Postgres, now)
	if err != nil {
		return nil, err
	}
	return &Cache{c}, nil
}

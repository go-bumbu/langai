// Package sqlitestore is an autotag.Store backed by SQLite. It needs no
// extension: the tags, items and links of each index live in plain tables, and
// the Tagger scores the few tags of a space itself. It uses SQLite's upserts,
// so SQLite 3.24 or newer.
//
// The store uses the *sql.DB it is given, so the application imports a SQLite
// driver and opens the database. SQLite allows one writer at a time. When
// several goroutines write, open the database so that a writer waits for its
// turn rather than failing with "database is locked", for example with
// github.com/mattn/go-sqlite3:
//
//	db, err := sql.Open("sqlite3", "autotag.db?_txlock=immediate&_busy_timeout=5000")
//
// _txlock=immediate makes every transaction take the write lock as it begins,
// and waiting for that lock is what the busy timeout covers. A transaction that
// starts with a read and takes the lock only at its first write cannot wait for
// it: it fails at once with "database is locked" if another writer holds it by
// then.
package sqlitestore

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/go-bumbu/langai/autotag"
	"github.com/go-bumbu/langai/autotag/internal/sqlstore"
)

type options struct{ generation string }

// Option customizes a store.
type Option func(*options)

// WithGeneration keeps the store's tables apart from those of every other
// generation in the same database (see tablename.ForPrefix), so the vectors of
// a different embedding model can be built next to the current ones before an
// application switches over. generation is "" (the default) or 1-12 lowercase
// letters and digits; New rejects any other.
func WithGeneration(generation string) Option {
	return func(o *options) { o.generation = generation }
}

// New returns a Store using db and checks that db is a working SQLite database.
// The tables of each index are created when it is first used.
func New(ctx context.Context, db *sql.DB, opts ...Option) (autotag.Store, error) {
	if db == nil {
		return nil, errors.New("nil database handle")
	}
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	s, err := sqlstore.New(db, sqlstore.SQLite, o.generation)
	if err != nil {
		return nil, err
	}
	var v string
	if err := db.QueryRowContext(ctx, `SELECT sqlite_version()`).Scan(&v); err != nil {
		return nil, fmt.Errorf("sqlite not available: %w", err)
	}
	return s, nil
}

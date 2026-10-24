// Package pgstore is an autotag.Store backed by PostgreSQL. It needs plain
// Postgres, with no extension: the tags, items and links of each index live in
// ordinary tables, and the Tagger scores the few tags of a space itself. It uses
// upserts (INSERT ... ON CONFLICT), so PostgreSQL 9.5 or newer.
//
// The store uses the *sql.DB it is given, so the application imports a Postgres
// driver and opens the database, for example with github.com/jackc/pgx/v5/stdlib:
//
//	db, err := sql.Open("pgx", "postgres://user:password@localhost:5432/app")
//
// Several goroutines can write at once with no special setup. A write that
// changes more than one row locks them in a fixed order, so concurrent writes
// queue up on the row locks rather than deadlock.
//
// That locking relies on READ COMMITTED isolation, Postgres's default, where
// each statement sees the rows committed before it ran. The store begins every
// transaction of its own at READ COMMITTED, whatever the setting
// default_transaction_isolation says. Its single-statement writes run outside
// a transaction, at the session's default level, so keep that at READ
// COMMITTED for the store's connections: under REPEATABLE READ or SERIALIZABLE
// they can fail with a serialization error (SQLSTATE 40001), which the store
// does not retry.
package pgstore

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

// New returns a Store using db and checks that db is a working Postgres
// database. The tables of each index are created when it is first used.
func New(ctx context.Context, db *sql.DB, opts ...Option) (autotag.Store, error) {
	if db == nil {
		return nil, errors.New("nil database handle")
	}
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	s, err := sqlstore.New(db, sqlstore.Postgres, o.generation)
	if err != nil {
		return nil, err
	}
	var v string
	if err := db.QueryRowContext(ctx, `SELECT current_setting('server_version_num')`).Scan(&v); err != nil {
		return nil, fmt.Errorf("postgres not available: %w", err)
	}
	return s, nil
}

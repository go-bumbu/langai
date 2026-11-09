// Package pgvec is a vector.Store backed by Postgres with the pgvector
// extension. The database user must be able to run CREATE EXTENSION vector,
// or the extension must already be installed.
//
// Each index has a table set of its own, created by the first write; a role
// that only reads can search an index another role created. A search ranks
// the chunks of its scope exactly: an approximate (HNSW) index would filter
// the scope only after collecting its nearest rows of the whole table, so a
// scope without rows among those would come back short. Text columns sort
// by byte (COLLATE "C"), the order the other stores use.
package pgvec

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/pgvector/pgvector-go"

	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/vector/internal/sqlstore"
)

// Store implements vector.Store on Postgres + pgvector, with one physical
// table set per index (e.g. a tenant) and generation. It is safe for
// concurrent use.
type Store struct {
	*sqlstore.Store
}

var _ vector.Store = (*Store)(nil)

// Option customizes a Store.
type Option func(*options)

type options struct{ generation string }

// WithGeneration keeps the store's tables apart from those of every other
// generation in the same database, so the vectors of a different embedding
// model can be built next to the current ones before an application switches
// over; DropGeneration retires the old ones afterwards. generation is "" (the
// default) or 1-12 lowercase letters and digits. An index's tables are named
// ss_<generation>_<hash>_<suffix>, where <hash> is the first 16 hex digits of
// the SHA-256 of the index, so any index string makes a safe identifier and
// two indexes never share tables; the default generation's are
// ss_<hash>_<suffix>, the names the stores used before generations existed.
func WithGeneration(generation string) Option {
	return func(o *options) { o.generation = generation }
}

// dialect is Postgres's: vectors as pgvector's vector type, id lists read
// with json_array_elements_text, and the schema created in one transaction
// under an advisory lock, so two processes creating an index's tables at once
// wait for each other. The schema before this one had no meta table.
var dialect = sqlstore.Dialect{
	Name:     "postgres",
	Numbered: true,
	Schema: func(docs, chunks, meta string, dims int) []string {
		return []string{
			`CREATE TABLE IF NOT EXISTS ` + docs + ` (
				partition TEXT COLLATE "C" NOT NULL, collection TEXT COLLATE "C" NOT NULL,
				doc_id TEXT COLLATE "C" NOT NULL, content_hash TEXT NOT NULL,
				PRIMARY KEY (partition, collection, doc_id))`,
			fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
				id BIGSERIAL PRIMARY KEY,
				partition TEXT COLLATE "C" NOT NULL, collection TEXT COLLATE "C" NOT NULL,
				doc_id TEXT COLLATE "C" NOT NULL, seq INT NOT NULL, text TEXT NOT NULL,
				embedding vector(%d) NOT NULL)`, chunks, dims),
			`CREATE INDEX IF NOT EXISTS ` + chunks + `_doc ON ` + chunks + ` (partition, collection, doc_id, seq)`,
			`CREATE TABLE IF NOT EXISTS ` + meta + ` (name TEXT COLLATE "C" PRIMARY KEY, value TEXT NOT NULL)`,
		}
	},
	LockDDL:     `SELECT pg_advisory_xact_lock(hashtext(?))`,
	Legacy:      "docs",
	TableExists: `SELECT to_regclass(?) IS NOT NULL`,
	ListTables:  `SELECT tablename FROM pg_tables WHERE schemaname = current_schema() AND tablename LIKE 'ss\_%' ESCAPE '\'`,
	Distance:    `(embedding <=> ?)::float8`,
	IDs:         `SELECT json_array_elements_text(CAST(? AS json))`,
	Encode:      func(v []float32) any { return pgvector.NewVector(v) },
	Decode: func(src any) ([]float32, error) {
		var v pgvector.Vector
		if err := v.Scan(src); err != nil {
			return nil, err
		}
		return v.Slice(), nil
	},
	MissingTable: func(err error) bool { return sqlState(err) == undefinedTable },
	LockRows:     `FOR UPDATE`,
	Retryable: func(err error) bool {
		code := sqlState(err)
		return code == deadlockDetected || code == serializationFailure
	},
}

// The SQLSTATEs the store tells apart: a statement naming a table that does
// not exist, and the transient failures a write transaction is run again
// for.
const (
	undefinedTable       = "42P01"
	deadlockDetected     = "40P01"
	serializationFailure = "40001"
)

// sqlState returns the SQLSTATE code of a Postgres error, "" for any other.
// Both pgx (pgconn.PgError) and lib/pq (pq.Error) report it through a
// SQLState method, so no driver is imported.
func sqlState(err error) string {
	var e interface{ SQLState() string }
	if errors.As(err, &e) {
		return e.SQLState()
	}
	return ""
}

// New returns a Store using db and ensures the pgvector extension exists.
// dims must match the embedder's Dimensions(); an index whose tables hold
// vectors of other dimensions fails with vector.ErrDimensions, and the
// vectors of another model, whatever their size, belong in another
// generation (WithGeneration).
func New(ctx context.Context, db *sql.DB, dims int, opts ...Option) (*Store, error) {
	if db == nil {
		return nil, errors.New("nil database handle")
	}
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	s, err := sqlstore.New(db, dialect, dims, o.generation)
	if err != nil {
		return nil, err
	}
	if _, err := db.ExecContext(ctx, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return nil, fmt.Errorf("pgvector extension not available (install it or grant CREATE EXTENSION): %w", err)
	}
	return &Store{s}, nil
}

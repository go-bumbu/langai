// Package sqlitevec is a vector.Store backed by SQLite with the sqlite-vec
// extension. The extension is compiled in (CGO) and registered for every
// SQLite connection the process opens afterwards, so an application that
// opens its connections elsewhere (through an ORM, say) imports this package,
// blank if need be, before it opens the first one.
//
// Each index has a table set of its own; the vectors are BLOB columns of
// packed float32s, and a search ranks the chunks of its scope exactly with
// sqlite-vec's vec_distance_cosine, so there is no cap on how many it
// returns.
//
// SQLite allows one writer at a time. Every write transaction of the store
// starts with a write, so it takes the write lock as it begins, and a busy
// timeout makes a second writer wait for its turn rather than fail with
// "database is locked". Open a file database (every connection of an
// in-memory one is a database of its own) with a busy timeout, and in WAL
// mode so that readers do not wait for writers, for example with
// github.com/mattn/go-sqlite3:
//
//	db, err := sql.Open("sqlite3", "search.db?_busy_timeout=5000&_journal_mode=WAL")
//
// _txlock=immediate is not needed for this store; an application whose own
// transactions on the same database read before they write still needs it.
package sqlitevec

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"

	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/vector/internal/sqlstore"
)

func init() { sqlite_vec.Auto() }

// Store implements vector.Store on SQLite + sqlite-vec with one physical
// table set per index (e.g. a tenant) and generation, created by the first
// write. It is safe for concurrent use.
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

// dialect is SQLite's: vectors as little-endian float32 BLOBs, ids lists
// read with json_each, and the schema created statement by statement, the
// meta table last. The tables of the schema before this one had a vec0
// virtual table, named vec.
var dialect = sqlstore.Dialect{
	Name: "sqlite",
	Schema: func(docs, chunks, meta string, _ int) []string {
		return []string{
			`CREATE TABLE IF NOT EXISTS ` + docs + ` (
				partition TEXT NOT NULL, collection TEXT NOT NULL,
				doc_id TEXT NOT NULL, content_hash TEXT NOT NULL,
				PRIMARY KEY (partition, collection, doc_id))`,
			`CREATE TABLE IF NOT EXISTS ` + chunks + ` (
				id INTEGER PRIMARY KEY,
				partition TEXT NOT NULL, collection TEXT NOT NULL,
				doc_id TEXT NOT NULL, seq INTEGER NOT NULL, text TEXT NOT NULL,
				embedding BLOB NOT NULL)`,
			`CREATE INDEX IF NOT EXISTS ` + chunks + `_doc ON ` + chunks + ` (partition, collection, doc_id, seq)`,
			`CREATE TABLE IF NOT EXISTS ` + meta + ` (name TEXT PRIMARY KEY, value TEXT NOT NULL)`,
		}
	},
	Legacy:      "vec",
	TableExists: `SELECT count(*) > 0 FROM sqlite_master WHERE type = 'table' AND name = ?`,
	ListTables:  `SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'ss\_%' ESCAPE '\'`,
	Distance:    `vec_distance_cosine(embedding, ?)`,
	IDs:         `SELECT value FROM json_each(?)`,
	Encode:      func(v []float32) any { return encode(v) },
	Decode:      decode,
}

// New returns a Store using db and checks the sqlite-vec extension is
// loaded. dims must match the embedder's Dimensions(); an index whose tables
// hold vectors of other dimensions fails with vector.ErrDimensions, and the
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
	var v string
	if err := db.QueryRowContext(ctx, `SELECT vec_version()`).Scan(&v); err != nil {
		return nil, fmt.Errorf("sqlite-vec extension not available: %w", err)
	}
	return &Store{s}, nil
}

// encode packs v as little-endian float32s, the layout sqlite-vec reads.
func encode(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

// decode is encode's inverse.
func decode(src any) ([]float32, error) {
	b, ok := src.([]byte)
	if !ok {
		return nil, fmt.Errorf("vector column holds %T, not a BLOB", src)
	}
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("vector blob of %d bytes is not a float32 array", len(b))
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, nil
}

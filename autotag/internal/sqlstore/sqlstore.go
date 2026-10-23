// Package sqlstore implements autotag.Store on database/sql for the databases
// a Dialect describes. The public backends, sqlitestore and pgstore, are this
// package with their own Dialect and their own checks of the database they get.
//
// Every index (a tenant) gets three plain tables per generation, created on
// first use: the tags, the items with their embeddings, and the links between
// them. There is no vector extension: a space has few tags, and the Tagger
// scores them in Go. Their names come from tablename, so an index never
// reaches SQL as an identifier. A partition is a column of those tables.
package sqlstore

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"

	"github.com/go-bumbu/langai/autotag"
	"github.com/go-bumbu/langai/internal/tablename"
)

// prefix starts the name of every table the store creates (see
// tablename.ForPrefix).
const prefix = "autotag"

// Dialect holds what differs between the SQL databases a Store runs on. The
// statements are written once, with ? placeholders, and are otherwise plain SQL
// that these databases share.
type Dialect struct {
	Blob, Float, Int string // vector, cutoff and version column types
	Numbered         bool   // $1, $2… placeholders instead of ?
	ForUpdate        string // row lock for reads inside a write transaction
	// Collate is appended to an id column in an ORDER BY, and to the item text a
	// write compares, so that order and equality are byte-wise on every backend,
	// as they are in Go. It is empty where the default collation already is.
	Collate string
}

// SQLite is the Dialect of SQLite, whose default collation, BINARY, compares
// bytes, and which has no row locks: a write transaction locks the database.
var SQLite = Dialect{Blob: "BLOB", Float: "REAL", Int: "INTEGER"}

// Postgres is the Dialect of PostgreSQL, whose default collation follows the
// database's locale: "C" compares bytes.
var Postgres = Dialect{
	Blob: "BYTEA", Float: "DOUBLE PRECISION", Int: "BIGINT",
	Numbered: true, ForUpdate: " FOR UPDATE", Collate: ` COLLATE "C"`,
}

// tables are the names of the tables of one index.
type tables struct{ tags, items, links string }

// Store implements autotag.Store on SQL, with one table set per index and
// generation, created lazily.
//
// A write that changes more than one row runs in a transaction. On a database
// with row locks, writes take their locks in one order: the item row first,
// then the tag rows, several of them in id order, then the link rows. A write
// that removes a row takes its lock before it touches the rows that depend on
// it, and does nothing more if the row was not there. So concurrent writes queue
// up rather than deadlock, no link is left behind for a row that is gone, and a
// delete of an absent row never removes a link made meanwhile for a new one.
type Store struct {
	db         *sql.DB
	d          Dialect
	generation string
	mu         sync.Mutex
	created    map[string]tables
}

var _ autotag.Store = (*Store)(nil)

// New returns a Store that keeps its tables in db, in the SQL of d. generation
// keeps the tables apart from those of every other generation in the same
// database (see tablename.ForPrefix): "" or 1-12 lowercase letters and digits.
// It does not look at the database, which its caller knows how to check.
func New(db *sql.DB, d Dialect, generation string) (*Store, error) {
	if db == nil {
		return nil, errors.New("nil database handle")
	}
	if err := tablename.CheckGeneration(generation); err != nil {
		return nil, err
	}
	return &Store{db: db, d: d, generation: generation, created: map[string]tables{}}, nil
}

// ensure returns the names of the index's tables, creating them if this store
// has not yet. Naming them fails on an empty index, so a method that calls it
// first never queries for one.
func (s *Store) ensure(ctx context.Context, index string) (tables, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if t, ok := s.created[index]; ok {
		return t, nil
	}
	var t tables
	for _, n := range []struct {
		suffix string
		name   *string
	}{{"tags", &t.tags}, {"items", &t.items}, {"links", &t.links}} {
		name, err := tablename.ForPrefix(prefix, s.generation, index, n.suffix)
		if err != nil {
			return tables{}, fmt.Errorf("autotag: %w", err)
		}
		*n.name = name
	}
	for _, st := range s.schema(t) {
		if _, err := s.db.ExecContext(ctx, st); err != nil {
			return tables{}, fmt.Errorf("create autotag schema for index %q: %w", index, err)
		}
	}
	s.created[index] = t
	return t, nil
}

// schema returns the statements that create the tables of one index.
func (s *Store) schema(t tables) []string {
	return []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %[1]s (
			partition TEXT NOT NULL, tag_id TEXT NOT NULL, name TEXT NOT NULL,
			name_vec %[2]s, vec %[2]s, cutoff %[3]s NOT NULL DEFAULT 0,
			model TEXT NOT NULL DEFAULT '', params TEXT NOT NULL DEFAULT '',
			version %[4]s NOT NULL, built %[4]s NOT NULL DEFAULT 0,
			PRIMARY KEY (partition, tag_id))`, t.tags, s.d.Blob, s.d.Float, s.d.Int),
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %[1]s (
			partition TEXT NOT NULL, item_id TEXT NOT NULL, text TEXT NOT NULL,
			vec %[2]s, model TEXT NOT NULL,
			PRIMARY KEY (partition, item_id))`, t.items, s.d.Blob),
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %[1]s (
			partition TEXT NOT NULL, item_id TEXT NOT NULL, tag_id TEXT NOT NULL,
			PRIMARY KEY (partition, item_id, tag_id))`, t.links),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %[1]s_tag ON %[1]s(partition, tag_id)`, t.links),
	}
}

// q fills a statement's %s verbs in, with table names and the Dialect's
// clauses, and numbers its ? placeholders where the Dialect wants it. A table
// name is a hex hash of the index (injection-safe), and every value reaches
// the database as a bound parameter, never through this function.
func (s *Store) q(format string, a ...any) string {
	query := fmt.Sprintf(format, a...)
	if s.d.Numbered {
		return rebind(query)
	}
	return query
}

// rebind numbers the ? placeholders of query: $1, $2… No statement holds a
// literal ?.
func rebind(query string) string {
	var b strings.Builder
	n := 0
	for i := 0; i < len(query); i++ {
		if query[i] != '?' {
			b.WriteByte(query[i])
			continue
		}
		n++
		b.WriteByte('$')
		b.WriteString(strconv.Itoa(n))
	}
	return b.String()
}

// encode returns the little-endian bytes of the float32s of v, and nil for an
// empty vector, which the database keeps as NULL.
func encode(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

// decode is encode's inverse. NULL and empty are the nil vector, and a blob
// that does not hold a whole number of float32s was cut short.
func decode(b []byte) ([]float32, error) {
	if len(b) == 0 {
		return nil, nil
	}
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("vector of %d bytes is not a whole number of float32s", len(b))
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, nil
}

// dbtx is what *sql.DB and *sql.Tx share, so a helper runs in either.
type dbtx interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

// scanner is what *sql.Row and *sql.Rows share.
type scanner interface{ Scan(dest ...any) error }

// inTx runs fn in a transaction: it commits when fn returns nil, and rolls back
// otherwise. fn must use tx and never s.db, whose other connection would wait
// for the locks tx holds.
func (s *Store) inTx(ctx context.Context, fn func(tx *sql.Tx) error) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit()
}

// exec runs a statement and drops its result.
func exec(ctx context.Context, q dbtx, query string, args ...any) error {
	_, err := q.ExecContext(ctx, query, args...)
	return err
}

// changed reports whether a statement, passed as ExecContext returns it,
// changed its one row.
func changed(res sql.Result, err error) (bool, error) {
	if err != nil {
		return false, err
	}
	n, err := res.RowsAffected()
	return n == 1, err
}

// exists reports whether the query finds a row.
func exists(ctx context.Context, q dbtx, query string, args ...any) (bool, error) {
	var one int
	err := q.QueryRowContext(ctx, query, args...).Scan(&one)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	return err == nil, err
}

// queryIDs reads a column of ids, in the order the query gives.
func queryIDs(ctx context.Context, q dbtx, query string, args ...any) ([]string, error) {
	rows, err := q.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

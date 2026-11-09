// Package sqlcache implements tagging.EmbeddingCache on database/sql for the
// databases a Dialect describes. The public backends, sqlitecache and pgcache,
// are this package with their own Dialect and their own check of the database
// they get.
//
// All indexes share one plain table, autotag_embeddings, keyed by index, model,
// role and the SHA-256 of the text. There is no vector extension and no
// per-index table: the cache is derived data that is only ever read by key, so
// an index never reaches SQL as an identifier, only as a bound value.
package sqlcache

import (
	"bytes"
	"cmp"
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/tagging"
)

// table is the name of the one table of the cache. The prefix is the legacy
// one of the tagging tables.
const table = "autotag_embeddings"

// chunk is how many hashes one statement matches. With the three other
// parameters it stays far below the limits of SQLite (32766) and Postgres
// (65535).
const chunk = 500

var (
	errEmptyIndex  = errors.New("tagging: empty index")
	errEmptyVector = errors.New("tagging: empty vector")
)

// checkIndex refuses an index no cache takes: the empty one, and one a
// database cannot store as text (see checkName).
func checkIndex(index string) error {
	if index == "" {
		return errEmptyIndex
	}
	return checkName("index", index)
}

// checkName refuses an index or a model with a NUL byte or invalid UTF-8, as
// every tagging.EmbeddingCache does: Postgres stores neither in a text column
// and SQLite both, so the check is made here, with one message, before either
// database is asked.
func checkName(what, name string) error {
	if !utf8.ValidString(name) || strings.ContainsRune(name, 0) {
		return fmt.Errorf("tagging: %s %q holds a NUL byte or is not valid UTF-8", what, name)
	}
	return nil
}

// Dialect holds what differs between the SQL databases a Cache runs on. The
// statements are written once, with ? placeholders, and are otherwise plain SQL
// that these databases share.
type Dialect struct {
	Blob, Int string // vector/hash and role/time column types
	Numbered  bool   // $1, $2… placeholders instead of ?
	// SchemaLock, when not empty, is a statement that makes the transaction
	// creating the schema wait for any other one doing the same.
	SchemaLock string
}

// SQLite is the Dialect of SQLite. It needs no SchemaLock: a transaction that
// writes already waits for the one writer the database allows.
var SQLite = Dialect{Blob: "BLOB", Int: "INTEGER"}

// Postgres is the Dialect of PostgreSQL. Two CREATE TABLE IF NOT EXISTS that
// run together both find no table and one then fails on Postgres's own
// catalog, so processes that start together take a transaction-level advisory
// lock first; its key is the eight bytes of "autotag_" as a number, for no
// reason but to be this table's own.
var Postgres = Dialect{Blob: "BYTEA", Int: "BIGINT", Numbered: true,
	SchemaLock: "SELECT pg_advisory_xact_lock(7022647215996888927)"}

// Cache implements tagging.EmbeddingCache on SQL. It is safe for concurrent
// use as far as its *sql.DB is.
type Cache struct {
	db  *sql.DB
	d   Dialect
	now func() time.Time
}

var _ tagging.EmbeddingCache = (*Cache)(nil)

// New returns a Cache that keeps its table in db, in the SQL of d, creating the
// table if it is missing. now is the clock the last-used times come from;
// tests inject theirs; a nil one is refused rather than left to panic on the
// first Put. It does not look at the database, which its caller knows how to
// check.
func New(ctx context.Context, db *sql.DB, d Dialect, now func() time.Time) (*Cache, error) {
	switch {
	case db == nil:
		return nil, errors.New("nil database handle")
	case now == nil:
		return nil, errors.New("nil clock")
	}
	c := &Cache{db: db, d: d, now: now}
	if err := c.createSchema(ctx); err != nil {
		return nil, fmt.Errorf("create autotag embeddings schema: %w", err)
	}
	return c, nil
}

// createSchema creates the table and its index if they are missing, in one
// transaction that first takes the Dialect's SchemaLock, so several processes
// starting together create them one after the other.
func (c *Cache) createSchema(ctx context.Context) error {
	tx, err := c.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	var statements []string
	if c.d.SchemaLock != "" {
		statements = append(statements, c.d.SchemaLock)
	}
	statements = append(statements,
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			idx TEXT NOT NULL, model TEXT NOT NULL, role %[2]s NOT NULL,
			hash %[3]s NOT NULL, vec %[3]s NOT NULL, used %[2]s NOT NULL,
			PRIMARY KEY (idx, model, role, hash))`, table, c.d.Int, c.d.Blob),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_used ON %s (used)`, table, table))
	for _, st := range statements {
		if _, err := tx.ExecContext(ctx, st); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// q numbers the ? placeholders of a statement where the Dialect wants it.
func (c *Cache) q(query string) string {
	if c.d.Numbered {
		return rebind(query)
	}
	return query
}

// group is the set of hashes of one model and role.
type group struct {
	model string
	role  sense.Role
}

// Get implements tagging.EmbeddingCache. Keys are grouped by model and role and
// matched 500 at a time; the entries found last used
// tagging.CacheRefreshAfter ago or longer, to the second, get their last-used
// time rewritten, as in every backend (see cachetest). When that
// write fails, on a read-only replica or a locked file, the vectors were read
// all the same: Get returns all of them together with an error wrapping
// tagging.ErrCacheWrite.
func (c *Cache) Get(ctx context.Context, index string, keys []tagging.CacheKey) (map[tagging.CacheKey][]float32, error) {
	if err := checkIndex(index); err != nil {
		return nil, err
	}
	out := make(map[tagging.CacheKey][]float32, len(keys))
	if len(keys) == 0 {
		return out, nil
	}
	groups, err := grouped(keys)
	if err != nil {
		return nil, err
	}
	now := c.now().Unix()
	staleAt := now - int64(tagging.CacheRefreshAfter/time.Second)
	var notRefreshed error
	for g, hashes := range groups {
		for start := 0; start < len(hashes); start += chunk {
			part := hashes[start:min(start+chunk, len(hashes))]
			stale, err := c.read(ctx, index, g, part, staleAt, out)
			if err != nil {
				return nil, err
			}
			if notRefreshed != nil {
				continue // the database takes no write: read the rest and say so once
			}
			// only the entries still stale: two readers do not rewrite a
			// row one after the other for nothing
			notRefreshed = c.touch(ctx, index, g, stale, now, &staleAt)
		}
	}
	if notRefreshed != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return out, fmt.Errorf("%w: refresh last-used times: %w", tagging.ErrCacheWrite, notRefreshed)
	}
	return out, nil
}

// grouped returns the hashes of keys by model and role, each key once. It
// refuses a model no database stores (see checkName).
func grouped(keys []tagging.CacheKey) (map[group][][]byte, error) {
	groups := map[group][][]byte{}
	seen := map[tagging.CacheKey]bool{}
	for _, k := range keys {
		if seen[k] {
			continue
		}
		if err := checkName("model", k.Model); err != nil {
			return nil, err
		}
		seen[k] = true
		g := group{k.Model, k.Role}
		groups[g] = append(groups[g], k.Hash[:])
	}
	return groups, nil
}

// read adds to out the entries of hashes that the cache holds and returns the
// hashes of those last used at staleAt or before.
func (c *Cache) read(ctx context.Context, index string, g group, hashes [][]byte, staleAt int64, out map[tagging.CacheKey][]float32) ([][]byte, error) {
	args := append([]any{index, g.model, int64(g.role)}, bytesArgs(hashes)...)
	rows, err := c.db.QueryContext(ctx, c.q(`SELECT hash, vec, used FROM `+table+
		` WHERE idx = ? AND model = ? AND role = ? AND hash IN (`+placeholders(len(hashes))+`)`), args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var stale [][]byte
	for rows.Next() {
		var hash, blob []byte
		var used int64
		if err := rows.Scan(&hash, &blob, &used); err != nil {
			return nil, err
		}
		k := tagging.CacheKey{Model: g.model, Role: g.role}
		if len(hash) != len(k.Hash) {
			return nil, fmt.Errorf("hash of %d bytes, want %d", len(hash), len(k.Hash))
		}
		copy(k.Hash[:], hash)
		v, err := decode(blob)
		if err != nil {
			return nil, err
		}
		out[k] = v
		if used <= staleAt {
			stale = append(stale, hash)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return stale, nil
}

// touch sets the last-used time of the entries of hashes to now, 500 at a
// time: of all of them, or with usedBy only of the ones last used at that
// time or before.
func (c *Cache) touch(ctx context.Context, index string, g group, hashes [][]byte, now int64, usedBy *int64) error {
	for start := 0; start < len(hashes); start += chunk {
		part := hashes[start:min(start+chunk, len(hashes))]
		args := []any{now, index, g.model, int64(g.role)}
		guard := ""
		if usedBy != nil {
			guard = ` AND used <= ?`
			args = append(args, *usedBy)
		}
		args = append(args, bytesArgs(part)...)
		if _, err := c.db.ExecContext(ctx, c.q(`UPDATE `+table+` SET used = ? WHERE idx = ? AND model = ? AND role = ?`+guard+
			` AND hash IN (`+placeholders(len(part))+`)`), args...); err != nil {
			return err
		}
	}
	return nil
}

// Touch implements tagging.EmbeddingCache.
func (c *Cache) Touch(ctx context.Context, index string, keys []tagging.CacheKey) error {
	if err := checkIndex(index); err != nil {
		return err
	}
	groups, err := grouped(keys)
	if err != nil {
		return err
	}
	now := c.now().Unix()
	for g, hashes := range groups {
		if err := c.touch(ctx, index, g, hashes, now, nil); err != nil {
			return err
		}
	}
	return nil
}

// Put implements tagging.EmbeddingCache. It stores every vector in one
// transaction, so a failure leaves none of them, upserting them in the order of
// the primary key (see keyOrder), so concurrent Puts of the same keys do not
// deadlock.
func (c *Cache) Put(ctx context.Context, index string, vecs map[tagging.CacheKey][]float32) error {
	if err := checkIndex(index); err != nil {
		return err
	}
	for k, v := range vecs {
		if err := checkName("model", k.Model); err != nil {
			return err
		}
		if len(v) == 0 {
			return errEmptyVector
		}
	}
	if len(vecs) == 0 {
		return nil
	}
	now := c.now().Unix()
	tx, err := c.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	stmt, err := tx.PrepareContext(ctx, c.q(`INSERT INTO `+table+` (idx, model, role, hash, vec, used) VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT (idx, model, role, hash) DO UPDATE SET vec = excluded.vec, used = excluded.used`))
	if err != nil {
		return err
	}
	defer func() { _ = stmt.Close() }()
	for _, k := range keyOrder(vecs) {
		if _, err := stmt.ExecContext(ctx, index, k.Model, int64(k.Role), k.Hash[:], encode(vecs[k]), now); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// keyOrder returns the keys of vecs in the order of the primary key within one
// index: model, role, hash. Two Puts of overlapping keys then lock their rows in
// the same order, so the second waits for the first instead of deadlocking
// with it, as they do in Go's random map order on Postgres.
func keyOrder(vecs map[tagging.CacheKey][]float32) []tagging.CacheKey {
	return slices.SortedFunc(maps.Keys(vecs), func(a, b tagging.CacheKey) int {
		return cmp.Or(strings.Compare(a.Model, b.Model), cmp.Compare(a.Role, b.Role), bytes.Compare(a.Hash[:], b.Hash[:]))
	})
}

// DeleteIndex implements tagging.EmbeddingCache.
func (c *Cache) DeleteIndex(ctx context.Context, index string) error {
	if err := checkIndex(index); err != nil {
		return err
	}
	_, err := c.db.ExecContext(ctx, c.q(`DELETE FROM `+table+` WHERE idx = ?`), index)
	return err
}

// Prune implements tagging.EmbeddingCache.
func (c *Cache) Prune(ctx context.Context, unusedSince time.Time) (int64, error) {
	res, err := c.db.ExecContext(ctx, c.q(`DELETE FROM `+table+` WHERE used < ?`), unusedSince.Unix())
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

// placeholders returns n comma-separated ? placeholders.
func placeholders(n int) string {
	return strings.TrimSuffix(strings.Repeat("?,", n), ",")
}

// bytesArgs converts hashes to statement arguments.
func bytesArgs(hashes [][]byte) []any {
	args := make([]any, len(hashes))
	for i, h := range hashes {
		args[i] = h
	}
	return args
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

// encode returns the little-endian bytes of the float32s of v.
func encode(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

// decode is encode's inverse; a blob that does not hold a whole number of
// float32s was cut short.
func decode(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("vector of %d bytes is not a whole number of float32s", len(b))
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, nil
}

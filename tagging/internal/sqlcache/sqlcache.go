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

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/tagging"
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

// Dialect holds what differs between the SQL databases a Cache runs on. The
// statements are written once, with ? placeholders, and are otherwise plain SQL
// that these databases share.
type Dialect struct {
	Blob, Int string // vector/hash and role/time column types
	Numbered  bool   // $1, $2… placeholders instead of ?
}

// SQLite is the Dialect of SQLite.
var SQLite = Dialect{Blob: "BLOB", Int: "INTEGER"}

// Postgres is the Dialect of PostgreSQL.
var Postgres = Dialect{Blob: "BYTEA", Int: "BIGINT", Numbered: true}

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
	for _, st := range []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			idx TEXT NOT NULL, model TEXT NOT NULL, role %[2]s NOT NULL,
			hash %[3]s NOT NULL, vec %[3]s NOT NULL, used %[2]s NOT NULL,
			PRIMARY KEY (idx, model, role, hash))`, table, d.Int, d.Blob),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_used ON %s (used)`, table, table),
	} {
		if _, err := db.ExecContext(ctx, st); err != nil {
			return nil, fmt.Errorf("create autotag embeddings schema: %w", err)
		}
	}
	return c, nil
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
	role  langai.Role
}

// Get implements tagging.EmbeddingCache. Keys are grouped by model and role and
// matched 500 at a time; the entries found older than
// tagging.CacheRefreshAfter get their last-used time rewritten.
func (c *Cache) Get(ctx context.Context, index string, keys []tagging.CacheKey) (map[tagging.CacheKey][]float32, error) {
	if index == "" {
		return nil, errEmptyIndex
	}
	out := make(map[tagging.CacheKey][]float32, len(keys))
	if len(keys) == 0 {
		return out, nil
	}
	groups := map[group][][]byte{}
	seen := map[tagging.CacheKey]bool{}
	for _, k := range keys {
		if seen[k] {
			continue
		}
		seen[k] = true
		g := group{k.Model, k.Role}
		groups[g] = append(groups[g], k.Hash[:])
	}
	now := c.now().Unix()
	staleBefore := now - int64(tagging.CacheRefreshAfter/time.Second)
	for g, hashes := range groups {
		for start := 0; start < len(hashes); start += chunk {
			part := hashes[start:min(start+chunk, len(hashes))]
			stale, err := c.read(ctx, index, g, part, staleBefore, out)
			if err != nil {
				return nil, err
			}
			if err := c.touch(ctx, index, g, stale, now, staleBefore); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// read adds to out the entries of hashes that the cache holds and returns the
// hashes of those last used before staleBefore.
func (c *Cache) read(ctx context.Context, index string, g group, hashes [][]byte, staleBefore int64, out map[tagging.CacheKey][]float32) ([][]byte, error) {
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
		if used < staleBefore {
			stale = append(stale, hash)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return stale, nil
}

// touch sets the last-used time of the stale entries to now. The used < bound
// guard keeps two readers from rewriting a row one after the other for
// nothing.
func (c *Cache) touch(ctx context.Context, index string, g group, stale [][]byte, now, staleBefore int64) error {
	for start := 0; start < len(stale); start += chunk {
		part := stale[start:min(start+chunk, len(stale))]
		args := append([]any{now, index, g.model, int64(g.role), staleBefore}, bytesArgs(part)...)
		if _, err := c.db.ExecContext(ctx, c.q(`UPDATE `+table+` SET used = ? WHERE idx = ? AND model = ? AND role = ? AND used < ? AND hash IN (`+placeholders(len(part))+`)`), args...); err != nil {
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
	if index == "" {
		return errEmptyIndex
	}
	for _, v := range vecs {
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
	if index == "" {
		return errEmptyIndex
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

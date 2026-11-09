// Package sqlstore implements vector.Store on database/sql for the databases
// a Dialect describes. The public backends, vector/sqlitevec and vector/pgvec,
// are this package with their own Dialect and their own check of the
// database they get.
//
// Every index has a table set of its own, named by internal/tablename: docs
// (one row per document and its content hash), chunks (one row per chunk,
// its text and its vector) and meta (the dimensions of the vectors and the
// model of each collection). Only writes create the tables; a read of an
// index without them finds nothing. Searches are exact: the chunks of the
// scope are ranked by cosine distance and the best one of each document
// kept, with a window function.
package sqlstore

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/go-bumbu/langai/internal/tablename"
	"github.com/go-bumbu/langai/vector"
)

// prefix is what every table name of the stores starts with (see
// tablename.For).
const prefix = "ss"

// modelKey is the meta name the model of a collection is kept under, followed
// by the collection.
const modelKey = "model/"

// Dialect holds what differs between the SQL databases a Store runs on. The
// statements are written once, with ? placeholders, and are otherwise plain
// SQL that these databases share.
type Dialect struct {
	// Name is the database's name in error messages.
	Name string
	// Numbered makes the placeholders $1, $2… instead of ?.
	Numbered bool
	// Schema returns the statements that create the docs, chunks and meta
	// tables of an index, given their names, for vectors of dims dimensions.
	// The meta table comes last: its existence says the others exist.
	Schema func(docs, chunks, meta string, dims int) []string
	// LockDDL, when not empty, runs the schema statements in one transaction
	// that first takes this lock, bound to the meta table's name, so that two
	// processes creating an index's tables wait for each other. Without it the
	// statements run one by one.
	LockDDL string
	// Legacy is the suffix of a table that only the schema before the meta
	// table had. An index that has it and no meta table is refused with an
	// error that says how to migrate.
	Legacy string
	// TableExists is a query that answers one row with one boolean: whether
	// the table named by its one parameter exists.
	TableExists string
	// ListTables is a query that answers the names of the tables whose name
	// starts with "ss_".
	ListTables string
	// Distance is the expression of the cosine distance between the embedding
	// column and the query vector bound to its one placeholder.
	Distance string
	// IDs is the subquery of the ids in the JSON array bound to its one
	// placeholder.
	IDs string
	// Encode turns a vector into the argument the embedding column is bound
	// to, and Decode turns what the column scans into back into a vector.
	Encode func([]float32) any
	Decode func(src any) ([]float32, error)
}

// Store implements vector.Store on SQL. It is safe for concurrent use as far
// as its *sql.DB is.
type Store struct {
	db         *sql.DB
	d          Dialect
	dims       int
	generation string

	mu    sync.Mutex
	ready map[string]bool // indexes whose tables exist and have dims
}

var _ vector.Store = (*Store)(nil)

// New returns a Store over db for vectors of dims dimensions, keeping its
// tables in generation (see tablename.For). It does not look at the
// database, which its caller knows how to check.
func New(db *sql.DB, d Dialect, dims int, generation string) (*Store, error) {
	if db == nil {
		return nil, errors.New("nil database handle")
	}
	if dims <= 0 {
		return nil, fmt.Errorf("invalid vector dimensions %d", dims)
	}
	if err := tablename.CheckGeneration(generation); err != nil {
		return nil, err
	}
	return &Store{db: db, d: d, dims: dims, generation: generation, ready: map[string]bool{}}, nil
}

// tables are the names of one index's tables.
type tables struct{ docs, chunks, meta, legacy string }

func (s *Store) tables(index string) (tables, error) {
	var t tables
	var err error
	for _, n := range []struct {
		dst    *string
		suffix string
	}{{&t.docs, "docs"}, {&t.chunks, "chunks"}, {&t.meta, "meta"}, {&t.legacy, s.d.Legacy}} {
		if *n.dst, err = tablename.For(s.generation, index, n.suffix); err != nil {
			return tables{}, err
		}
	}
	return t, nil
}

// q numbers the ? placeholders of a statement where the Dialect wants it. No
// statement holds a literal ?.
func (s *Store) q(query string) string {
	if !s.d.Numbered {
		return query
	}
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

// open returns the tables of index and whether they exist, creating them
// when create is set. Either way it checks that they hold vectors of the
// store's dimensions and are not of the schema before the meta table. Only a
// positive answer is remembered, so a reader notices the tables a writer
// creates later.
func (s *Store) open(ctx context.Context, index string, create bool) (tables, bool, error) {
	t, err := s.tables(index)
	if err != nil {
		return tables{}, false, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ready[index] {
		return t, true, nil
	}
	exists, err := s.exists(ctx, t.meta)
	if err != nil {
		return tables{}, false, err
	}
	if !exists {
		legacy, err := s.exists(ctx, t.legacy)
		if err != nil {
			return tables{}, false, err
		}
		if legacy {
			return tables{}, false, fmt.Errorf("vector: index %q has %s tables of the schema before this version: delete them (DeleteIndex) or open the store WithGeneration and index again", index, s.d.Name)
		}
		if !create {
			return t, false, nil
		}
		if err := s.create(ctx, t); err != nil {
			return tables{}, false, fmt.Errorf("create %s schema for index %q: %w", s.d.Name, index, err)
		}
	}
	known, err := s.checkDims(ctx, index, t.meta)
	if err != nil {
		return tables{}, false, err
	}
	s.ready[index] = known
	return t, true, nil
}

func (s *Store) exists(ctx context.Context, table string) (bool, error) {
	var ok bool
	if err := s.db.QueryRowContext(ctx, s.q(s.d.TableExists), table).Scan(&ok); err != nil {
		return false, fmt.Errorf("look up table: %w", err)
	}
	return ok, nil
}

// create runs the Dialect's schema statements and records the store's
// dimensions, in one locked transaction when the Dialect has a lock.
func (s *Store) create(ctx context.Context, t tables) error {
	stmts := s.d.Schema(t.docs, t.chunks, t.meta, s.dims)
	dims := s.q(`INSERT INTO ` + t.meta + ` (name, value) VALUES ('dims', ?) ON CONFLICT (name) DO NOTHING`)
	if s.d.LockDDL == "" {
		for _, st := range stmts {
			if _, err := s.db.ExecContext(ctx, st); err != nil {
				return err
			}
		}
		_, err := s.db.ExecContext(ctx, dims, strconv.Itoa(s.dims))
		return err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx, s.q(s.d.LockDDL), t.meta); err != nil {
		return err
	}
	for _, st := range stmts {
		if _, err := tx.ExecContext(ctx, st); err != nil {
			return err
		}
	}
	if _, err := tx.ExecContext(ctx, dims, strconv.Itoa(s.dims)); err != nil {
		return err
	}
	return tx.Commit()
}

// checkDims fails with vector.ErrDimensions when the index's tables were
// built for vectors of other dimensions than the store's, and reports false
// when they record none yet: another process is creating them.
func (s *Store) checkDims(ctx context.Context, index, meta string) (bool, error) {
	var v string
	err := s.db.QueryRowContext(ctx, s.q(`SELECT value FROM `+meta+` WHERE name = 'dims'`)).Scan(&v)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if v != strconv.Itoa(s.dims) {
		return false, fmt.Errorf("%w: index %q holds vectors of %s dimensions and the store was opened for %d; open a store WithGeneration to build the new model's vectors next to them", vector.ErrDimensions, index, v, s.dims)
	}
	return true, nil
}

// Upsert implements vector.Store. Every document is written in one
// transaction, in the order of their ids so that two concurrent writes of
// the same documents queue instead of deadlocking, and each starts with its
// docs row: that row's lock makes a second writer of the same document wait
// until the first is done, so the two never leave two sets of chunks. The
// first statement of the transaction is a write, so SQLite takes its write
// lock as the transaction begins and a busy timeout covers the wait for it.
func (s *Store) Upsert(ctx context.Context, c vector.Coord, model string, docs ...vector.Doc) error {
	if err := vector.CheckWrite(c, s.dims, docs); err != nil {
		return err
	}
	if len(docs) == 0 {
		return nil
	}
	t, _, err := s.open(ctx, c.Index, true)
	if err != nil {
		return err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := s.claimModel(ctx, tx, t.meta, c.Collection, model); err != nil {
		return err
	}
	insert, err := tx.PrepareContext(ctx, s.q(`INSERT INTO `+t.chunks+` (partition, collection, doc_id, seq, text, embedding) VALUES (?, ?, ?, ?, ?, ?)`))
	if err != nil {
		return err
	}
	defer func() { _ = insert.Close() }()
	sorted := slices.SortedFunc(slices.Values(docs), func(a, b vector.Doc) int { return strings.Compare(a.ID, b.ID) })
	for _, d := range sorted {
		if _, err := tx.ExecContext(ctx, s.q(`INSERT INTO `+t.docs+` (partition, collection, doc_id, content_hash) VALUES (?, ?, ?, ?)
			ON CONFLICT (partition, collection, doc_id) DO UPDATE SET content_hash = excluded.content_hash`),
			c.Partition, c.Collection, d.ID, d.Hash); err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, s.q(`DELETE FROM `+t.chunks+` WHERE partition = ? AND collection = ? AND doc_id = ?`),
			c.Partition, c.Collection, d.ID); err != nil {
			return err
		}
		for _, ch := range d.Chunks {
			if _, err := insert.ExecContext(ctx, c.Partition, c.Collection, d.ID, ch.Seq, ch.Text, s.d.Encode(ch.Vector)); err != nil {
				return err
			}
		}
	}
	return tx.Commit()
}

// claimModel records model as the collection's when it has none yet and
// fails with vector.ErrModelMismatch when it has another. An empty model
// checks nothing. Its first statement is a write (see Upsert); without a
// model the docs upsert after it is.
func (s *Store) claimModel(ctx context.Context, tx *sql.Tx, meta, collection, model string) error {
	if model == "" {
		return nil
	}
	key := modelKey + collection
	if _, err := tx.ExecContext(ctx, s.q(`INSERT INTO `+meta+` (name, value) VALUES (?, ?) ON CONFLICT (name) DO NOTHING`), key, model); err != nil {
		return err
	}
	var have string
	if err := tx.QueryRowContext(ctx, s.q(`SELECT value FROM `+meta+` WHERE name = ?`), key).Scan(&have); err != nil {
		return err
	}
	if have != model {
		return fmt.Errorf("%w: collection %q holds %q, not %q", vector.ErrModelMismatch, collection, have, model)
	}
	return nil
}

// Delete implements vector.Store, removing the docs rows first, as Upsert
// writes them first.
func (s *Store) Delete(ctx context.Context, c vector.Coord, ids ...string) error {
	t, ok, err := s.open(ctx, c.Index, false)
	if err != nil || !ok || len(ids) == 0 {
		return err
	}
	list, err := jsonIDs(ids)
	if err != nil {
		return err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	for _, table := range []string{t.docs, t.chunks} {
		if _, err := tx.ExecContext(ctx, s.q(`DELETE FROM `+table+` WHERE partition = ? AND collection = ? AND doc_id IN (`+s.d.IDs+`)`),
			c.Partition, c.Collection, list); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// DeleteIndex implements vector.Store by dropping the index's tables, those
// of the schema before the meta table included.
func (s *Store) DeleteIndex(ctx context.Context, index string) error {
	t, err := s.tables(index)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.ready, index)
	for _, table := range []string{t.meta, t.docs, t.chunks, t.legacy} {
		if _, err := s.db.ExecContext(ctx, `DROP TABLE IF EXISTS `+table); err != nil {
			return fmt.Errorf("delete index %q: %w", index, err)
		}
	}
	return nil
}

// Hashes implements vector.Store.
func (s *Store) Hashes(ctx context.Context, c vector.Coord, ids ...string) (map[string]string, error) {
	t, ok, err := s.open(ctx, c.Index, false)
	if err != nil {
		return nil, err
	}
	out := make(map[string]string, len(ids))
	if !ok || len(ids) == 0 {
		return out, nil
	}
	list, err := jsonIDs(ids)
	if err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT doc_id, content_hash FROM `+t.docs+` WHERE partition = ? AND collection = ? AND doc_id IN (`+s.d.IDs+`)`),
		c.Partition, c.Collection, list)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var id, hash string
		if err := rows.Scan(&id, &hash); err != nil {
			return nil, err
		}
		out[id] = hash
	}
	return out, rows.Err()
}

// Docs implements vector.Store.
func (s *Store) Docs(ctx context.Context, c vector.Coord, after string, limit int) ([]vector.DocInfo, error) {
	t, ok, err := s.open(ctx, c.Index, false)
	if err != nil || !ok {
		return nil, err
	}
	query := `SELECT doc_id, content_hash FROM ` + t.docs + ` WHERE partition = ? AND collection = ? AND doc_id > ? ORDER BY doc_id`
	args := []any{c.Partition, c.Collection, after}
	if limit > 0 {
		query += ` LIMIT ?`
		args = append(args, limit)
	}
	rows, err := s.db.QueryContext(ctx, s.q(query), args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []vector.DocInfo
	for rows.Next() {
		var d vector.DocInfo
		if err := rows.Scan(&d.ID, &d.Hash); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// Chunks implements vector.Store.
func (s *Store) Chunks(ctx context.Context, c vector.Coord, id string) ([]vector.Chunk, error) {
	t, ok, err := s.open(ctx, c.Index, false)
	if err != nil || !ok {
		return nil, err
	}
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT seq, text, embedding FROM `+t.chunks+`
		WHERE partition = ? AND collection = ? AND doc_id = ? ORDER BY seq, id`), c.Partition, c.Collection, id)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []vector.Chunk
	for rows.Next() {
		var ch vector.Chunk
		var src any
		if err := rows.Scan(&ch.Seq, &ch.Text, &src); err != nil {
			return nil, err
		}
		if ch.Vector, err = s.d.Decode(src); err != nil {
			return nil, fmt.Errorf("chunk %d: %w", ch.Seq, err)
		}
		out = append(out, ch)
	}
	return out, rows.Err()
}

// Search implements vector.Store with one exact query: every chunk in scope
// is ranked by its distance to the query vector, the best of each document
// (the lowest Seq on a tie) is kept, and the q.Limit best of those are
// returned. The cosine distance (0 = identical, 2 = opposite) is reported as
// 1-distance, so that higher is better.
func (s *Store) Search(ctx context.Context, c vector.Coord, q vector.Query) ([]vector.Match, error) {
	if err := vector.CheckSearch(c, s.dims, q); err != nil {
		return nil, err
	}
	t, ok, err := s.open(ctx, c.Index, false)
	if err != nil || !ok || q.Limit <= 0 {
		return nil, err
	}
	if err := s.checkModel(ctx, t.meta, c.Collection, q); err != nil {
		return nil, err
	}
	where, args, err := s.scope(c, q)
	if err != nil {
		return nil, err
	}
	args = append([]any{s.d.Encode(q.Vector)}, args...)
	args = append(args, q.Limit)
	query := `SELECT c.partition, c.collection, c.doc_id, c.seq, c.text, b.d FROM (
			SELECT id, partition, collection, doc_id, seq, d FROM (
				SELECT id, partition, collection, doc_id, seq, d,
					row_number() OVER (PARTITION BY partition, collection, doc_id ORDER BY d, seq) AS rn
				FROM (SELECT id, partition, collection, doc_id, seq, ` + s.d.Distance + ` AS d FROM ` + t.chunks + where + `) scored
			) ranked WHERE rn = 1 ORDER BY d, doc_id, seq, partition, collection LIMIT ?
		) b JOIN ` + t.chunks + ` c ON c.id = b.id
		ORDER BY b.d, b.doc_id, b.seq, b.partition, b.collection`
	rows, err := s.db.QueryContext(ctx, s.q(query), args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []vector.Match
	for rows.Next() {
		m := vector.Match{Coord: vector.Coord{Index: c.Index}}
		var dist float64
		if err := rows.Scan(&m.Coord.Partition, &m.Coord.Collection, &m.DocID, &m.Seq, &m.Text, &dist); err != nil {
			return nil, err
		}
		m.Score = 1 - dist
		out = append(out, m)
	}
	return out, rows.Err()
}

// scope returns the WHERE clause of q's scope within c's index and its
// arguments: the coordinates q does not open and the id lists. Only constant
// SQL goes into the clause.
func (s *Store) scope(c vector.Coord, q vector.Query) (string, []any, error) {
	var conds []string
	var args []any
	if !q.AllPartitions {
		conds = append(conds, "partition = ?")
		args = append(args, c.Partition)
	}
	if !q.AllCollections {
		conds = append(conds, "collection = ?")
		args = append(args, c.Collection)
	}
	for _, l := range []struct {
		ids []string
		op  string
	}{{q.Only, "IN"}, {q.Exclude, "NOT IN"}} {
		if len(l.ids) == 0 {
			continue
		}
		list, err := jsonIDs(l.ids)
		if err != nil {
			return "", nil, err
		}
		conds = append(conds, "doc_id "+l.op+" ("+s.d.IDs+")")
		args = append(args, list)
	}
	if len(conds) == 0 {
		return "", args, nil
	}
	return " WHERE " + strings.Join(conds, " AND "), args, nil
}

// checkModel fails with vector.ErrModelMismatch when q names a model and a
// collection in its scope recorded another.
func (s *Store) checkModel(ctx context.Context, meta, collection string, q vector.Query) error {
	if q.Model == "" {
		return nil
	}
	query := `SELECT name, value FROM ` + meta + ` WHERE name = ? AND value <> ?`
	args := []any{modelKey + collection, q.Model}
	if q.AllCollections {
		query = `SELECT name, value FROM ` + meta + ` WHERE substr(name, 1, ?) = ? AND value <> ? ORDER BY name LIMIT 1`
		args = []any{len(modelKey), modelKey, q.Model}
	}
	var name, have string
	err := s.db.QueryRowContext(ctx, s.q(query), args...).Scan(&name, &have)
	if errors.Is(err, sql.ErrNoRows) {
		return nil
	}
	if err != nil {
		return err
	}
	return fmt.Errorf("%w: collection %q holds %q, not %q", vector.ErrModelMismatch, strings.TrimPrefix(name, modelKey), have, q.Model)
}

// jsonIDs encodes ids as the JSON array the Dialect's IDs subquery reads.
func jsonIDs(ids []string) (string, error) {
	b, err := json.Marshal(ids)
	return string(b), err
}

// DropGeneration drops every table of generation, all its indexes', so that
// a migration can retire the generation it moved away from. It refuses the
// store's own generation, whose tables it is using.
func (s *Store) DropGeneration(ctx context.Context, generation string) error {
	if err := tablename.CheckGeneration(generation); err != nil {
		return err
	}
	if generation == s.generation {
		return fmt.Errorf("generation %q is the store's own: drop it from a store of another generation", generation)
	}
	ours := regexp.MustCompile(`^` + prefix + `_[0-9a-f]{16}_[a-z0-9_]+$`)
	if generation != "" {
		ours = regexp.MustCompile(`^` + prefix + `_` + generation + `_[0-9a-f]{16}_[a-z0-9_]+$`)
	}
	rows, err := s.db.QueryContext(ctx, s.d.ListTables)
	if err != nil {
		return fmt.Errorf("list tables: %w", err)
	}
	var names []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			_ = rows.Close()
			return err
		}
		if ours.MatchString(name) {
			names = append(names, name)
		}
	}
	_ = rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	// a virtual table sorts ahead of its shadow tables, which dropping it drops
	slices.Sort(names)
	for _, name := range names {
		if _, err := s.db.ExecContext(ctx, `DROP TABLE IF EXISTS `+name); err != nil {
			return fmt.Errorf("drop %s: %w", name, err)
		}
	}
	return nil
}

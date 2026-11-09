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
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/go-bumbu/sense/internal/tablename"
	"github.com/go-bumbu/sense/vector"
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
	// The tables of an index are created and dropped in one transaction, and
	// LockDDL, when not empty, is its first statement: a lock, bound to the
	// meta table's name, that makes two processes creating or dropping an
	// index's tables wait for each other. BeginDDL, when not empty, starts
	// that transaction on a connection of its own instead of BeginTx (SQLite's
	// BEGIN IMMEDIATE, which takes the write lock at once, so the busy timeout
	// covers the wait for it and no other writer can slip in between).
	LockDDL  string
	BeginDDL string
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
	// Distance returns the expression of the cosine distance between two
	// vector expressions: a stored embedding and a query vector.
	Distance func(embedding, query string) string
	// Queries returns the WITH clause that defines the table q(i, v), which
	// turns the one argument EncodeQueries makes of the query vectors into
	// one row per vector: its index i, from 0, and the vector v. The clause
	// binds it to its one placeholder, so a search of any number of vectors
	// takes one parameter, and q must be computed once per statement, not
	// once per chunk. dims is the store's.
	Queries       func(dims int) string
	EncodeQueries func([][]float32) any
	// IDs is the subquery of the ids in the JSON array bound to its one
	// placeholder.
	IDs string
	// Encode turns a vector into the argument the embedding column is bound
	// to, and Decode turns what the column scans into back into a vector.
	Encode func([]float32) any
	Decode func(src any) ([]float32, error)
	// MissingTable reports whether err says a statement named a table that
	// does not exist: another Store, in this process or another, dropped the
	// index after this one found its tables.
	MissingTable func(err error) bool
	// LockRows, when not empty, is the clause that makes a SELECT lock the
	// rows it reads (FOR UPDATE): a delete takes the docs rows' locks with it
	// first, in id order, the order Upsert takes them, so the two never
	// deadlock whatever order the delete's plan visits the rows in. A
	// database that runs one writer at a time (SQLite) needs none.
	// ShareRows is the clause of a shared row lock (FOR SHARE), which a
	// writer takes on its collection's model record (see claimModel).
	LockRows  string
	ShareRows string
	// Retryable, when set, reports whether err is a transient failure of a
	// write transaction (a deadlock, a serialization failure) that is worth
	// running the transaction again for.
	Retryable func(err error) bool
}

// maxAttempts is the most times a write transaction runs while it fails
// with an error the Dialect calls Retryable.
const maxAttempts = 3

// Store implements vector.Store on SQL. It is safe for concurrent use as far
// as its *sql.DB is.
type Store struct {
	db         *sql.DB
	d          Dialect
	dims       int
	generation string

	// mu guards ready and opening and is never held across a database call,
	// so a slow open of one index stalls no other
	mu      sync.Mutex
	ready   map[string]bool          // indexes whose tables exist and have dims
	opening map[string]chan struct{} // indexes being opened; closed when done
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
	return &Store{
		db: db, d: d, dims: dims, generation: generation,
		ready: map[string]bool{}, opening: map[string]chan struct{}{},
	}, nil
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

// withTables runs fn on the tables of index, opening them first (see open)
// and creating them when create is set; when the index has no tables (and
// create is not set) it runs nothing and reports false. When fn finds a
// table missing, because another Store dropped the index after this one
// opened it, the index is forgotten and opened once more, so a read then
// finds nothing and a write creates the tables again; fn must start over
// from scratch on that second run.
func (s *Store) withTables(ctx context.Context, index string, create bool, fn func(tables) error) (bool, error) {
	for retried := false; ; retried = true {
		t, ok, err := s.open(ctx, index, create)
		if err == nil && !ok {
			return false, nil
		}
		if err == nil {
			err = fn(t)
		}
		if retried || err == nil || !s.d.MissingTable(err) {
			return ok, err
		}
		s.forget(index)
	}
}

// forget drops what the store remembers of index's tables, so the next call
// looks them up again.
func (s *Store) forget(index string) {
	s.mu.Lock()
	delete(s.ready, index)
	s.mu.Unlock()
}

// open returns the tables of index and whether they exist, creating them
// when create is set. Either way it checks that they hold vectors of the
// store's dimensions and are not of the schema before the meta table. Only a
// positive answer is remembered, so a reader notices the tables a writer
// creates later.
//
// The answer for a remembered index costs no database call. Otherwise one
// caller per index looks it up while the others for that index wait for it
// (or for their context), and callers for other indexes go on: the store's
// mutex is never held across a database call, so the first write of a new
// index waiting for SQLite's write lock stalls nobody else.
func (s *Store) open(ctx context.Context, index string, create bool) (tables, bool, error) {
	t, err := s.tables(index)
	if err != nil {
		return tables{}, false, err
	}
	for {
		s.mu.Lock()
		if s.ready[index] {
			s.mu.Unlock()
			return t, true, nil
		}
		if busy, ok := s.opening[index]; ok {
			s.mu.Unlock()
			select {
			case <-busy:
				continue // look again: it is ready now, or this caller tries itself
			case <-ctx.Done():
				return tables{}, false, ctx.Err()
			}
		}
		done := make(chan struct{})
		s.opening[index] = done
		s.mu.Unlock()

		ok, known, err := s.probe(ctx, t, index, create)

		s.mu.Lock()
		delete(s.opening, index)
		if known {
			s.ready[index] = true
		}
		s.mu.Unlock()
		close(done)
		if err != nil {
			return tables{}, false, err
		}
		return t, ok, nil
	}
}

// probe looks index's tables up for open, creating them when create is set:
// whether they exist, and whether they record the store's dimensions, which
// they do not yet while another process is creating them.
//
// The tables appear and disappear together, in one transaction, but the
// lookups of the meta table and of the legacy marker are two statements
// (and on Postgres the marker, docs, is a table of this schema too): a
// creation that commits between them reads as the schema before this one.
// So that answer is checked again under the DDL lock, which every creation
// and drop holds until it commits.
func (s *Store) probe(ctx context.Context, t tables, index string, create bool) (ok, known bool, err error) {
	exists, err := s.exists(ctx, s.db, t.meta)
	if err != nil {
		return false, false, err
	}
	if !exists {
		if exists, err = s.metaUnlessLegacy(ctx, t, index); err != nil {
			return false, false, err
		}
	}
	if !exists {
		if !create {
			return false, false, nil
		}
		if err := s.create(ctx, t); err != nil {
			return false, false, fmt.Errorf("create schema for index %q: %w", index, err)
		}
	}
	known, err = s.checkDims(ctx, index, t.meta)
	if err != nil {
		return false, false, err
	}
	return true, known, nil
}

// metaUnlessLegacy is probe's answer for an index whose meta table it did
// not find: false when the legacy marker is missing too, the error naming
// the old schema when the marker is there and, under the DDL lock, the meta
// table still is not, and true when the meta table turned up meanwhile.
func (s *Store) metaUnlessLegacy(ctx context.Context, t tables, index string) (bool, error) {
	legacy, err := s.exists(ctx, s.db, t.legacy)
	if err != nil || !legacy {
		return false, err
	}
	var exists bool
	err = s.ddl(ctx, t, func(ex execer) error {
		var err error
		exists, err = s.exists(ctx, ex, t.meta)
		return err
	})
	if err != nil {
		return false, err
	}
	if !exists {
		return false, ownf("%w: index %q (%s): delete them (DeleteIndex) or open the store WithGeneration and index again", vector.ErrLegacySchema, index, s.d.Name)
	}
	return true, nil
}

// execer is what the statements of a DDL transaction run on: a *sql.Tx, or
// the *sql.Conn of a transaction begun with Dialect.BeginDDL.
type execer interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

func (s *Store) exists(ctx context.Context, ex execer, table string) (bool, error) {
	var ok bool
	if err := ex.QueryRowContext(ctx, s.q(s.d.TableExists), table).Scan(&ok); err != nil {
		return false, fmt.Errorf("look up table: %w", err)
	}
	return ok, nil
}

// ddl runs fn in one transaction that creates or drops t, holding the
// Dialect's DDL lock (or SQLite's write lock) from its start.
func (s *Store) ddl(ctx context.Context, t tables, fn func(execer) error) error {
	if s.d.BeginDDL != "" {
		return s.ddlOnConn(ctx, fn)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if s.d.LockDDL != "" {
		if _, err := tx.ExecContext(ctx, s.q(s.d.LockDDL), t.meta); err != nil {
			return err
		}
	}
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit()
}

// ddlOnConn is ddl for a Dialect with BeginDDL: the transaction runs on a
// connection of its own, begun with that statement, and a connection that
// cannot be rolled back is discarded rather than handed back to the pool
// inside a transaction.
func (s *Store) ddlOnConn(ctx context.Context, fn func(execer) error) error {
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close() }()
	if _, err := conn.ExecContext(ctx, s.d.BeginDDL); err != nil {
		return err
	}
	if err = fn(conn); err == nil {
		if _, err = conn.ExecContext(ctx, "COMMIT"); err == nil {
			return nil
		}
	}
	if _, rbErr := conn.ExecContext(context.WithoutCancel(ctx), "ROLLBACK"); rbErr != nil {
		_ = conn.Raw(func(any) error { return driver.ErrBadConn })
	}
	return err
}

// create runs the Dialect's schema statements and records the store's
// dimensions, in one DDL transaction.
func (s *Store) create(ctx context.Context, t tables) error {
	return s.ddl(ctx, t, func(ex execer) error {
		for _, st := range s.d.Schema(t.docs, t.chunks, t.meta, s.dims) {
			if _, err := ex.ExecContext(ctx, st); err != nil {
				return err
			}
		}
		_, err := ex.ExecContext(ctx, s.q(`INSERT INTO `+t.meta+` (name, value) VALUES ('dims', ?) ON CONFLICT (name) DO NOTHING`), strconv.Itoa(s.dims))
		return err
	})
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
	if err := vector.CheckWrite(c, model, s.dims, docs); err != nil {
		return err
	}
	if len(docs) == 0 {
		return nil
	}
	return s.wrap("upsert", s.upsert(ctx, c, model, docs))
}

func (s *Store) upsert(ctx context.Context, c vector.Coord, model string, docs []vector.Doc) error {
	_, err := s.withTables(ctx, c.Index, true, func(t tables) error {
		return s.retry(ctx, func() error { return s.writeDocs(ctx, t, c, model, docs) })
	})
	return err
}

// errContended reports a claim of a collection's model that found the
// record taken over while it looked (see claimModel): its transaction runs
// again.
var errContended = ownf("vector: the collection's model changed while it was being checked; try again")

// retry runs the write transaction fn, and again, up to maxAttempts runs in
// all, while it fails with an error the Dialect calls Retryable or with
// errContended.
func (s *Store) retry(ctx context.Context, fn func() error) error {
	for attempt := 1; ; attempt++ {
		err := fn()
		again := errors.Is(err, errContended) || (s.d.Retryable != nil && s.d.Retryable(err))
		if err == nil || !again || attempt == maxAttempts || ctx.Err() != nil {
			return err
		}
	}
}

// writeDocs is Upsert's transaction.
func (s *Store) writeDocs(ctx context.Context, t tables, c vector.Coord, model string, docs []vector.Doc) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := s.claimModel(ctx, tx, t, c.Collection, model); err != nil {
		return err
	}
	// the chunks insert is prepared after the first docs row is written, so
	// that the transaction takes its locks in the order every other one
	// does: meta, docs, chunks
	var insert *sql.Stmt
	defer func() {
		if insert != nil {
			_ = insert.Close()
		}
	}()
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
		if insert == nil {
			if insert, err = tx.PrepareContext(ctx, s.q(`INSERT INTO `+t.chunks+` (partition, collection, doc_id, seq, text, embedding) VALUES (?, ?, ?, ?, ?, ?)`)); err != nil {
				return err
			}
		}
		for _, ch := range d.Chunks {
			if _, err := insert.ExecContext(ctx, c.Partition, c.Collection, d.ID, ch.Seq, ch.Text, s.d.Encode(ch.Vector)); err != nil {
				return err
			}
		}
	}
	return tx.Commit()
}

// claimModel records model as the collection's when it has none yet, or
// another while it holds no documents, and fails with
// vector.ErrModelMismatch when another is recorded and the collection still
// holds documents. An empty model checks nothing. Its first statement is a
// write (see Upsert); without a model the docs upsert after it is.
//
// Where the Dialect locks rows, a writer of the recorded model holds a
// shared lock on the record until it commits, and taking the record over
// needs an exclusive one, after which the collection is looked at again: so
// no writer of the old model can add documents between the look and the
// takeover, and none can take it over while another writes.
func (s *Store) claimModel(ctx context.Context, tx *sql.Tx, t tables, collection, model string) error {
	if model == "" {
		return nil
	}
	key := modelKey + collection
	if _, err := tx.ExecContext(ctx, s.q(`INSERT INTO `+t.meta+` (name, value) VALUES (?, ?) ON CONFLICT (name) DO NOTHING`), key, model); err != nil {
		return err
	}
	read := func(lock string) (string, error) {
		var have string
		err := tx.QueryRowContext(ctx, s.q(`SELECT value FROM `+t.meta+` WHERE name = ? `+lock), key).Scan(&have)
		return have, err
	}
	have, err := read("")
	if err != nil {
		return err
	}
	if s.d.LockRows != "" {
		// a writer of the recorded model shares the record; a takeover locks
		// it alone from the start, never upgrading a shared lock it holds,
		// so two takeovers queue rather than deadlock
		lock := s.d.LockRows
		if have == model {
			lock = s.d.ShareRows
		}
		if have, err = read(lock); err != nil {
			return err
		}
		if have != model && lock == s.d.ShareRows {
			// taken over between the two reads: upgrading the shared lock
			// could deadlock, so the transaction runs again from the start
			return errContended
		}
	}
	if have == model {
		return nil
	}
	held, err := s.holds(ctx, tx, t, collection)
	if err != nil {
		return err
	}
	if held {
		return fmt.Errorf("%w: collection %q holds %q, not %q", vector.ErrModelMismatch, collection, have, model)
	}
	_, err = tx.ExecContext(ctx, s.q(`UPDATE `+t.meta+` SET value = ? WHERE name = ?`), model, key)
	return err
}

// queryer is what holds reads with: the database or a transaction.
type queryer interface {
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

// holds reports whether any partition of collection has documents.
func (s *Store) holds(ctx context.Context, q queryer, t tables, collection string) (bool, error) {
	var held bool
	err := q.QueryRowContext(ctx, s.q(`SELECT EXISTS (SELECT 1 FROM `+t.docs+` WHERE collection = ?)`), collection).Scan(&held)
	return held, err
}

// Model implements vector.Store.
func (s *Store) Model(ctx context.Context, c vector.Coord) (string, error) {
	if err := vector.CheckCoord(c); err != nil {
		return "", err
	}
	var model string
	_, err := s.withTables(ctx, c.Index, false, func(t tables) error {
		err := s.db.QueryRowContext(ctx, s.q(`SELECT m.value FROM `+t.meta+` m WHERE m.name = ?
			AND EXISTS (SELECT 1 FROM `+t.docs+` d WHERE d.collection = ?)`), modelKey+c.Collection, c.Collection).Scan(&model)
		if errors.Is(err, sql.ErrNoRows) {
			model = ""
			return nil
		}
		return err
	})
	if err != nil {
		return "", s.wrap("model", err)
	}
	return model, nil
}

// Delete implements vector.Store in one transaction (see deleteWhere), which
// locks the documents in the order Upsert does, so the two never deadlock;
// a write transaction that fails with a deadlock or serialization error
// anyway runs again, up to maxAttempts times.
func (s *Store) Delete(ctx context.Context, c vector.Coord, ids ...string) error {
	if err := vector.CheckRead(c, ids...); err != nil {
		return err
	}
	if len(ids) == 0 {
		return nil
	}
	return s.wrap("delete", s.delete(ctx, c, ids))
}

func (s *Store) delete(ctx context.Context, c vector.Coord, ids []string) error {
	list, err := jsonIDs(ids)
	if err != nil {
		return err
	}
	_, err = s.withTables(ctx, c.Index, false, func(t tables) error {
		return s.retry(ctx, func() error {
			return s.deleteWhere(ctx, t, nil, `partition = ? AND collection = ? AND doc_id IN (`+s.d.IDs+`)`, c.Partition, c.Collection, list)
		})
	})
	return err
}

// DeleteCollection implements vector.Store as Delete does, for every
// document of c, and drops the collection's model record in the same
// transaction when no partition of it holds documents any more.
func (s *Store) DeleteCollection(ctx context.Context, c vector.Coord) error {
	if err := vector.CheckCoord(c); err != nil {
		return err
	}
	_, err := s.withTables(ctx, c.Index, false, func(t tables) error {
		return s.retry(ctx, func() error {
			return s.deleteWhere(ctx, t, &c.Collection, `partition = ? AND collection = ?`, c.Partition, c.Collection)
		})
	})
	return s.wrap("delete collection", err)
}

// deleteWhere deletes the documents of t that cond, a constant clause over
// its args, picks, in one transaction: the docs rows first, as Upsert
// writes them first, after taking their locks in id order where the
// Dialect locks rows (see Dialect.LockRows), then their chunks. With a
// collection, it then drops that collection's model record if no partition
// of it holds documents any more, having locked the record first, as
// claimModel does, so no writer of that model adds documents meanwhile.
func (s *Store) deleteWhere(ctx context.Context, t tables, collection *string, cond string, args ...any) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if collection != nil && s.d.LockRows != "" {
		if _, err := tx.ExecContext(ctx, s.q(`SELECT value FROM `+t.meta+` WHERE name = ? `+s.d.LockRows), modelKey+*collection); err != nil {
			return err
		}
	}
	if s.d.LockRows != "" {
		if _, err := tx.ExecContext(ctx, s.q(`SELECT doc_id FROM `+t.docs+` WHERE `+cond+` ORDER BY doc_id `+s.d.LockRows), args...); err != nil {
			return err
		}
	}
	for _, table := range []string{t.docs, t.chunks} {
		if _, err := tx.ExecContext(ctx, s.q(`DELETE FROM `+table+` WHERE `+cond), args...); err != nil {
			return err
		}
	}
	if collection != nil {
		held, err := s.holds(ctx, tx, t, *collection)
		if err != nil {
			return err
		}
		if !held {
			if _, err := tx.ExecContext(ctx, s.q(`DELETE FROM `+t.meta+` WHERE name = ?`), modelKey+*collection); err != nil {
				return err
			}
		}
	}
	return tx.Commit()
}

// DeleteIndex implements vector.Store by dropping the index's tables, those
// of the schema before the meta table included, all in one DDL transaction:
// no other Store ever sees some of them gone and the rest still there, and
// a failure part of the way drops none, so no later write can bring back a
// meta table over old documents. They are dropped in the order every write
// transaction locks them, meta, docs, chunks, so a drop waiting for a
// writer never deadlocks with it.
func (s *Store) DeleteIndex(ctx context.Context, index string) error {
	if index == "" {
		return vector.ErrEmptyIndex
	}
	t, err := s.tables(index)
	if err != nil {
		return s.wrap("delete index", err)
	}
	defer s.forget(index)
	err = s.retry(ctx, func() error {
		return s.ddl(ctx, t, func(ex execer) error {
			for _, table := range []string{t.meta, t.docs, t.chunks, t.legacy} {
				if _, err := ex.ExecContext(ctx, `DROP TABLE IF EXISTS `+table); err != nil {
					return err
				}
			}
			return nil
		})
	})
	return s.wrap("delete index", err)
}

// Hashes implements vector.Store.
func (s *Store) Hashes(ctx context.Context, c vector.Coord, ids ...string) (map[string]string, error) {
	if err := vector.CheckRead(c, ids...); err != nil {
		return nil, err
	}
	out, err := s.hashes(ctx, c, ids)
	return out, s.wrap("hashes", err)
}

func (s *Store) hashes(ctx context.Context, c vector.Coord, ids []string) (map[string]string, error) {
	out := make(map[string]string, len(ids))
	if len(ids) == 0 {
		return out, nil
	}
	list, err := jsonIDs(ids)
	if err != nil {
		return nil, err
	}
	_, err = s.withTables(ctx, c.Index, false, func(t tables) error {
		clear(out)
		rows, err := s.db.QueryContext(ctx, s.q(`SELECT doc_id, content_hash FROM `+t.docs+` WHERE partition = ? AND collection = ? AND doc_id IN (`+s.d.IDs+`)`),
			c.Partition, c.Collection, list)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var id, hash string
			if err := rows.Scan(&id, &hash); err != nil {
				return err
			}
			out[id] = hash
		}
		return rows.Err()
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Docs implements vector.Store.
func (s *Store) Docs(ctx context.Context, c vector.Coord, after string, limit int) ([]vector.DocInfo, error) {
	if err := vector.CheckDocs(c, after); err != nil {
		return nil, err
	}
	out, err := s.docs(ctx, c, after, limit)
	return out, s.wrap("docs", err)
}

func (s *Store) docs(ctx context.Context, c vector.Coord, after string, limit int) ([]vector.DocInfo, error) {
	args := []any{c.Partition, c.Collection, after}
	paging := ""
	if limit > 0 {
		paging = ` LIMIT ?`
		args = append(args, limit)
	}
	var out []vector.DocInfo
	_, err := s.withTables(ctx, c.Index, false, func(t tables) error {
		out = nil
		rows, err := s.db.QueryContext(ctx, s.q(`SELECT doc_id, content_hash FROM `+t.docs+` WHERE partition = ? AND collection = ? AND doc_id > ? ORDER BY doc_id`+paging), args...)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var d vector.DocInfo
			if err := rows.Scan(&d.ID, &d.Hash); err != nil {
				return err
			}
			out = append(out, d)
		}
		return rows.Err()
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Chunks implements vector.Store.
func (s *Store) Chunks(ctx context.Context, c vector.Coord, id string) ([]vector.Chunk, error) {
	if err := vector.CheckRead(c, id); err != nil {
		return nil, err
	}
	out, err := s.chunks(ctx, c, id)
	return out, s.wrap("chunks", err)
}

func (s *Store) chunks(ctx context.Context, c vector.Coord, id string) ([]vector.Chunk, error) {
	var out []vector.Chunk
	_, err := s.withTables(ctx, c.Index, false, func(t tables) error {
		out = nil
		rows, err := s.db.QueryContext(ctx, s.q(`SELECT seq, text, embedding FROM `+t.chunks+`
			WHERE partition = ? AND collection = ? AND doc_id = ? ORDER BY seq, id`), c.Partition, c.Collection, id)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var ch vector.Chunk
			var src any
			if err := rows.Scan(&ch.Seq, &ch.Text, &src); err != nil {
				return err
			}
			if ch.Vector, err = s.d.Decode(src); err != nil {
				return fmt.Errorf("chunk %d: %w", ch.Seq, err)
			}
			out = append(out, ch)
		}
		return rows.Err()
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Search implements vector.Store with one exact query, whatever the number
// of query vectors: every chunk in scope is scored by its smallest distance
// to any of them, in one scan of the chunks, the best chunk of each
// document (the lowest Seq on a tie) is kept, and the q.Limit best of those
// are returned, each with the first query vector at that distance. The
// cosine distance (0 = identical, 2 = opposite) is reported as 1-distance,
// so that higher is better.
func (s *Store) Search(ctx context.Context, c vector.Coord, q vector.Query) ([]vector.Match, error) {
	if err := vector.CheckSearch(c, s.dims, q); err != nil {
		return nil, err
	}
	if q.Limit <= 0 {
		return nil, nil
	}
	out, err := s.search(ctx, c, q)
	return out, s.wrap("search", err)
}

func (s *Store) search(ctx context.Context, c vector.Coord, q vector.Query) ([]vector.Match, error) {
	where, args, err := s.scope(c, q)
	if err != nil {
		return nil, err
	}
	// one vector is bound as it is; several come as one argument, turned
	// into the table q by the Dialect, and a chunk's distance is its
	// smallest to any row of q
	with, dist, qi := "", s.d.Distance("ch.embedding", "?"), "0"
	arg := s.d.Encode(q.Vectors[0])
	if len(q.Vectors) > 1 {
		with = s.d.Queries(s.dims)
		dist = `(SELECT min(` + s.d.Distance("ch.embedding", "q.v") + `) FROM q)`
		qi = `COALESCE((SELECT min(q.i) FROM q WHERE ` + s.d.Distance("c.embedding", "q.v") + ` = b.d), 0)`
		arg = s.d.EncodeQueries(q.Vectors)
	}
	args = append([]any{arg}, args...)
	args = append(args, q.Limit)
	var out []vector.Match
	_, err = s.withTables(ctx, c.Index, false, func(t tables) error {
		out = nil
		if err := s.checkModel(ctx, t, c.Collection, q); err != nil {
			return err
		}
		// scored: each chunk's distance to its closest query vector; ranked:
		// the best chunk of each document; b: the q.Limit best documents,
		// joined back for their text and the index of the closest vector
		query := with + `
			SELECT c.partition, c.collection, c.doc_id, c.seq, c.text, b.d, ` + qi + ` FROM (
				SELECT id, partition, collection, doc_id, seq, d FROM (
					SELECT id, partition, collection, doc_id, seq, d,
						row_number() OVER (PARTITION BY partition, collection, doc_id ORDER BY d, seq) AS rn
					FROM (SELECT id, partition, collection, doc_id, seq, ` + dist + ` AS d
						FROM ` + t.chunks + ` ch` + where + `) scored
				) ranked WHERE rn = 1 ORDER BY d, doc_id, seq, partition, collection LIMIT ?
			) b JOIN ` + t.chunks + ` c ON c.id = b.id
			ORDER BY b.d, b.doc_id, b.seq, b.partition, b.collection`
		rows, err := s.db.QueryContext(ctx, s.q(query), args...)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			m := vector.Match{Coord: vector.Coord{Index: c.Index}}
			var dist float64
			if err := rows.Scan(&m.Coord.Partition, &m.Coord.Collection, &m.DocID, &m.Seq, &m.Text, &dist, &m.Query); err != nil {
				return err
			}
			m.Score = 1 - dist
			out = append(out, m)
		}
		return rows.Err()
	})
	if err != nil {
		return nil, err
	}
	return out, nil
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
// collection in its scope recorded another and still holds documents.
func (s *Store) checkModel(ctx context.Context, t tables, collection string, q vector.Query) error {
	if q.Model == "" {
		return nil
	}
	// a record holds only while its collection has documents
	held := ` AND EXISTS (SELECT 1 FROM ` + t.docs + ` d WHERE d.collection = substr(m.name, ?))`
	query := `SELECT m.name, m.value FROM ` + t.meta + ` m WHERE m.name = ? AND m.value <> ?` + held
	args := []any{modelKey + collection, q.Model, len(modelKey) + 1}
	if q.AllCollections {
		query = `SELECT m.name, m.value FROM ` + t.meta + ` m WHERE substr(m.name, 1, ?) = ? AND m.value <> ?` + held + ` ORDER BY m.name LIMIT 1`
		args = []any{len(modelKey), modelKey, q.Model, len(modelKey) + 1}
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

// storeError is an error this package words itself, prefixed already, that
// wrap passes on as it is.
type storeError struct{ error }

func (e storeError) Unwrap() error { return e.error }

// ownf formats a storeError.
func ownf(format string, a ...any) error { return storeError{fmt.Errorf(format, a...)} }

// wrap prefixes err, a driver's or one of database/sql's, with "vector:",
// the backend's name and the operation, so the caller can tell where it
// comes from; the package's own errors and the vector sentinels pass as
// they are, and %w keeps every one of them matchable.
func (s *Store) wrap(op string, err error) error {
	var own storeError
	switch {
	case err == nil, errors.As(err, &own),
		errors.Is(err, vector.ErrDimensions), errors.Is(err, vector.ErrModelMismatch):
		return err
	}
	return fmt.Errorf("vector: %s %s: %w", s.d.Name, op, err)
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

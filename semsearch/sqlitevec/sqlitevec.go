// Package sqlitevec is a semsearch.Store backed by SQLite with the sqlite-vec
// extension. The extension is compiled in (CGO) and registered for every
// SQLite connection the process opens afterwards, so an application that
// opens its connections elsewhere (through an ORM, say) imports this package,
// blank if need be, before it opens the first one.
package sqlitevec

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"sync"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"

	"github.com/go-bumbu/langai/internal/tablename"
	"github.com/go-bumbu/langai/semsearch"
)

func init() { sqlite_vec.Auto() }

// Store implements semsearch.Store on SQLite + sqlite-vec with one physical
// table set per index (e.g. a tenant) and generation, created lazily. partition
// and collection are vec0 metadata columns, filtered by equality in the KNN
// query.
type Store struct {
	db         *sql.DB
	dims       int
	generation string
	mu         sync.Mutex
	created    map[string]bool
}

// Option customizes a Store.
type Option func(*Store)

// WithGeneration keeps the store's tables apart from those of every other
// generation in the same database (see tablename.For), so the vectors of a
// different embedding model can be built next to the current ones before an
// application switches over. generation is "" (the default, the tables that
// existed before generations) or 1-12 lowercase letters and digits.
func WithGeneration(generation string) Option {
	return func(s *Store) { s.generation = generation }
}

// New returns a Store using db and checks the sqlite-vec extension is
// loaded. dims must match the embedder's Dimensions(); vectors of another
// model, whatever their size, belong in another generation (WithGeneration)
// rather than in this one's tables.
func New(ctx context.Context, db *sql.DB, dims int, opts ...Option) (*Store, error) {
	if db == nil {
		return nil, errors.New("nil database handle")
	}
	if dims <= 0 {
		return nil, fmt.Errorf("invalid vector dimensions %d", dims)
	}
	s := &Store{db: db, dims: dims, created: map[string]bool{}}
	for _, opt := range opts {
		opt(s)
	}
	if err := tablename.CheckGeneration(s.generation); err != nil {
		return nil, err
	}
	var v string
	if err := db.QueryRowContext(ctx, `SELECT vec_version()`).Scan(&v); err != nil {
		return nil, fmt.Errorf("sqlite-vec extension not available: %w", err)
	}
	return s, nil
}

func (s *Store) ensure(ctx context.Context, index string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.created[index] {
		return nil
	}
	docs, err := tablename.For(s.generation, index, "docs")
	if err != nil {
		return err
	}
	chunks, _ := tablename.For(s.generation, index, "chunks")
	vec, _ := tablename.For(s.generation, index, "vec")
	stmts := []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			partition TEXT NOT NULL, collection TEXT NOT NULL,
			doc_id TEXT NOT NULL, content_hash TEXT NOT NULL,
			PRIMARY KEY (partition, collection, doc_id))`, docs),
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			partition TEXT NOT NULL, collection TEXT NOT NULL,
			doc_id TEXT NOT NULL, seq INTEGER NOT NULL, text TEXT NOT NULL)`, chunks),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_doc ON %s(partition, collection, doc_id)`, chunks, chunks),
		fmt.Sprintf(`CREATE VIRTUAL TABLE IF NOT EXISTS %s USING vec0(
			embedding float[%d] distance_metric=cosine, partition text, collection text)`, vec, s.dims),
	}
	for _, st := range stmts {
		if _, err := s.db.ExecContext(ctx, st); err != nil {
			return fmt.Errorf("create %s schema: %w", index, err)
		}
	}
	s.created[index] = true
	return nil
}

func (s *Store) Upsert(ctx context.Context, index, partition, collection, docID, contentHash string, chunks []semsearch.Chunk) error {
	for _, c := range chunks {
		if len(c.Vector) != s.dims {
			return fmt.Errorf("chunk %d has %d dimensions, store expects %d", c.Seq, len(c.Vector), s.dims)
		}
	}
	if err := s.ensure(ctx, index); err != nil {
		return err
	}
	docs, _ := tablename.For(s.generation, index, "docs")
	chunksT, _ := tablename.For(s.generation, index, "chunks")
	vecT, _ := tablename.For(s.generation, index, "vec")
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := deleteDoc(ctx, tx, chunksT, vecT, partition, collection, docID); err != nil {
		return err
	}
	for _, c := range chunks {
		res, err := tx.ExecContext(ctx,
			fmt.Sprintf(`INSERT INTO %s (partition, collection, doc_id, seq, text) VALUES (?, ?, ?, ?, ?)`, chunksT),
			partition, collection, docID, c.Seq, c.Text)
		if err != nil {
			return err
		}
		id, err := res.LastInsertId()
		if err != nil {
			return err
		}
		blob, err := sqlite_vec.SerializeFloat32(c.Vector)
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx,
			fmt.Sprintf(`INSERT INTO %s (rowid, embedding, partition, collection) VALUES (?, ?, ?, ?)`, vecT),
			id, blob, partition, collection); err != nil {
			return err
		}
	}
	if _, err := tx.ExecContext(ctx,
		fmt.Sprintf(`INSERT INTO %s (partition, collection, doc_id, content_hash) VALUES (?, ?, ?, ?)
			ON CONFLICT(partition, collection, doc_id) DO UPDATE SET content_hash = excluded.content_hash`, docs),
		partition, collection, docID, contentHash); err != nil {
		return err
	}
	return tx.Commit()
}

func deleteDoc(ctx context.Context, tx *sql.Tx, chunksT, vecT, partition, collection, docID string) error {
	rows, err := tx.QueryContext(ctx,
		fmt.Sprintf(`SELECT id FROM %s WHERE partition=? AND collection=? AND doc_id=?`, chunksT),
		partition, collection, docID)
	if err != nil {
		return err
	}
	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			_ = rows.Close()
			return err
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	_ = rows.Close()
	for _, id := range ids {
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s WHERE rowid = ?`, vecT), id); err != nil {
			return err
		}
	}
	_, err = tx.ExecContext(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE partition=? AND collection=? AND doc_id=?`, chunksT),
		partition, collection, docID)
	return err
}

func (s *Store) Delete(ctx context.Context, index, partition, collection, docID string) error {
	if err := s.ensure(ctx, index); err != nil {
		return err
	}
	docs, _ := tablename.For(s.generation, index, "docs")
	chunksT, _ := tablename.For(s.generation, index, "chunks")
	vecT, _ := tablename.For(s.generation, index, "vec")
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := deleteDoc(ctx, tx, chunksT, vecT, partition, collection, docID); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE partition=? AND collection=? AND doc_id=?`, docs),
		partition, collection, docID); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *Store) ContentHash(ctx context.Context, index, partition, collection, docID string) (string, error) {
	if err := s.ensure(ctx, index); err != nil {
		return "", err
	}
	docs, _ := tablename.For(s.generation, index, "docs")
	var h string
	err := s.db.QueryRowContext(ctx,
		fmt.Sprintf(`SELECT content_hash FROM %s WHERE partition=? AND collection=? AND doc_id=?`, docs),
		partition, collection, docID).Scan(&h)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	return h, err
}

// Search implements semsearch.Store.
func (s *Store) Search(ctx context.Context, index, partition, collection string, vector []float32, limit int) ([]semsearch.Match, error) {
	return s.SearchScope(ctx, index, semsearch.ScopeFilter{Partition: partition, Collection: collection}, vector, limit)
}

// SearchScope implements semsearch.Store as an exact KNN query on the vec0
// table, filtered by its partition/collection metadata columns. Its cosine
// distance (0 = identical, 2 = opposite) is reported as 1-distance so that
// higher is better, matching the semsearch contract.
func (s *Store) SearchScope(ctx context.Context, index string, f semsearch.ScopeFilter, vector []float32, limit int) ([]semsearch.Match, error) {
	if limit <= 0 {
		return nil, nil
	}
	if len(vector) != s.dims {
		return nil, fmt.Errorf("query vector has %d dimensions, store expects %d", len(vector), s.dims)
	}
	if err := s.ensure(ctx, index); err != nil {
		return nil, err
	}
	chunksT, _ := tablename.For(s.generation, index, "chunks")
	vecT, _ := tablename.For(s.generation, index, "vec")
	blob, err := sqlite_vec.SerializeFloat32(vector)
	if err != nil {
		return nil, err
	}
	where := []string{"embedding MATCH ?", "k = ?"}
	args := []any{blob, limit}
	if f.Partition != "" {
		where = append(where, "partition = ?")
		args = append(args, f.Partition)
	}
	if f.Collection != "" {
		where = append(where, "collection = ?")
		args = append(args, f.Collection)
	}
	//nolint:gosec // table name is a hex sha256 of the index (injection-safe); all values are bound ? parameters
	q := fmt.Sprintf(`
		SELECT c.doc_id, c.partition, c.collection, c.seq, c.text, v.distance
		FROM (SELECT rowid, distance FROM %s WHERE %s ORDER BY distance) v
		JOIN %s c ON c.id = v.rowid
		ORDER BY v.distance`, vecT, strings.Join(where, " AND "), chunksT)
	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []semsearch.Match
	for rows.Next() {
		var m semsearch.Match
		var dist float64
		if err := rows.Scan(&m.DocID, &m.Partition, &m.Collection, &m.Seq, &m.Text, &dist); err != nil {
			return nil, err
		}
		m.Score = 1 - dist
		out = append(out, m)
	}
	return out, rows.Err()
}

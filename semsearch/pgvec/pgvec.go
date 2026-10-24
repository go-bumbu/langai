// Package pgvec is a semsearch.Store backed by Postgres with the pgvector
// extension. The database user must be able to run CREATE EXTENSION vector,
// or the extension must already be installed.
package pgvec

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/pgvector/pgvector-go"

	"github.com/go-bumbu/langai/internal/tablename"
	"github.com/go-bumbu/langai/semsearch"
)

// Store implements semsearch.Store on Postgres + pgvector, with one physical
// table set per index (e.g. a tenant) and generation, created lazily. partition
// and collection are ordinary indexed TEXT columns, and searches rank the
// rows of their scope exactly (see ensure).
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

// New returns a Store using db and ensures the pgvector extension exists.
// dims must match the embedder's Dimensions(); vectors of another model,
// whatever their size, belong in another generation (WithGeneration) rather
// than in this one's tables.
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
	if _, err := db.ExecContext(ctx, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return nil, fmt.Errorf("pgvector extension not available (install it or grant CREATE EXTENSION): %w", err)
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
	stmts := []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			partition TEXT NOT NULL, collection TEXT NOT NULL,
			doc_id TEXT NOT NULL, content_hash TEXT NOT NULL,
			PRIMARY KEY (partition, collection, doc_id))`, docs),
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			id BIGSERIAL PRIMARY KEY,
			partition TEXT NOT NULL, collection TEXT NOT NULL,
			doc_id TEXT NOT NULL, seq INT NOT NULL, text TEXT NOT NULL,
			embedding vector(%d) NOT NULL)`, chunks, s.dims),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_doc ON %s(partition, collection, doc_id)`, chunks, chunks),
		// Searches are exact: the index above narrows them to their scope,
		// whose rows are then ranked by distance. Earlier versions also built
		// an HNSW index here, which filters only after collecting its
		// ef_search nearest rows of the whole table, so a scope without rows
		// among those came back short or empty. Drop it where it exists.
		fmt.Sprintf(`DROP INDEX IF EXISTS %s_embedding_idx`, chunks),
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
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE partition = $1 AND collection = $2 AND doc_id = $3`, chunksT),
		partition, collection, docID); err != nil {
		return err
	}
	for _, c := range chunks {
		if _, err := tx.ExecContext(ctx,
			fmt.Sprintf(`INSERT INTO %s (partition, collection, doc_id, seq, text, embedding) VALUES ($1, $2, $3, $4, $5, $6)`, chunksT),
			partition, collection, docID, c.Seq, c.Text, pgvector.NewVector(c.Vector)); err != nil {
			return err
		}
	}
	if _, err := tx.ExecContext(ctx,
		fmt.Sprintf(`INSERT INTO %s (partition, collection, doc_id, content_hash) VALUES ($1, $2, $3, $4)
			ON CONFLICT (partition, collection, doc_id) DO UPDATE SET content_hash = excluded.content_hash`, docs),
		partition, collection, docID, contentHash); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *Store) Delete(ctx context.Context, index, partition, collection, docID string) error {
	if err := s.ensure(ctx, index); err != nil {
		return err
	}
	docs, _ := tablename.For(s.generation, index, "docs")
	chunksT, _ := tablename.For(s.generation, index, "chunks")
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE partition = $1 AND collection = $2 AND doc_id = $3`, chunksT),
		partition, collection, docID); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE partition = $1 AND collection = $2 AND doc_id = $3`, docs),
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
		fmt.Sprintf(`SELECT content_hash FROM %s WHERE partition = $1 AND collection = $2 AND doc_id = $3`, docs),
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

// SearchScope implements semsearch.Store. The <=> operator is pgvector's
// cosine distance (0 = identical, 2 = opposite); scores are reported as
// 1-distance so that higher is better, matching the semsearch contract.
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
	where := []string{}
	args := []any{pgvector.NewVector(vector)} // $1
	n := 2
	if f.Partition != "" {
		where = append(where, fmt.Sprintf("partition = $%d", n))
		args = append(args, f.Partition)
		n++
	}
	if f.Collection != "" {
		where = append(where, fmt.Sprintf("collection = $%d", n))
		args = append(args, f.Collection)
		n++
	}
	clause := ""
	if len(where) > 0 {
		clause = "WHERE " + strings.Join(where, " AND ")
	}
	args = append(args, limit) // $n
	//nolint:gosec // table name is a hex sha256 of the index (injection-safe); all values are bound $n parameters
	q := fmt.Sprintf(`SELECT doc_id, partition, collection, seq, text, (embedding <=> $1)::float8
		FROM %s %s ORDER BY embedding <=> $1 LIMIT $%d`, chunksT, clause, n)
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

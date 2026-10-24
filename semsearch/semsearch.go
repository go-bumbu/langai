// Package semsearch is a reusable semantic-search library: it turns documents
// into embedding vectors (via a pluggable Embedder), stores them in a
// pluggable vector Store, and answers free-text queries with the closest
// documents. It knows nothing about what the documents are: callers hand it
// opaque document ids and plain text.
//
// Two production Store backends live in subpackages: sqlitevec (SQLite with
// the sqlite-vec extension, requires CGO) and pgvec (Postgres with the
// pgvector extension). MemStore in this package is a brute-force in-memory
// backend for tests, tiny corpora and evaluation. The production Embedder is
// the yzma subpackage, a GGUF model run locally through llama.cpp;
// LazyEmbedder defers loading one to its first use, and FakeEmbedder is a
// deterministic embedder for tests. The eval subpackage measures an
// Embedder's search and tagging quality over a labelled dataset.
package semsearch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
)

// Role tells an Embedder what a text is for. Asymmetric models are trained
// with different prefixes for the text being searched with and the texts it
// is searched against; symmetric models ignore the role.
type Role int

const (
	// RoleDocument marks text that is being indexed.
	RoleDocument Role = iota
	// RoleQuery marks text that is being searched with.
	RoleQuery
)

// String implements fmt.Stringer for log and error messages.
func (r Role) String() string {
	switch r {
	case RoleDocument:
		return "document"
	case RoleQuery:
		return "query"
	}
	return fmt.Sprintf("Role(%d)", int(r))
}

// Embedder turns texts into embedding vectors. All vectors returned must have
// exactly Dimensions() entries; the result slice matches the input order. The
// role lets an asymmetric model apply its query or document prefix.
type Embedder interface {
	Embed(ctx context.Context, role Role, texts []string) ([][]float32, error)
	Dimensions() int
}

// Chunk is one embedded piece of a document as handed to a Store: its
// position within the document, the text that was embedded, and its vector.
type Chunk struct {
	Seq    int
	Text   string
	Vector []float32
}

// Match is one search hit. Partition/Collection identify which coordinate it
// came from (useful when a SearchScope spans several).
type Match struct {
	DocID      string
	Partition  string
	Collection string
	Seq        int
	Text       string
	Score      float64
}

// ScopeFilter narrows a SearchScope. An empty Partition and/or Collection means
// "do not constrain that coordinate".
type ScopeFilter struct {
	Partition  string
	Collection string
}

// Store persists chunk vectors partitioned by an opaque index (the hard
// isolation boundary) and located within it by two opaque coordinates,
// partition and collection. Implementations must return results ordered by
// descending score and must never let one index's data appear in another's.
type Store interface {
	Upsert(ctx context.Context, index, partition, collection, docID, contentHash string, chunks []Chunk) error
	Delete(ctx context.Context, index, partition, collection, docID string) error
	ContentHash(ctx context.Context, index, partition, collection, docID string) (string, error)
	Search(ctx context.Context, index, partition, collection string, vector []float32, limit int) ([]Match, error)
	SearchScope(ctx context.Context, index string, f ScopeFilter, vector []float32, limit int) ([]Match, error)
}

// Index ties an Embedder, a Store and a Chunker together into the high-level
// operations callers use: index a document, remove it, search.
type Index struct {
	embedder   Embedder
	store      Store
	chunker    Chunker
	minScore   float64
	index      string
	partition  string
	collection string
}

// Option customizes an Index.
type Option func(*Index)

// WithChunker replaces the default markdown-aware chunker.
func WithChunker(c Chunker) Option {
	return func(ix *Index) { ix.chunker = c }
}

// WithMinScore drops search hits scoring below min.
//
// Without it, Search returns the nearest documents, not the *near* ones: a
// nearest-neighbour query always fills its limit, so a corpus smaller than the
// limit answers every query with every document, however unrelated. A floor is
// what turns "closest" into "relevant" — and it is a caller's decision, since
// the useful value depends on the embedder and on whether the caller would
// rather show a weak hit than nothing. Cosine backends score 1 for an exact
// match and around 0 for unrelated text.
func WithMinScore(min float64) Option {
	return func(ix *Index) { ix.minScore = min }
}

// New returns an Index using the given embedder and store.
func New(embedder Embedder, store Store, index, partition, collection string, opts ...Option) (*Index, error) {
	if embedder == nil {
		return nil, errors.New("nil embedder")
	}
	if store == nil {
		return nil, errors.New("nil store")
	}
	// no floor by default: math.Inf(-1) rather than 0, because 0 is a
	// meaningful cutoff on cosine backends and would silently become policy
	ix := &Index{
		embedder: embedder, store: store, chunker: &MarkdownChunker{},
		minScore: math.Inf(-1), index: index, partition: partition, collection: collection,
	}
	for _, opt := range opts {
		opt(ix)
	}
	return ix, nil
}

// Index chunks, embeds and stores a document, replacing whatever was indexed
// under docID before. It is a cheap no-op when the text is unchanged since
// the last call, so it is safe to call eagerly (e.g. a full re-index at
// startup).
func (ix *Index) Index(ctx context.Context, docID, text string) error {
	if docID == "" {
		return errors.New("empty document id")
	}
	hash := ContentHash(text)
	stored, err := ix.store.ContentHash(ctx, ix.index, ix.partition, ix.collection, docID)
	if err != nil {
		return fmt.Errorf("semsearch: read content hash: %w", err)
	}
	if stored == hash {
		return nil
	}
	parts := ix.chunker.Chunk(text)
	chunks := make([]Chunk, 0, len(parts))
	if len(parts) > 0 {
		vectors, err := ix.embedder.Embed(ctx, RoleDocument, parts)
		if err != nil {
			return fmt.Errorf("semsearch: embed document: %w", err)
		}
		if len(vectors) != len(parts) {
			return fmt.Errorf("semsearch: embedder returned %d vectors for %d texts", len(vectors), len(parts))
		}
		for i, part := range parts {
			chunks = append(chunks, Chunk{Seq: i, Text: part, Vector: vectors[i]})
		}
	}
	if err := ix.store.Upsert(ctx, ix.index, ix.partition, ix.collection, docID, hash, chunks); err != nil {
		return fmt.Errorf("semsearch: store document: %w", err)
	}
	return nil
}

// Delete removes a document from the index. Unknown ids are a no-op.
func (ix *Index) Delete(ctx context.Context, docID string) error {
	if docID == "" {
		return errors.New("empty document id")
	}
	return ix.store.Delete(ctx, ix.index, ix.partition, ix.collection, docID)
}

// Search embeds the query and returns the closest documents, best first, at
// most one Match per document (its best-scoring chunk). Hits below the
// WithMinScore floor are dropped, so a query with nothing relevant to find can
// legitimately return fewer results than limit — or none. An empty query
// returns no results.
func (ix *Index) Search(ctx context.Context, query string, limit int) ([]Match, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}
	if limit <= 0 {
		limit = 10
	}
	vectors, err := ix.embedder.Embed(ctx, RoleQuery, []string{query})
	if err != nil {
		return nil, fmt.Errorf("semsearch: embed query: %w", err)
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("semsearch: embedder returned %d vectors for 1 query", len(vectors))
	}
	// over-fetch chunks so that several chunks of the same document don't
	// crowd other documents out of the final per-document list
	raw, err := ix.store.Search(ctx, ix.index, ix.partition, ix.collection, vectors[0], limit*4)
	if err != nil {
		return nil, fmt.Errorf("semsearch: search: %w", err)
	}
	seen := make(map[string]bool, len(raw))
	out := make([]Match, 0, limit)
	for _, m := range raw {
		// the store returns hits best first, so the first one below the floor
		// ends the list — no point looking at weaker chunks
		if m.Score < ix.minScore {
			break
		}
		if seen[m.DocID] {
			continue
		}
		seen[m.DocID] = true
		out = append(out, m)
		if len(out) == limit {
			break
		}
	}
	return out, nil
}

// ContentHash is the fingerprint used to detect unchanged documents.
func ContentHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

// Normalize scales vec to unit length in place, so cosine similarity and dot
// product rank alike. An all-zero vector has no direction and stays zero.
func Normalize(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return
	}
	inv := 1 / math.Sqrt(sum)
	for i := range vec {
		vec[i] = float32(float64(vec[i]) * inv)
	}
}

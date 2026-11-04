// Package search is a reusable semantic-search capability: it turns documents
// into embedding vectors (via a langai.Embedder), stores them in a
// vector.Store, and answers free-text queries with the closest documents. It
// knows nothing about what the documents are: callers hand it opaque document
// ids and plain text.
//
// The Stores live in the vector package and its subpackages, and the
// production Embedder in provider/yzma, a GGUF model run locally through
// llama.cpp.
package search

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/vector"
)

// Index ties an Embedder, a Store and a Chunker together into the high-level
// operations callers use: index a document, remove it, search.
type Index struct {
	embedder   langai.Embedder
	store      vector.Store
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
func New(embedder langai.Embedder, store vector.Store, index, partition, collection string, opts ...Option) (*Index, error) {
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
		return fmt.Errorf("search: read content hash: %w", err)
	}
	if stored == hash {
		return nil
	}
	parts := ix.chunker.Chunk(text)
	chunks := make([]vector.Chunk, 0, len(parts))
	if len(parts) > 0 {
		vectors, err := ix.embedder.Embed(ctx, langai.RoleDocument, parts)
		if err != nil {
			return fmt.Errorf("search: embed document: %w", err)
		}
		if len(vectors) != len(parts) {
			return fmt.Errorf("search: embedder returned %d vectors for %d texts", len(vectors), len(parts))
		}
		for i, part := range parts {
			chunks = append(chunks, vector.Chunk{Seq: i, Text: part, Vector: vectors[i]})
		}
	}
	if err := ix.store.Upsert(ctx, ix.index, ix.partition, ix.collection, docID, hash, chunks); err != nil {
		return fmt.Errorf("search: store document: %w", err)
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
func (ix *Index) Search(ctx context.Context, query string, limit int) ([]vector.Match, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}
	if limit <= 0 {
		limit = 10
	}
	vectors, err := ix.embedder.Embed(ctx, langai.RoleQuery, []string{query})
	if err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("search: embedder returned %d vectors for 1 query", len(vectors))
	}
	// over-fetch chunks so that several chunks of the same document don't
	// crowd other documents out of the final per-document list
	raw, err := ix.store.Search(ctx, ix.index, ix.partition, ix.collection, vectors[0], limit*4)
	if err != nil {
		return nil, fmt.Errorf("search: search: %w", err)
	}
	seen := make(map[string]bool, len(raw))
	out := make([]vector.Match, 0, limit)
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

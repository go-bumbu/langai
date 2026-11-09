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

var (
	// ErrEmptyID is returned by every call that takes a document id when the
	// id is empty, before any model or store is asked anything.
	ErrEmptyID = errors.New("search: empty document id")
	// ErrNotIndexed is returned by SearchIndexed for an id with no stored
	// vectors: never indexed, deleted, or indexed with nothing to embed.
	ErrNotIndexed = errors.New("search: no stored vectors")
)

// Index ties an Embedder, a Store and a Chunker together into the high-level
// operations callers use: index a document, remove it, search.
type Index struct {
	embedder langai.Embedder
	store    vector.Store
	chunker  Chunker
	minScore float64
	coord    vector.Coord
	model    string // the embedder's, when it names it
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

// New returns an Index using the given embedder and store, over the
// collection at index, partition and collection; an empty partition or
// collection is a value like any other, never a wildcard. When the embedder
// names its model (langai.Identified), the store records it with the first
// vectors written to the collection and refuses another model's afterwards.
func New(embedder langai.Embedder, store vector.Store, index, partition, collection string, opts ...Option) (*Index, error) {
	if embedder == nil {
		return nil, errors.New("search: nil embedder")
	}
	if store == nil {
		return nil, errors.New("search: nil store")
	}
	// no floor by default: math.Inf(-1) rather than 0, because 0 is a
	// meaningful cutoff on cosine backends and would silently become policy
	ix := &Index{
		embedder: embedder, store: store, chunker: &MarkdownChunker{}, minScore: math.Inf(-1),
		coord: vector.Coord{Index: index, Partition: partition, Collection: collection},
		model: modelOf(embedder),
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
		return ErrEmptyID
	}
	hash := ContentHash(text)
	stored, err := ix.store.Hashes(ctx, ix.coord, docID)
	if err != nil {
		return fmt.Errorf("search: read content hash: %w", err)
	}
	if h, ok := stored[docID]; ok && h == hash {
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
			if err := vector.Validate(vectors[i], ix.embedder.Dimensions()); err != nil {
				return fmt.Errorf("search: embed document %q chunk %d: %w", docID, i, err)
			}
			chunks = append(chunks, vector.Chunk{Seq: i, Text: part, Vector: vectors[i]})
		}
	}
	if err := ix.store.Upsert(ctx, ix.coord, ix.model, vector.Doc{ID: docID, Hash: hash, Chunks: chunks}); err != nil {
		return fmt.Errorf("search: store document: %w", err)
	}
	return nil
}

// Delete removes a document from the index. Unknown ids are a no-op.
func (ix *Index) Delete(ctx context.Context, docID string) error {
	if docID == "" {
		return ErrEmptyID
	}
	if err := ix.store.Delete(ctx, ix.coord, docID); err != nil {
		return fmt.Errorf("search: delete %q: %w", docID, err)
	}
	return nil
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
	if err := vector.Validate(vectors[0], ix.embedder.Dimensions()); err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	// the store answers one match per document, its best chunk
	raw, err := ix.store.Search(ctx, ix.coord, vector.Query{Vector: vectors[0], Limit: limit, Model: ix.model})
	if err != nil {
		return nil, fmt.Errorf("search: search: %w", err)
	}
	return aboveFloor(raw, ix.minScore), nil
}

// aboveFloor returns the leading matches of raw (best first) that score at
// least min. A NaN score never passes.
func aboveFloor(raw []vector.Match, min float64) []vector.Match {
	for i, m := range raw {
		if !(m.Score >= min) {
			return raw[:i]
		}
	}
	return raw
}

// modelOf returns the model e names, "" when it names none.
func modelOf(e any) string {
	if id, ok := e.(langai.Identified); ok {
		return id.Model()
	}
	return ""
}

// ContentHash is the fingerprint used to detect unchanged documents.
func ContentHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

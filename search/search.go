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
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/vector"
)

var (
	// ErrEmptyID is returned by every call that takes a document id when the
	// id is empty, before any model or store is asked anything. It wraps
	// vector.ErrInvalidID, which an id no store can keep (not valid UTF-8,
	// or holding a NUL byte) is refused with in the same way.
	ErrEmptyID = fmt.Errorf("search: empty document id: %w", vector.ErrInvalidID)
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
	maxQuery int // the most query vectors per search
	coord    vector.Coord
	model    string // the embedder's, when it names it
	settings string // what shapes the vectors besides the text, for hash
}

// Option customizes an Index.
type Option func(*Index)

// WithChunker replaces the default markdown-aware chunker. A chunker with
// settings should implement Fingerprinter, so that changing them indexes
// every document again.
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

// WithMaxQueryVectors sets the most stored chunks SearchIndexed searches
// with, DefaultMaxQueryVectors by default (and when n <= 0): a document of
// more chunks searches with n of them, evenly spaced from the first.
func WithMaxQueryVectors(n int) Option {
	return func(ix *Index) { ix.maxQuery = n }
}

// New returns an Index using the given embedder and store, over the
// collection at index, partition and collection; an empty partition or
// collection is a value like any other, never a wildcard. An empty index is
// vector.ErrEmptyIndex, and a partition or collection that is not valid
// UTF-8 or holds a NUL byte vector.ErrInvalidID. When the embedder
// names its model (langai.Identified), the store records it with the
// vectors written to the collection and refuses another model's while the
// collection holds documents; Reset empties it for a new model.
func New(embedder langai.Embedder, store vector.Store, index, partition, collection string, opts ...Option) (*Index, error) {
	if embedder == nil {
		return nil, errors.New("search: nil embedder")
	}
	if store == nil {
		return nil, errors.New("search: nil store")
	}
	coord := vector.Coord{Index: index, Partition: partition, Collection: collection}
	if err := vector.CheckCoord(coord); err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	// no floor by default: math.Inf(-1) rather than 0, because 0 is a
	// meaningful cutoff on cosine backends and would silently become policy
	ix := &Index{
		embedder: embedder, store: store, chunker: &MarkdownChunker{}, minScore: math.Inf(-1), maxQuery: DefaultMaxQueryVectors,
		coord: coord, model: modelOf(embedder),
	}
	for _, opt := range opts {
		opt(ix)
	}
	if ix.chunker == nil {
		return nil, errors.New("search: nil chunker")
	}
	if ix.maxQuery <= 0 {
		ix.maxQuery = DefaultMaxQueryVectors
	}
	ix.settings = settings("text", fingerprintOf(ix.chunker), ix.model, embedder.Dimensions())
	return ix, nil
}

// hash is the hash Index stores with a document of text: it changes with
// the text, the chunker's settings and the model, so that a change of the
// text or the chunker indexes the document again. A change of a named model
// (langai.Identified) does not get that far: the store refuses the new
// model's vectors while the collection holds the old one's, and IndexBatch
// says so before embedding anything; Reset empties the collection for the
// new model to rebuild. The model is in the hash all the same, so that the
// rebuild never finds a document unchanged.
func (ix *Index) hash(text string) string {
	return stamp(ix.settings, []byte(text))
}

// Index chunks, embeds and stores one document, replacing whatever was
// indexed under docID before. It is a cheap no-op when the text is unchanged
// since the last call; to index many documents, e.g. a full re-index at
// startup, IndexBatch does the same in bounded batches.
func (ix *Index) Index(ctx context.Context, docID, text string) error {
	res, err := ix.IndexBatch(ctx, []Doc{{ID: docID, Text: text}})
	if err != nil {
		return err
	}
	return res.Failed[docID]
}

// Match is one hit of an Index search: the document, its score (cosine, 1
// for the same direction, around 0 for unrelated text) and the chunk of it
// that matched best, by position (Seq) and text, which makes a snippet.
type Match struct {
	DocID string
	Score float64
	Seq   int
	Text  string
}

// Search embeds the query and returns the closest documents, best first, at
// most limit (10 when <= 0) and one Match per document (its best-scoring
// chunk). Hits below the WithMinScore floor are dropped, so a query with
// nothing relevant to find can legitimately return fewer results than limit
// — or none. An empty or blank query returns no results, and so does one
// that embeds to the zero vector, which has no direction to rank by.
func (ix *Index) Search(ctx context.Context, query string, limit int) ([]Match, error) {
	query = strings.TrimSpace(sanitize(query))
	if query == "" {
		return nil, nil
	}
	vectors, err := ix.embedder.Embed(ctx, langai.RoleQuery, []string{query})
	if err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("search: embedder returned %d vectors for 1 query", len(vectors))
	}
	return ix.rank(ctx, vector.Query{Limit: limit, Model: ix.model}, vectors)
}

// SearchIndexed returns the documents most like the already indexed docID,
// best first, at most limit (10 when <= 0), searching with its stored chunk
// vectors instead of embedding its text again: no model runs. It is one
// store search, and a document is scored by its best chunk against any of
// docID's; a document of more chunks than the cap (DefaultMaxQueryVectors,
// see WithMaxQueryVectors) searches with that many of them, evenly spaced,
// so its cost stays bounded. docID itself is never among the hits. An id
// with no stored chunks (unknown, deleted, or empty when indexed) is
// ErrNotIndexed.
func (ix *Index) SearchIndexed(ctx context.Context, docID string, limit int) ([]Match, error) {
	chunks, err := storedVectors(ctx, ix.store, ix.coord, docID)
	if err != nil {
		return nil, err
	}
	chunks = spread(chunks, ix.maxQuery)
	vectors := make([][]float32, len(chunks))
	for i, c := range chunks {
		vectors[i] = c.Vector
	}
	return ix.rank(ctx, vector.Query{Limit: limit, Model: ix.model, Exclude: []string{docID}}, vectors)
}

// rank runs the shared ranking over the index's collection and floor.
func (ix *Index) rank(ctx context.Context, q vector.Query, vectors [][]float32) ([]Match, error) {
	hits, err := rank(ctx, ix.store, ix.coord, q, vectors, ix.embedder.Dimensions(), ix.minScore)
	if err != nil {
		return nil, err
	}
	out := make([]Match, len(hits))
	for i, h := range hits {
		out[i] = Match{DocID: h.DocID, Score: h.Score, Seq: h.Seq, Text: h.Text}
	}
	return out, nil
}

// Package vector stores embedding vectors and finds the closest ones. Store is
// the port, partitioned by an opaque index (the hard isolation boundary) and
// located within it by two opaque coordinates, partition and collection.
// MemStore in this package is a brute-force in-memory Store for tests, tiny
// corpora and evaluation; the production Stores live in subpackages: sqlitevec
// (SQLite with the sqlite-vec extension, requires CGO) and pgvec (Postgres
// with the pgvector extension).
package vector

import (
	"context"
	"math"
)

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
// Chunks returns a document's stored chunks, vectors included, ordered by
// Seq, so a caller can search with what is already indexed instead of
// embedding it again; an unknown document has none (nil, no error).
type Store interface {
	Upsert(ctx context.Context, index, partition, collection, docID, contentHash string, chunks []Chunk) error
	Delete(ctx context.Context, index, partition, collection, docID string) error
	ContentHash(ctx context.Context, index, partition, collection, docID string) (string, error)
	Chunks(ctx context.Context, index, partition, collection, docID string) ([]Chunk, error)
	Search(ctx context.Context, index, partition, collection string, vector []float32, limit int) ([]Match, error)
	SearchScope(ctx context.Context, index string, f ScopeFilter, vector []float32, limit int) ([]Match, error)
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

// Package vector stores embedding vectors and finds the closest ones. Store is
// the port: its documents live in collections, each found by a Coord, an
// opaque index (the hard isolation boundary, e.g. a tenant) and two opaque
// coordinates within it, partition and collection. MemStore in this package
// is a brute-force in-memory Store for tests, tiny corpora and evaluation;
// the production Stores live in subpackages: sqlitevec (SQLite with the
// sqlite-vec extension, requires CGO) and pgvec (Postgres with the pgvector
// extension). vectortest is the conformance suite all three pass.
package vector

import (
	"context"
	"errors"
	"fmt"
	"math"
)

var (
	// ErrDimensions reports a vector whose length is not the store's: a
	// query or chunk from another model, or a store opened with other
	// dimensions than its tables were built with.
	ErrDimensions = errors.New("vector: wrong number of dimensions")
	// ErrInvalidVector reports a vector that cannot be compared by cosine:
	// one holding a NaN or an infinity, or the zero vector, which has no
	// direction.
	ErrInvalidVector = errors.New("vector: invalid vector")
	// ErrModelMismatch reports a write or a search naming another model than
	// the one a collection's vectors were made with.
	ErrModelMismatch = errors.New("vector: collection holds another model's vectors")
)

var errEmptyIndex = errors.New("vector: empty index")

// Coord locates one collection: Index is the hard isolation boundary and must
// not be empty; Partition and Collection place the collection within it. An
// empty Partition or Collection is a value like any other: it matches only
// the empty one, never every one.
type Coord struct {
	Index      string
	Partition  string
	Collection string
}

// String implements fmt.Stringer for error messages.
func (c Coord) String() string {
	return fmt.Sprintf("%q/%q/%q", c.Index, c.Partition, c.Collection)
}

// Chunk is one embedded piece of a document as handed to a Store: its
// position within the document, the text that was embedded, and its vector.
type Chunk struct {
	Seq    int
	Text   string
	Vector []float32
}

// Doc is one document as handed to Upsert: its id, the caller's fingerprint
// of its content, and its chunks. A document without chunks keeps its hash
// and is never found.
type Doc struct {
	ID     string
	Hash   string
	Chunks []Chunk
}

// DocInfo is one document as Docs lists it.
type DocInfo struct {
	ID   string
	Hash string
}

// Query is what Search ranks. By default it ranks the documents of exactly
// the collection Search is given; AllPartitions and AllCollections open that
// coordinate, so the search spans every partition or collection of the
// index. A non-empty Only keeps the documents with those ids, and Exclude
// leaves its ids out (e.g. the document a "more like this" search starts
// from). A non-empty Model must be the model the collections in scope were
// written with (see Store).
type Query struct {
	Vector         []float32
	Limit          int
	AllPartitions  bool
	AllCollections bool
	Only           []string
	Exclude        []string
	Model          string
}

// Match is one search hit: the best-scoring chunk of a document and its
// cosine score. Coord says which collection the document is in, which tells
// apart two documents of one id in a search that spans several.
type Match struct {
	Coord Coord
	DocID string
	Seq   int
	Text  string
	Score float64
}

// Store persists the chunk vectors of documents in collections and ranks
// them against a query vector by cosine. Implementations must never let one
// index's data appear in another's, and must be safe for concurrent use.
//
// Upsert stores docs in c in one atomic step, each replacing what was stored
// under its id; when model is not empty, the first write to a collection
// records it, and a later write or search with another non-empty model fails
// with ErrModelMismatch (every partition of a collection shares its model).
// Delete removes the documents of the given ids, and DeleteIndex every
// collection of an index; unknown ids and indexes are no error. Hashes
// returns the stored hash of each of the given ids it knows. Docs returns
// the documents of c ordered by id (byte by byte), at most limit of them
// (all when limit <= 0) and only those after the id after, so a caller can
// page through a collection, e.g. to find what it deleted while the store
// was away; the number of documents is the length of the full list. Chunks
// returns a document's stored chunks, vectors included, ordered by Seq, so a
// caller can search with what is already indexed instead of embedding it
// again; an unknown document has none.
//
// Search returns the best chunk of each of the q.Limit best-scoring
// documents, in descending score, then by document id, then by Seq. Vectors
// must pass Validate for the store's dimensions, or the call fails with
// ErrDimensions or ErrInvalidVector. Reads of an index never written return
// nothing and create nothing.
type Store interface {
	Upsert(ctx context.Context, c Coord, model string, docs ...Doc) error
	Delete(ctx context.Context, c Coord, ids ...string) error
	DeleteIndex(ctx context.Context, index string) error
	Hashes(ctx context.Context, c Coord, ids ...string) (map[string]string, error)
	Docs(ctx context.Context, c Coord, after string, limit int) ([]DocInfo, error)
	Chunks(ctx context.Context, c Coord, id string) ([]Chunk, error)
	Search(ctx context.Context, c Coord, q Query) ([]Match, error)
}

// Validate reports whether vec can be stored or searched with in a store of
// dims dimensions: it must have exactly dims entries (ErrDimensions), all of
// them finite and not all zero (ErrInvalidVector).
func Validate(vec []float32, dims int) error {
	if len(vec) != dims {
		return fmt.Errorf("%w: %d, want %d", ErrDimensions, len(vec), dims)
	}
	var sum float64
	for i, v := range vec {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Errorf("%w: entry %d is %v", ErrInvalidVector, i, v)
		}
		sum += f * f
	}
	if sum == 0 {
		return fmt.Errorf("%w: zero vector", ErrInvalidVector)
	}
	return nil
}

// CheckWrite checks what every Store's Upsert is handed, the same way in
// each: a non-empty index, non-empty and distinct document ids, and chunk
// vectors that pass Validate for dims. It is for implementations of Store.
func CheckWrite(c Coord, dims int, docs []Doc) error {
	if c.Index == "" {
		return errEmptyIndex
	}
	seen := make(map[string]bool, len(docs))
	for _, d := range docs {
		if d.ID == "" {
			return errors.New("vector: empty document id")
		}
		if seen[d.ID] {
			return fmt.Errorf("vector: document %q appears twice", d.ID)
		}
		seen[d.ID] = true
		for _, ch := range d.Chunks {
			if err := Validate(ch.Vector, dims); err != nil {
				return fmt.Errorf("document %q chunk %d: %w", d.ID, ch.Seq, err)
			}
		}
	}
	return nil
}

// CheckSearch checks what every Store's Search is handed, the same way in
// each: a non-empty index and a query vector that passes Validate for dims.
// It is for implementations of Store.
func CheckSearch(c Coord, dims int, q Query) error {
	if c.Index == "" {
		return errEmptyIndex
	}
	if err := Validate(q.Vector, dims); err != nil {
		return fmt.Errorf("query: %w", err)
	}
	return nil
}

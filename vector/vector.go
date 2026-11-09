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
	"strings"
	"unicode/utf8"
)

var (
	// ErrDimensions reports a vector whose length is not the store's: a
	// query or chunk from another model, or a store opened with other
	// dimensions than its tables were built with.
	ErrDimensions = errors.New("vector: wrong number of dimensions")
	// ErrInvalidVector reports a vector that cannot be compared by cosine:
	// one holding a NaN or an infinity, the zero vector, which has no
	// direction, or one whose float32 norm underflows or overflows (see
	// Validate).
	ErrInvalidVector = errors.New("vector: invalid vector")
	// ErrModelMismatch reports a write or a search naming another model than
	// the one a collection's vectors were made with.
	ErrModelMismatch = errors.New("vector: collection holds another model's vectors")
	// ErrEmptyIndex reports a Coord, or an index handed to DeleteIndex,
	// whose index is empty: the index is the isolation boundary and must be
	// named. Every method of every Store checks it first.
	ErrEmptyIndex = errors.New("vector: empty index")
	// ErrInvalidID reports a document id, a partition, a collection or a
	// model that no Store can keep alike: one that is not valid UTF-8 or
	// holds a NUL byte (a Postgres TEXT column refuses both, and SQLite's
	// JSON id lists turn invalid bytes into U+FFFD), and an empty document
	// id. Every method of every Store checks the strings it is handed
	// before it reads or writes anything (see CheckCoord and CheckID).
	ErrInvalidID = errors.New("vector: invalid id")
)

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
// under its id; when model is not empty, the write records it as the
// collection's model, and while the collection holds documents (in any of
// its partitions, which share one model) a write or search with another
// non-empty model fails with ErrModelMismatch. A collection without
// documents holds no model: the next write records its own, so emptying a
// collection (DeleteCollection) is how it moves to another model. Model
// returns the model of c's collection, "" when it records none or holds no
// documents, so a caller can find a mismatch before it spends time on
// embedding.
//
// Delete removes the documents of the given ids, DeleteCollection every
// document of c (its partition and collection, other partitions of the
// collection untouched), and DeleteIndex every collection of an index;
// unknown ids, collections and indexes are no error. Hashes
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
//
// Every method checks its arguments before it reads or writes anything, the
// same way in every implementation (CheckCoord, CheckID, CheckRead,
// CheckDocs, CheckWrite, CheckSearch): an empty index is ErrEmptyIndex, and
// a document id, partition, collection or model no backend can keep alike
// is ErrInvalidID. An error of the database underneath comes back wrapped
// with "vector:", the backend's name and the operation.
type Store interface {
	Upsert(ctx context.Context, c Coord, model string, docs ...Doc) error
	Delete(ctx context.Context, c Coord, ids ...string) error
	DeleteCollection(ctx context.Context, c Coord) error
	DeleteIndex(ctx context.Context, index string) error
	Hashes(ctx context.Context, c Coord, ids ...string) (map[string]string, error)
	Docs(ctx context.Context, c Coord, after string, limit int) ([]DocInfo, error)
	Chunks(ctx context.Context, c Coord, id string) ([]Chunk, error)
	Model(ctx context.Context, c Coord) (string, error)
	Search(ctx context.Context, c Coord, q Query) ([]Match, error)
}

// Validate reports whether vec can be stored or searched with in a store of
// dims dimensions: it must have exactly dims entries (ErrDimensions), all of
// them finite, and a sum of squares that is neither 0 nor infinite when
// computed in float32, as the SQL backends compute the norm
// (ErrInvalidVector). The zero vector fails so, and so does one too small or
// too large for a float32 norm, such as {1e-25, 1e-25} or {1e20, 1e20},
// which a backend would score as infinite, NaN or 0 whatever its direction.
func Validate(vec []float32, dims int) error {
	if len(vec) != dims {
		return fmt.Errorf("%w: %d, want %d", ErrDimensions, len(vec), dims)
	}
	var sum float32
	for i, v := range vec {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Errorf("%w: entry %d is %v", ErrInvalidVector, i, v)
		}
		// the conversion rounds the square on its own, so it is never fused
		// into a multiply-add of higher precision than the backends'
		sum += float32(v * v)
	}
	switch {
	case sum == 0:
		return fmt.Errorf("%w: zero vector, or its float32 norm underflows to 0", ErrInvalidVector)
	case math.IsInf(float64(sum), 0):
		return fmt.Errorf("%w: its float32 norm overflows", ErrInvalidVector)
	}
	return nil
}

// storable reports whether every Store keeps s as it is: valid UTF-8
// without NUL bytes.
func storable(s string) bool {
	return utf8.ValidString(s) && strings.IndexByte(s, 0) < 0
}

// CheckCoord checks the Coord every method of a Store is handed, the same
// way in each: a non-empty index (ErrEmptyIndex), and a partition and a
// collection that are valid UTF-8 without NUL bytes (ErrInvalidID). The
// index itself may hold any bytes: the SQL stores hash it into their table
// names and never store it. It is for implementations of Store.
func CheckCoord(c Coord) error {
	if c.Index == "" {
		return ErrEmptyIndex
	}
	if !storable(c.Partition) {
		return fmt.Errorf("%w: partition %q is not valid UTF-8 or holds a NUL byte", ErrInvalidID, c.Partition)
	}
	if !storable(c.Collection) {
		return fmt.Errorf("%w: collection %q is not valid UTF-8 or holds a NUL byte", ErrInvalidID, c.Collection)
	}
	return nil
}

// CheckID checks a document id a Store is handed: not empty, valid UTF-8
// and without NUL bytes (ErrInvalidID). It is for implementations of Store
// and for callers that want to refuse such an id before they do any work
// for it.
func CheckID(id string) error {
	if id == "" {
		return fmt.Errorf("%w: empty document id", ErrInvalidID)
	}
	if !storable(id) {
		return fmt.Errorf("%w: document id %q is not valid UTF-8 or holds a NUL byte", ErrInvalidID, id)
	}
	return nil
}

// CheckRead checks what a Store's reads and deletes are handed: c passes
// CheckCoord and every id passes CheckID. It is for implementations of
// Store.
func CheckRead(c Coord, ids ...string) error {
	if err := CheckCoord(c); err != nil {
		return err
	}
	for _, id := range ids {
		if err := CheckID(id); err != nil {
			return err
		}
	}
	return nil
}

// CheckDocs checks what a Store's Docs is handed: c passes CheckCoord and
// after is empty (the first page) or passes CheckID. It is for
// implementations of Store.
func CheckDocs(c Coord, after string) error {
	if err := CheckCoord(c); err != nil {
		return err
	}
	if after == "" {
		return nil
	}
	return CheckID(after)
}

// checkModel checks a model name: valid UTF-8 without NUL bytes, empty
// allowed (no model).
func checkModel(model string) error {
	if !storable(model) {
		return fmt.Errorf("%w: model %q is not valid UTF-8 or holds a NUL byte", ErrInvalidID, model)
	}
	return nil
}

// CheckWrite checks what every Store's Upsert is handed, the same way in
// each: a Coord that passes CheckCoord, a model that is valid UTF-8 without
// NUL bytes, distinct document ids that pass CheckID, and chunk vectors
// that pass Validate for dims. It is for implementations of Store.
func CheckWrite(c Coord, model string, dims int, docs []Doc) error {
	if err := CheckCoord(c); err != nil {
		return err
	}
	if err := checkModel(model); err != nil {
		return err
	}
	seen := make(map[string]bool, len(docs))
	for _, d := range docs {
		if err := CheckID(d.ID); err != nil {
			return err
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
// each: a Coord that passes CheckCoord, a model and Only and Exclude ids
// as CheckWrite and CheckID take them, and a query vector that passes
// Validate for dims. It is for implementations of Store.
func CheckSearch(c Coord, dims int, q Query) error {
	if err := CheckRead(c, q.Only...); err != nil {
		return err
	}
	if err := CheckRead(c, q.Exclude...); err != nil {
		return err
	}
	if err := checkModel(q.Model); err != nil {
		return err
	}
	if err := Validate(q.Vector, dims); err != nil {
		return fmt.Errorf("query: %w", err)
	}
	return nil
}

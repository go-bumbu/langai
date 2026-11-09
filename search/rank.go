package search

import (
	"context"
	"fmt"

	"github.com/go-bumbu/langai/vector"
)

// defaultLimit is the number of hits a search returns when the caller asks
// for none (limit <= 0).
const defaultLimit = 10

// DefaultMaxQueryVectors is the most query vectors one search hands the
// store: a "more like this" search (SearchIndexed) of a document of more
// chunks, or of a video of more frames or segments, and a search by a video
// of more frames (SearchVideo, SearchFrames), search with that many of
// them, evenly spaced from the first, so that the cost of a search stays
// bounded whatever the length of what it starts from. The store scores
// every chunk in scope against every query vector, so a search costs about
// that many times one of a single vector. WithMaxQueryVectors and
// WithMediaMaxQueryVectors change it.
const DefaultMaxQueryVectors = 64

// spread returns at most n of items, evenly spaced from the first, in
// order; all of them when n <= 0 or n is not below their number.
func spread[T any](items []T, n int) []T {
	if n <= 0 || len(items) <= n {
		return items
	}
	out := make([]T, n)
	for i := range out {
		out[i] = items[i*len(items)/n]
	}
	return out
}

// rank searches the store once with all of vectors, the query shaped like
// base, and returns the best match of each of the best documents: best
// first, then by document id, at most base.Limit of them (defaultLimit
// when <= 0), those under minScore left out. A document scores its best
// chunk against any of the vectors, and Match.Query is the index in vectors
// of the one that scored it (the first on a tie); the store answers one
// match per document, so the many chunks of one document never crowd the
// others out.
//
// A zero vector, which has no direction to rank by, is left out, so a query
// that embeds to nothing but zeros finds nothing, as an empty one does; any
// other vector the store would refuse (of other than dims entries, NaN or
// infinite) is an error.
func rank(ctx context.Context, store vector.Store, c vector.Coord, base vector.Query, vectors [][]float32, dims int, minScore float64) ([]vector.Match, error) {
	if base.Limit <= 0 {
		base.Limit = defaultLimit
	}
	kept := make([][]float32, 0, len(vectors))
	index := make([]int, 0, len(vectors)) // kept[i] is vectors[index[i]]
	for i, v := range vectors {
		if isZero(v) {
			continue
		}
		if err := vector.Validate(v, dims); err != nil {
			return nil, fmt.Errorf("search: query vector %d: %w", i, err)
		}
		kept = append(kept, v)
		index = append(index, i)
	}
	if len(kept) == 0 {
		return nil, nil
	}
	base.Vectors = kept
	raw, err := store.Search(ctx, c, base)
	if err != nil {
		return nil, fmt.Errorf("search: search: %w", err)
	}
	out := aboveFloor(raw, minScore)
	for i, m := range out {
		if m.Query < 0 || m.Query >= len(index) {
			return nil, fmt.Errorf("search: the store named query vector %d of %d", m.Query, len(index))
		}
		out[i].Query = index[m.Query]
	}
	return out, nil
}

// isZero reports whether v has entries and all of them are zero.
func isZero(v []float32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return len(v) > 0
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

// checkID refuses an id before any work is done for it: ErrEmptyID for an
// empty one, vector.ErrInvalidID for one no store can keep.
func checkID(id string) error {
	if id == "" {
		return ErrEmptyID
	}
	if err := vector.CheckID(id); err != nil {
		return fmt.Errorf("search: %w", err)
	}
	return nil
}

// storedVectors returns the vectors stored for docID, for a "more like this"
// search: ErrEmptyID for an empty id, ErrNotIndexed when there are none.
func storedVectors(ctx context.Context, store vector.Store, c vector.Coord, docID string) ([]vector.Chunk, error) {
	if err := checkID(docID); err != nil {
		return nil, err
	}
	chunks, err := store.Chunks(ctx, c, docID)
	if err != nil {
		return nil, fmt.Errorf("search: read stored vectors of %q: %w", docID, err)
	}
	if len(chunks) == 0 {
		return nil, fmt.Errorf("%w: %q", ErrNotIndexed, docID)
	}
	return chunks, nil
}

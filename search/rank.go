package search

import (
	"cmp"
	"context"
	"fmt"
	"slices"

	"github.com/go-bumbu/langai/vector"
)

// defaultLimit is the number of hits a search returns when the caller asks
// for none (limit <= 0).
const defaultLimit = 10

// hit is one document's best match over several query vectors, and the
// index of the query vector that found it.
type hit struct {
	match vector.Match
	query int
}

// rank searches the store with each of vectors, every search shaped like
// base, and keeps each document's best match: best first, then by document
// id, at most base.Limit of them (defaultLimit when <= 0), those under
// minScore left out. Each search asks for base.Limit documents and the store
// answers one match per document, so the many chunks of one document never
// crowd the others out. On a tie the earlier query vector wins.
//
// A zero vector, which has no direction to rank by, is skipped, so a query
// that embeds to nothing but zeros finds nothing, as an empty one does; any
// other vector the store would refuse (of other than dims entries, NaN or
// infinite) is an error.
func rank(ctx context.Context, store vector.Store, c vector.Coord, base vector.Query, vectors [][]float32, dims int, minScore float64) ([]hit, error) {
	if base.Limit <= 0 {
		base.Limit = defaultLimit
	}
	best := make(map[string]hit)
	for i, v := range vectors {
		if isZero(v) {
			continue
		}
		if err := vector.Validate(v, dims); err != nil {
			return nil, fmt.Errorf("search: query vector %d: %w", i, err)
		}
		q := base
		q.Vector = v
		raw, err := store.Search(ctx, c, q)
		if err != nil {
			return nil, fmt.Errorf("search: search: %w", err)
		}
		for _, m := range aboveFloor(raw, minScore) {
			if cur, ok := best[m.DocID]; ok && cur.match.Score >= m.Score {
				continue
			}
			best[m.DocID] = hit{match: m, query: i}
		}
	}
	out := make([]hit, 0, len(best))
	for _, h := range best {
		out = append(out, h)
	}
	slices.SortFunc(out, func(a, b hit) int {
		return cmp.Or(cmp.Compare(b.match.Score, a.match.Score), cmp.Compare(a.match.DocID, b.match.DocID))
	})
	if len(out) > base.Limit {
		out = out[:base.Limit]
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

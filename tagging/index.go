package tagging

import "context"

// TagVector is a tag as a TagIndex holds it: built from the catalog by the
// Tagger, never written by the application.
type TagVector struct {
	ID, Name    string // Name: for path shadowing
	Fingerprint string // what Vec and Cutoff were built from (see engine)
	Vec         []float32
	Cutoff      float64
}

// TagIndex holds each space's tag vectors and picks the ones worth scoring. A
// space is one (index, partition) pair; an index is never empty, a partition
// may be.
type TagIndex interface {
	// State returns the space's state and the fingerprint of every tag held,
	// by id; ok is false, and the state "", when the index knows nothing of the
	// space: it holds nothing and no state was recorded, or it was evicted. A
	// known space's state is never "": it is the one recorded by the last Apply
	// that set one or, when an Apply could not record its own, a value of the
	// index's choosing that no caller records, so every caller takes the space
	// for stale and passes that value on as its from. The map may be shared:
	// callers must not modify it.
	State(ctx context.Context, index, partition string) (state string, fingerprints map[string]string, ok bool, err error)
	// Apply upserts put and removes drop; an id in both ends up held: put wins.
	// from is the state the caller diffed against: what State reported, "" when
	// it reported ok == false. Apply records state, unless it is "", only if
	// from is still the space's state, so the space holds what the caller read
	// plus these changes. Otherwise the space was evicted and recreated, or
	// another Apply moved its state, so it holds something no caller built:
	// Apply still takes put and drop, but leaves the space under a state of its
	// own choosing, and the next call diffs again. A state of "" records nothing
	// and, when from matches, leaves the recorded state as it is.
	Apply(ctx context.Context, index, partition, from, state string, put []TagVector, drop []string) error
	// Candidates returns the tags to score vec against: a text's vector, or
	// for a text in several pieces (see TaggerConfig.ChunkBytes), the mean of
	// theirs, which every piece is then scored against the candidates by. The
	// slice and the vectors in it are shared: callers must not modify them.
	Candidates(ctx context.Context, index, partition string, vec []float32) ([]TagVector, error)
}

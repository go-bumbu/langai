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
	// State returns the state recorded by the last Apply that set one, and
	// the fingerprint of every tag held, by id; ok is false when the space
	// holds nothing and no state was recorded.
	State(ctx context.Context, index, partition string) (state string, fingerprints map[string]string, ok bool, err error)
	// Apply upserts put, removes drop, and records state unless it is "".
	// An id in both put and drop ends up held: put wins.
	Apply(ctx context.Context, index, partition, state string, put []TagVector, drop []string) error
	// Candidates returns the tags to score vec against. The slices are
	// shared: callers must not modify them.
	Candidates(ctx context.Context, index, partition string, vec []float32) ([]TagVector, error)
}

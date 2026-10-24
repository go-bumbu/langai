package autotag

import (
	"cmp"
	"math"
	"slices"

	"github.com/go-bumbu/langai/semsearch"
)

// centroid returns the unit-length mean of vecs (dims long), or nil when
// there are none.
func centroid(vecs [][]float32, dims int) []float32 {
	if len(vecs) == 0 {
		return nil
	}
	c := make([]float32, dims)
	for _, m := range vecs {
		for i := range c {
			c[i] += m[i]
		}
	}
	semsearch.Normalize(c)
	return c
}

// blend combines a tag's name-embedding prior with the centroid of its
// members using Bayesian-style smoothing: with n members and smoothing
// constant k, the centroid's weight is n/(n+k), so a tag with zero members
// tags purely by name, and its vector gradually shifts toward the member
// centroid as members accumulate.
func blend(nameVec []float32, memberVecs [][]float32, k float64) []float32 {
	out := make([]float32, len(nameVec))
	copy(out, nameVec)
	semsearch.Normalize(out)
	n := float64(len(memberVecs))
	if n == 0 {
		return out
	}
	c := centroid(memberVecs, len(nameVec))
	w := n / (n + k)
	for i := range out {
		out[i] = float32((1-w)*float64(out[i]) + w*float64(c[i]))
	}
	semsearch.Normalize(out)
	return out
}

// tagCutoff is the score a document must reach against a tag's blended
// vector (see blend) to be assigned that tag. The name prior and the member
// centroid score on different scales: a document is never very close to a
// short tag path, but always fairly close to other documents. So each part is
// judged against its own minimum (nameMin, memberMin), and the two are mixed
// with the blend's weight w = n/(n+k). The blended vector is renormalized, so
// the mixed minimum is divided by the length the blend had before that.
// Comparing a document's cosine to the blended vector with this cutoff is then
// exactly (1-w)·(cos(name) − nameMin) + w·(cos(centroid) − memberMin) ≥ 0.
func tagCutoff(nameVec []float32, memberVecs [][]float32, k, nameMin, memberMin float64) float64 {
	n := float64(len(memberVecs))
	if n == 0 {
		return nameMin
	}
	name := make([]float32, len(nameVec))
	copy(name, nameVec)
	semsearch.Normalize(name)
	c := centroid(memberVecs, len(nameVec))
	w := n / (n + k)
	var norm2 float64
	for i := range name {
		x := (1-w)*float64(name[i]) + w*float64(c[i])
		norm2 += x * x
	}
	if norm2 == 0 {
		return math.Inf(1) // name and centroid cancel out: nothing can match
	}
	return ((1-w)*nameMin + w*memberMin) / math.Sqrt(norm2)
}

// never is the cutoff of a tag that no score can reach. A cosine is at most 1,
// so 2 is out of reach, and unlike the +Inf or NaN that tagCutoff can return,
// it is a number every store can keep.
const never = 2.0

// storedCutoff returns the cutoff to store for a tag whose tagCutoff is c: c
// itself when a score can reach it, and never when none can. That is any c
// above 1, +Inf (the name and the centroid cancel out) among them, and a NaN
// (a NaN minimum score), which is why the test is "not at or below 1" and not
// "above 1": every comparison with a NaN is false.
func storedCutoff(c float64) float64 {
	if !(c <= 1) {
		return never
	}
	return c
}

// cosine returns the cosine of the angle between a and b, or 0 when their
// lengths differ or either is the zero vector. It divides by both lengths
// instead of taking the vectors to be unit length, which a normalized float32
// vector only nearly is.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// rank returns the ids of the tags for which vec, a document's embedding,
// scores at least the tag's own cutoff, best score first and equal scores by
// id, with no limit on how many. Each tag has its own cutoff, so a tag below
// its cutoff says nothing about the next: every tag is checked. A tag with no
// vector yet, or one built under another model than model, has nothing to
// compare vec with and is skipped. It fails closed: a NaN score or cutoff picks
// nothing. rank sorts a slice of its own, so tags keeps its order and its
// vectors.
func rank(vec []float32, tags []TagRecord, model string) []string {
	type pick struct {
		id    string
		score float64
	}
	var picks []pick
	for _, tg := range tags {
		if tg.Vec == nil || tg.Model != model {
			continue
		}
		score := cosine(vec, tg.Vec)
		// "not at or above" and not "below": every comparison with a NaN is false,
		// and with no limit on how many tags are picked a NaN reaching here would
		// pick every one
		if !(score >= tg.Cutoff) {
			continue
		}
		picks = append(picks, pick{id: tg.ID, score: score})
	}
	slices.SortStableFunc(picks, func(a, b pick) int {
		return cmp.Or(cmp.Compare(b.score, a.score), cmp.Compare(a.id, b.id))
	})
	var ids []string
	for _, p := range picks {
		ids = append(ids, p.id)
	}
	return ids
}

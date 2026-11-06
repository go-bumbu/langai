package tagging

import (
	"cmp"
	"math"
	"slices"
	"strings"

	"github.com/go-bumbu/langai/vector"
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
	vector.Normalize(c)
	return c
}

// memberMean returns the mean of vecs (dims long), each made unit length
// first, and not normalized after: a document's dot product with it is the
// mean of its cosines to the members. Unlike the cosine to their centroid,
// that mean does not grow with the number of members (the centroid of many
// members sits closer to everything they share), so one minimum holds for a
// tag with two members and for one with fifty. It returns nil when there are
// none.
func memberMean(vecs [][]float32, dims int) []float64 {
	if len(vecs) == 0 {
		return nil
	}
	sum := make([]float64, dims)
	for _, m := range vecs {
		for i, x := range unit64(m) {
			sum[i] += x
		}
	}
	for i := range sum {
		sum[i] /= float64(len(vecs))
	}
	return sum
}

// unit64 returns v made unit length, in float64; the zero vector stays zero.
func unit64(v []float32) []float64 {
	out := make([]float64, len(v))
	var n2 float64
	for i, x := range v {
		out[i] = float64(x)
		n2 += out[i] * out[i]
	}
	if n2 == 0 {
		return out
	}
	inv := 1 / math.Sqrt(n2)
	for i := range out {
		out[i] *= inv
	}
	return out
}

// mixed is (1-w)·name + w·mean, the name made unit length, before any
// normalization, for a tag with n members and smoothing constant k
// (w = n/(n+k)); with no members it is the unit name.
func mixed(nameVec []float32, memberVecs [][]float32, k float64) []float64 {
	out := unit64(nameVec)
	n := float64(len(memberVecs))
	if n == 0 {
		return out
	}
	mean := memberMean(memberVecs, len(nameVec))
	w := n / (n + k)
	for i := range out {
		out[i] = (1-w)*out[i] + w*mean[i]
	}
	return out
}

// blend combines a tag's name-embedding prior with its members using
// Bayesian-style smoothing: with n members and smoothing constant k, the
// members' weight is n/(n+k), so a tag with zero members tags purely by name,
// and its vector gradually shifts toward its members as they accumulate. The
// members enter as their mean (see memberMean), and the blend is made unit
// length.
func blend(nameVec []float32, memberVecs [][]float32, k float64) []float32 {
	m := mixed(nameVec, memberVecs, k)
	out := make([]float32, len(m))
	for i, x := range m {
		out[i] = float32(x)
	}
	vector.Normalize(out)
	return out
}

// tagCutoff is the score a document must reach against a tag's blended
// vector (see blend) to be assigned that tag. The name prior and the members
// score on different scales: a document is never very close to a short tag
// path, but always fairly close to other documents. So each part is judged
// against its own minimum (nameMin for the cosine to the name, memberMin for
// the mean cosine to the members), and the two are mixed with the blend's
// weight w = n/(n+k). The blended vector is normalized, so the mixed minimum is
// divided by the length the blend had before that. Comparing a unit document
// vector's cosine to the blended vector with this cutoff is then exactly
// (1-w)·(cos(name) − nameMin) + w·(mean cos(member) − memberMin) ≥ 0.
func tagCutoff(nameVec []float32, memberVecs [][]float32, k, nameMin, memberMin float64) float64 {
	n := float64(len(memberVecs))
	if n == 0 {
		return nameMin
	}
	var norm2 float64
	for _, x := range mixed(nameVec, memberVecs, k) {
		norm2 += x * x
	}
	if norm2 == 0 {
		return math.Inf(1) // name and members cancel out: nothing can match
	}
	w := n / (n + k)
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
//
// With onePath, of the tags that reach their cutoff on one path of the tag
// tree, only the one that clears it by most is picked (see shadowOnPaths).
func rank(vec []float32, tags []TagRecord, model string, onePath bool) []string {
	var ids []string
	for _, s := range scores(vec, tags, model, onePath) {
		if s.Picked() {
			ids = append(ids, s.ID)
		}
	}
	return ids
}

// scores returns vec's score against every tag that has a vector of model,
// as scoreVectors does: a tag built under another model is skipped.
func scores(vec []float32, tags []TagRecord, model string, onePath bool) []TagScore {
	var vecs []TagVector
	for _, tg := range tags {
		if tg.Model == model {
			vecs = append(vecs, TagVector{ID: tg.ID, Name: tg.Name, Vec: tg.Vec, Cutoff: tg.Cutoff})
		}
	}
	return scoreVectors(vec, vecs, onePath)
}

// scoreVectors returns vec's score against every tag that has a vector, with
// the tag's cutoff, best score first and equal scores by id. A NaN score
// sorts last. With onePath, a tag that reaches its cutoff but is shadowed on
// its path says by which tag (see shadowOnPaths).
func scoreVectors(vec []float32, tags []TagVector, onePath bool) []TagScore {
	var out []TagScore
	paths := map[string][]string{}
	for _, tg := range tags {
		if tg.Vec == nil {
			continue
		}
		out = append(out, TagScore{ID: tg.ID, Score: cosine(vec, tg.Vec), Cutoff: tg.Cutoff})
		paths[tg.ID] = tagPath(tg.Name)
	}
	slices.SortStableFunc(out, func(a, b TagScore) int {
		return cmp.Or(cmp.Compare(b.Score, a.Score), cmp.Compare(a.ID, b.ID))
	})
	if onePath {
		shadowOnPaths(out, paths)
	}
	return out
}

// tagPath splits a tag name into the levels of its path, "dev / go" into
// "dev" and "go", each trimmed and lowercased; empty levels are dropped.
func tagPath(name string) []string {
	var levels []string
	for l := range strings.SplitSeq(name, "/") {
		if l = strings.ToLower(strings.TrimSpace(l)); l != "" {
			levels = append(levels, l)
		}
	}
	return levels
}

// onPath reports whether one of a and b is an ancestor of the other: its
// levels are the start of the other's, which has more.
func onPath(a, b []string) bool {
	if len(a) == len(b) {
		return false
	}
	if len(a) > len(b) {
		a, b = b, a
	}
	return slices.Equal(a, b[:len(a)])
}

// shadowOnPaths compares, among the tags that reach their cutoff, every
// ancestor with every descendant by margin (score minus cutoff), and marks the
// one of each pair that loses with ShadowedBy, naming the winner with the
// widest margin: a descendant wins with a margin at least the ancestor's, an
// ancestor with a wider one.
//
// A path implies its ancestors, so "dev" next to "dev / go" adds nothing, and
// an item about Go clears "dev / go" by more than "dev". A child has to earn
// its place the other way too: when "dev" fits an item better than "dev /
// web" does, the item is about programming in general and "dev / web" is an
// incidental match. Two children that both fit better than their parent are
// both picked, and tags on different paths do not meet. The tag with the
// widest margin of a family always stays (on a tie with its descendant, the
// deepest tied one).
func shadowOnPaths(out []TagScore, paths map[string][]string) {
	var picked []int
	for i, s := range out {
		if s.Picked() {
			picked = append(picked, i)
		}
	}
	margin := func(i int) float64 { return out[i].Score - out[i].Cutoff }
	by := map[int]int{} // loser -> winner with the widest margin
	for _, a := range picked {
		for _, d := range picked {
			pa, pd := paths[out[a].ID], paths[out[d].ID]
			if len(pa) >= len(pd) || !onPath(pa, pd) {
				continue // a is not an ancestor of d
			}
			loser, winner := a, d
			if margin(a) > margin(d) {
				loser, winner = d, a
			}
			if w, ok := by[loser]; !ok || margin(winner) > margin(w) {
				by[loser] = winner
			}
		}
	}
	for loser, winner := range by {
		out[loser].ShadowedBy = out[winner].ID
	}
}

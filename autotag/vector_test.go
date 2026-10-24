package autotag

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// TestBlend_ShiftsFromNamePriorToMemberCentroid pins the smoothing contract:
// with k members' worth of smoothing, one member barely moves the vector off
// the name prior, while many members pull it onto the centroid.
func TestBlend_ShiftsFromNamePriorToMemberCentroid(t *testing.T) {
	const k = 3.0
	name := []float32{1, 0}
	member := []float32{0, 1}

	noMembers := blend(name, nil, k)
	if noMembers[0] != 1 || noMembers[1] != 0 {
		t.Fatalf("with no members the blend must be the name prior, got %v", noMembers)
	}

	// n=1: centroid weight 1/(1+3) = 0.25, so the name still dominates
	one := blend(name, [][]float32{member}, k)
	if one[0] <= one[1] {
		t.Fatalf("one member should stay name-dominated, got %v", one)
	}
	// n=9: centroid weight 9/(9+3) = 0.75, so the centroid dominates
	many := make([][]float32, 9)
	for i := range many {
		many[i] = member
	}
	nine := blend(name, many, k)
	if nine[1] <= nine[0] {
		t.Fatalf("nine members should be centroid-dominated, got %v", nine)
	}
	// more members must move strictly further toward the centroid
	if nine[1] <= one[1] {
		t.Fatalf("centroid weight did not grow with members: one=%v nine=%v", one, nine)
	}
	for _, v := range [][]float32{noMembers, one, nine} {
		if l := length(v); math.Abs(l-1) > 1e-6 {
			t.Fatalf("blend must return a unit vector, got length %f for %v", l, v)
		}
	}
}

func TestTagCutoff_NoMembersIsTheNameMinimum(t *testing.T) {
	if got := tagCutoff(unitVec(1, 2, 3), nil, 3, 0.46, 0.70); got != 0.46 {
		t.Fatalf("cutoff = %v, want 0.46", got)
	}
}

// Comparing a document against the blended vector with tagCutoff must decide
// exactly like judging the name against nameMin and the members' centroid
// against memberMin, mixed by the blend weight.
func TestTagCutoff_MatchesTheTwoScaleRule(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	const dims, k, a, b = 16, 3.0, 0.46, 0.70
	for n := 1; n <= 6; n++ {
		name := randUnit(r, dims)
		members := make([][]float32, n)
		for i := range members {
			members[i] = randUnit(r, dims)
		}
		vec := blend(name, members, k)
		cut := tagCutoff(name, members, k, a, b)
		c := centroid(members, dims)
		w := float64(n) / (float64(n) + k)
		// queries from far away to right on top of the tag, so both
		// outcomes are exercised
		for step := 0; step <= 200; step++ {
			beta := float64(step) / 200
			noise := randUnit(r, dims)
			mix := make([]float64, dims)
			for i := range mix {
				mix[i] = beta*float64(vec[i]) + (1-beta)*float64(noise[i])
			}
			q := unitVec(mix...)
			margin := (1-w)*(dot(q, name)-a) + w*(dot(q, c)-b)
			if math.Abs(margin) < 1e-5 {
				continue // float32 rounding decides the tie
			}
			if got, want := dot(q, vec) >= cut, margin >= 0; got != want {
				t.Fatalf("n=%d: blended %.5f vs cutoff %.5f says %v, two-scale margin %.5f says %v",
					n, dot(q, vec), cut, got, margin, want)
			}
		}
	}
}

func TestTagCutoff_ApproachesTheMemberMinimumWithManyMembers(t *testing.T) {
	m := unitVec(1, 1, 0) // 45° away from the name
	members := make([][]float32, 1000)
	for i := range members {
		members[i] = m
	}
	if got := tagCutoff(unitVec(1, 0, 0), members, 3, 0.46, 0.70); math.Abs(got-0.70) > 0.01 {
		t.Fatalf("cutoff with 1000 members = %.4f, want about 0.70", got)
	}
}

// TestTagCutoff_OpposedNameAndCentroidMatchNothing: a centroid pointing
// exactly away from the name cancels it out at equal weights, and a blend of
// nothing has no direction any document could score against.
func TestTagCutoff_OpposedNameAndCentroidMatchNothing(t *testing.T) {
	members := [][]float32{unitVec(-1, 0), unitVec(-1, 0), unitVec(-1, 0)}
	if got := tagCutoff(unitVec(1, 0), members, 3, 0.46, 0.70); !math.IsInf(got, 1) {
		t.Fatalf("cutoff = %v, want +Inf", got)
	}
}

func TestStoredCutoff_KeepsScoresAndNeverStoresWhatNoScoreReaches(t *testing.T) {
	for in, want := range map[float64]float64{0.46: 0.46, 1: 1, 1.2: never, math.Inf(1): never, -0.1: -0.1} {
		if got := storedCutoff(in); got != want {
			t.Errorf("storedCutoff(%v) = %v, want %v", in, got, want)
		}
	}
	if got := storedCutoff(math.NaN()); got != never {
		t.Errorf("storedCutoff(NaN) = %v, want never", got)
	}
}

func TestCosine_IsZeroForMismatchedOrZeroVectors(t *testing.T) {
	if cosine([]float32{1, 0}, []float32{1, 0, 0}) != 0 || cosine([]float32{0, 0}, []float32{1, 0}) != 0 {
		t.Fatal("mismatched or zero vectors must score 0")
	}
	if got := cosine([]float32{2, 0}, []float32{1, 1}); math.Abs(got-math.Sqrt2/2) > 1e-6 {
		t.Fatalf("cosine = %v, want %v (not unit-length inputs)", got, math.Sqrt2/2)
	}
}

// rank judges each tag against its own cutoff: a tag below its cutoff does not
// hide the ones after it, a score equal to the cutoff qualifies, and nothing
// limits how many tags come back. They come best score first and equal scores
// by id, whatever order the tags were given in, and that order is left as it
// was. A tag with no vector, or one built under another model, has nothing to
// compare and is skipped. It fails closed: no score reaches a NaN cutoff,
// though every comparison with a NaN is false.
func TestRank_JudgesEachTagAgainstItsOwnCutoff(t *testing.T) {
	const dims = 4
	q := unitVec(1, 0, 0, 0)
	// tag is a tag of model "m" whose vector scores about score against q: the
	// vector is a float32, so its exact cosine is a few 1e-8 off, which only a
	// case sitting right on its cutoff would notice.
	tag := func(id string, axis int, score, cutoff float64) TagRecord {
		return TagRecord{ID: id, Vec: scoreVec(dims, axis, score), Cutoff: cutoff, Model: "m"}
	}
	// atCutoff takes its own exact cosine for its cutoff, so its score equals it
	atCutoff := tag("2", 2, 0.50, 0)
	atCutoff.Cutoff = cosine(q, atCutoff.Vec)
	ids := func(tags []TagRecord) []string {
		out := make([]string, len(tags))
		for i, tg := range tags {
			out[i] = tg.ID
		}
		return out
	}
	for _, tc := range []struct {
		name string
		tags []TagRecord
		want []string
	}{
		{"no tags", nil, nil},
		{"a tag below its cutoff does not hide the ones after it",
			[]TagRecord{tag("1", 1, 0.65, 0.70), tag("2", 2, 0.55, 0.50)}, []string{"2"}},
		{"a score equal to the cutoff qualifies",
			[]TagRecord{atCutoff}, []string{"2"}},
		{"no score reaches a NaN cutoff",
			[]TagRecord{tag("8", 1, 1, math.NaN())}, nil},
		// its zero score would reach its cutoff of 0, were it not skipped
		{"a tag without a vector is skipped",
			[]TagRecord{{ID: "1", Cutoff: 0, Model: "m"}, tag("2", 2, 0.80, 0.46)}, []string{"2"}},
		{"a tag built under another model is skipped",
			[]TagRecord{{ID: "1", Vec: scoreVec(dims, 1, 0.90), Cutoff: 0.46, Model: "other"}, tag("2", 2, 0.80, 0.46)},
			[]string{"2"}},
		{"every tag that clears its cutoff is returned, best score first",
			[]TagRecord{
				tag("1", 1, 0.75, 0.46), tag("2", 2, 0.60, 0.46), tag("3", 3, 0.90, 0.46), tag("4", 1, 0.70, 0.46),
				tag("5", 2, 0.85, 0.46), tag("6", 3, 0.65, 0.46), tag("7", 1, 0.80, 0.46),
			},
			[]string{"3", "5", "7", "1", "4", "6", "2"}},
		{"equal scores come back ordered by id",
			[]TagRecord{tag("b", 1, 0.80, 0.46), tag("a", 2, 0.80, 0.46)}, []string{"a", "b"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			given := ids(tc.tags)
			got := rank(q, tc.tags, "m")
			// nil, not just empty, when nothing qualifies
			if !slices.Equal(got, tc.want) || (got == nil) != (tc.want == nil) {
				t.Fatalf("rank = %#v, want %#v", got, tc.want)
			}
			if after := ids(tc.tags); !slices.Equal(after, given) {
				t.Fatalf("rank reordered the caller's tags: %v, was %v", after, given)
			}
		})
	}
}

// length returns the Euclidean length of v.
func length(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func unitVec(v ...float64) []float32 {
	var n float64
	for _, x := range v {
		n += x * x
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x / math.Sqrt(n))
	}
	return out
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func randUnit(r *rand.Rand, dims int) []float32 {
	v := make([]float64, dims)
	for i := range v {
		v[i] = r.NormFloat64()
	}
	return unitVec(v...)
}

// scoreVec returns a unit vector whose cosine with the first axis, where the
// tests put the document, is exactly score. The rest of its length goes on
// axis, so vectors on different axes have nothing in common but the document.
func scoreVec(dims, axis int, score float64) []float32 {
	v := make([]float64, dims)
	v[0], v[axis] = score, math.Sqrt(1-score*score)
	return unitVec(v...)
}

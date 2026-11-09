package tagging

import (
	"maps"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// TestBlend_ShiftsFromNamePriorToTheMembers pins the smoothing contract:
// with k members' worth of smoothing, one member barely moves the vector off
// the name prior, while many members pull it onto them.
func TestBlend_ShiftsFromNamePriorToTheMembers(t *testing.T) {
	const k = 3.0
	name := []float32{1, 0}
	member := []float32{0, 1}

	noMembers := blend(name, nil, k)
	if noMembers[0] != 1 || noMembers[1] != 0 {
		t.Fatalf("with no members the blend must be the name prior, got %v", noMembers)
	}

	// n=1: members' weight 1/(1+3) = 0.25, so the name still dominates
	one := blend(name, [][]float32{member}, k)
	if one[0] <= one[1] {
		t.Fatalf("one member should stay name-dominated, got %v", one)
	}
	// n=9: members' weight 9/(9+3) = 0.75, so they dominate
	many := make([][]float32, 9)
	for i := range many {
		many[i] = member
	}
	nine := blend(name, many, k)
	if nine[1] <= nine[0] {
		t.Fatalf("nine members should dominate, got %v", nine)
	}
	// more members must move strictly further toward them
	if nine[1] <= one[1] {
		t.Fatalf("members' weight did not grow with them: one=%v nine=%v", one, nine)
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
// exactly like judging the name against nameMin and the mean of the cosines to
// the members against memberMin, mixed by the blend weight.
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
			var mean float64
			for _, m := range members {
				mean += dot(q, m) / float64(n)
			}
			margin := (1-w)*(dot(q, name)-a) + w*(mean-b)
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

// TestTagCutoff_MembersCountAsTheMeanOfTheirCosines: a document that is one of
// two orthogonal members scores 1 against it and 0 against the other, a mean
// of 0.5, below a member minimum of 0.6, though its cosine to the two
// members' centroid is 0.71. With many members of each kind the cutoff is
// all but the members' one, and the document stays below it: the mean, unlike
// the centroid, is no closer to a member for there being more of them.
func TestTagCutoff_MembersCountAsTheMeanOfTheirCosines(t *testing.T) {
	a, b := unitVec(1, 0, 0), unitVec(0, 1, 0)
	var members [][]float32
	for range 500 {
		members = append(members, a, b)
	}
	name := unitVec(0, 0, 1)
	vec, cut := blend(name, members, 3), tagCutoff(name, members, 3, 0.46, 0.60)
	if got := cosine(a, vec); got >= cut {
		t.Fatalf("a member of two orthogonal kinds scores %.4f against cutoff %.4f, want below", got, cut)
	}
	if got := cosine(unitVec(1, 1, 0), vec); got < cut {
		t.Fatalf("a document between both kinds scores %.4f against cutoff %.4f, want above", got, cut)
	}
}

// TestTagCutoff_OpposedNameAndMembersMatchNothing: members pointing exactly
// away from the name cancel it out at equal weights, and a blend of nothing
// has no direction any document could score against.
func TestTagCutoff_OpposedNameAndMembersMatchNothing(t *testing.T) {
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

// scoreVectors judges each tag against its own cutoff: a tag below its cutoff
// does not hide the ones after it, a score equal to the cutoff qualifies, and
// nothing limits how many tags are picked. Every tag with a vector is scored,
// best score first and equal scores by id, whatever order the tags were given
// in, and that order is left as it was. A tag with no vector has nothing to
// compare and is skipped. It fails closed: no score reaches a NaN cutoff,
// though every comparison with a NaN is false.
func TestScoreVectors_JudgesEachTagAgainstItsOwnCutoff(t *testing.T) {
	const dims = 4
	q := unitVec(1, 0, 0, 0)
	// tag is a tag whose vector scores about score against q: the vector is a
	// float32, so its exact cosine is a few 1e-8 off, which only a case
	// sitting right on its cutoff would notice.
	tag := func(id string, axis int, score, cutoff float64) TagVector {
		return TagVector{ID: id, Vec: scoreVec(dims, axis, score), Cutoff: cutoff}
	}
	// atCutoff takes its own exact cosine for its cutoff, so its score equals it
	atCutoff := tag("2", 2, 0.50, 0)
	atCutoff.Cutoff = cosine(q, atCutoff.Vec)
	ids := func(tags []TagVector) []string {
		out := make([]string, len(tags))
		for i, tg := range tags {
			out[i] = tg.ID
		}
		return out
	}
	for _, tc := range []struct {
		name   string
		tags   []TagVector
		want   []string
		scored int
	}{
		{"no tags", nil, nil, 0},
		{"a tag below its cutoff does not hide the ones after it",
			[]TagVector{tag("1", 1, 0.65, 0.70), tag("2", 2, 0.55, 0.50)}, []string{"2"}, 2},
		{"a score equal to the cutoff qualifies",
			[]TagVector{atCutoff}, []string{"2"}, 1},
		{"no score reaches a NaN cutoff",
			[]TagVector{tag("8", 1, 1, math.NaN())}, nil, 1},
		// its zero score would reach its cutoff of 0, were it not skipped
		{"a tag without a vector is skipped",
			[]TagVector{{ID: "1", Cutoff: 0}, tag("2", 2, 0.80, 0.46)}, []string{"2"}, 1},
		{"every tag that clears its cutoff is picked, best score first",
			[]TagVector{
				tag("1", 1, 0.75, 0.46), tag("2", 2, 0.60, 0.46), tag("3", 3, 0.90, 0.46), tag("4", 1, 0.70, 0.46),
				tag("5", 2, 0.85, 0.46), tag("6", 3, 0.65, 0.46), tag("7", 1, 0.80, 0.46),
			},
			[]string{"3", "5", "7", "1", "4", "6", "2"}, 7},
		{"equal scores come back ordered by id",
			[]TagVector{tag("b", 1, 0.80, 0.46), tag("a", 2, 0.80, 0.46)}, []string{"a", "b"}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			given := ids(tc.tags)
			ss := mustScoreVectors(t, [][]float32{q}, tc.tags, true)
			if len(ss) != tc.scored {
				t.Fatalf("scoreVectors scored %d tags, want %d: %+v", len(ss), tc.scored, ss)
			}
			got := picked(ss)
			// nil, not just empty, when nothing qualifies
			if !slices.Equal(got, tc.want) || (got == nil) != (tc.want == nil) {
				t.Fatalf("picked = %#v, want %#v", got, tc.want)
			}
			if after := ids(tc.tags); !slices.Equal(after, given) {
				t.Fatalf("scoreVectors reordered the caller's tags: %v, was %v", after, given)
			}
		})
	}
}

// TestScoreVectors_AlongAPathTheBetterFitWins: among the tags that reach their
// cutoff, each ancestor meets each descendant by margin. A descendant at
// least as wide as its ancestor shadows it, a narrower one is shadowed by it,
// so two children that both beat their parent are both picked and a child the
// parent beats is not; the tag with the widest margin of a family always
// stays, tags on other paths are judged on their own, a tag below its cutoff
// takes no part, and without onePath every tag that reaches its cutoff is
// picked.
func TestScoreVectors_AlongAPathTheBetterFitWins(t *testing.T) {
	const dims = 6
	q := unitVec(1, 0, 0, 0, 0, 0)
	tag := func(id, name string, axis int, score, cutoff float64) TagVector {
		return TagVector{ID: id, Name: name, Vec: scoreVec(dims, axis, score), Cutoff: cutoff}
	}
	rank := func(tags []TagVector, onePath bool) []string {
		return picked(mustScoreVectors(t, [][]float32{q}, tags, onePath))
	}
	shadowed := func(tags []TagVector) map[string]string {
		out := map[string]string{}
		for _, s := range mustScoreVectors(t, [][]float32{q}, tags, true) {
			out[s.ID] = s.ShadowedBy
		}
		return out
	}
	tags := []TagVector{
		tag("dev", "dev", 1, 0.70, 0.46),                 // margin 0.24
		tag("go", "Dev/Go", 2, 0.66, 0.40),               // margin 0.26: beats dev
		tag("goerr", "dev / go / errors", 3, 0.62, 0.50), // margin 0.12: beaten by dev and go
		tag("web", "dev / web", 4, 0.80, 0.50),           // margin 0.30: beats dev too
		tag("cook", "cooking", 5, 0.50, 0.46),            // margin 0.04: another path
		tag("devops", "devops", 1, 0.45, 0.46),           // below its cutoff; not on dev's path anyway
	}
	for _, tc := range []struct {
		name   string
		web    TagVector
		want   []string
		shadow map[string]string
	}{
		{"two children that beat their parent are both picked", tags[3], []string{"web", "go", "cook"},
			map[string]string{"dev": "web", "go": "", "goerr": "go", "web": "", "cook": "", "devops": ""}},
		{"a child its parent beats is not", tag("web", "dev / web", 4, 0.60, 0.50), // margin 0.10
			[]string{"go", "cook"}, map[string]string{"dev": "go", "go": "", "goerr": "go", "web": "dev", "cook": "", "devops": ""}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tags[3] = tc.web
			if got := rank(tags, true); !slices.Equal(got, tc.want) {
				t.Fatalf("picked = %v, want %v", got, tc.want)
			}
			if got := shadowed(tags); !maps.Equal(got, tc.shadow) {
				t.Fatalf("ShadowedBy = %v, want %v", got, tc.shadow)
			}
		})
	}
	if got, want := rank(tags, false), []string{"dev", "go", "goerr", "web", "cook"}; !slices.Equal(got, want) {
		t.Fatalf("picked without onePath = %v, want %v", got, want)
	}
	// a parent cleared by more than every child is what the item is about
	tags[0] = tag("dev", "dev", 1, 0.80, 0.46) // margin 0.34
	if got, want := rank(tags, true), []string{"dev", "cook"}; !slices.Equal(got, want) {
		t.Fatalf("with dev cleared by most, picked = %v, want %v", got, want)
	}
	// on a tie the descendant stays
	tie := []TagVector{tag("a", "a", 1, 0.60, 0.40), tag("b", "a / b", 2, 0.60, 0.40)}
	if got, want := rank(tie, true), []string{"b"}; !slices.Equal(got, want) {
		t.Fatalf("on a tie, picked = %v, want %v", got, want)
	}
}

func TestTagPath(t *testing.T) {
	for name, want := range map[string][]string{
		"dev / go":          {"dev", "go"},
		" Dev/GO ":          {"dev", "go"},
		"dev // go /":       {"dev", "go"},
		"cooking":           {"cooking"},
		"":                  nil,
		"a / b / c / d / e": {"a", "b", "c", "d", "e"},
	} {
		if got := tagPath(name); !slices.Equal(got, want) {
			t.Errorf("tagPath(%q) = %q, want %q", name, got, want)
		}
	}
	for _, tc := range []struct {
		a, b string
		want bool
	}{
		{"dev", "dev / go", true}, {"dev / go", "dev", true}, {"dev / go", "dev / go", false},
		{"dev", "devops", false}, {"dev / go", "dev / web", false}, {"go", "dev / go", false},
	} {
		if got := onPath(tagPath(tc.a), tagPath(tc.b)); got != tc.want {
			t.Errorf("onPath(%q, %q) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
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

package eval

import (
	"math"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// ranked a b c d with b and d relevant: the hand-computed values below are
// recall@1 = 0, @2 = 1/2, @4 = 1; RR = 1/2; DCG@4 = 1/log2(3) + 1/log2(5),
// ideal DCG = 1/log2(2) + 1/log2(3).
func TestRetrievalMetrics(t *testing.T) {
	ranked := []string{"a", "b", "c", "d"}
	rel := toSet([]string{"b", "d"})
	for k, want := range map[int]float64{1: 0, 2: 0.5, 4: 1, 10: 1} {
		if got := recallAt(ranked, rel, k); !near(got, want) {
			t.Errorf("recall@%d = %f, want %f", k, got, want)
		}
	}
	if got := reciprocalRank(ranked, rel); !near(got, 0.5) {
		t.Errorf("RR = %f, want 0.5", got)
	}
	wantNDCG := (1/math.Log2(3) + 1/math.Log2(5)) / (1 + 1/math.Log2(3))
	if got := ndcgAt(ranked, rel, 4); !near(got, wantNDCG) {
		t.Errorf("nDCG@4 = %f, want %f", got, wantNDCG)
	}
	// nDCG is capped by k on both sides: with k = 1 the ideal is one hit
	if got := ndcgAt([]string{"b"}, rel, 1); !near(got, 1) {
		t.Errorf("nDCG@1 of a relevant first hit = %f, want 1", got)
	}
}

func TestRetrievalMetrics_EdgeCases(t *testing.T) {
	none := toSet(nil)
	if recallAt([]string{"a"}, none, 5) != 0 || ndcgAt([]string{"a"}, none, 5) != 0 || reciprocalRank([]string{"a"}, none) != 0 {
		t.Error("an empty relevant set must score 0, not divide by zero")
	}
	rel := toSet([]string{"z"})
	if recallAt(nil, rel, 5) != 0 || reciprocalRank(nil, rel) != 0 || ndcgAt(nil, rel, 5) != 0 {
		t.Error("no hits must score 0")
	}
}

// hits x 0.9, y 0.5, z 0.3 with x and z expected.
func TestDecisions(t *testing.T) {
	hits := []scored{{"x", 0.9}, {"y", 0.5}, {"z", 0.3}}
	exp := toSet([]string{"x", "z"})
	cases := []struct {
		name       string
		floor      float64
		tp, fp, fn int
	}{
		{"floor between y and z", 0.4, 1, 1, 1},
		{"floor below everything", 0.2, 2, 1, 0},
		{"floor above everything", 0.95, 0, 0, 2},
		{"floor inclusive", 0.5, 1, 1, 1},
	}
	for _, c := range cases {
		tp, fp, fn := decisions(hits, exp, c.floor)
		if tp != c.tp || fp != c.fp || fn != c.fn {
			t.Errorf("%s: decisions = %d %d %d, want %d %d %d", c.name, tp, fp, fn, c.tp, c.fp, c.fn)
		}
	}
}

// TestDecisions_AssignsEveryHitAtOrAboveTheFloor: nothing caps how many tags a
// document gets, as in autotag, so the floor alone decides. Hits a to f
// score at or above the floor (f exactly at it) and all six are assigned; g
// scores below it and is not, though it is expected.
func TestDecisions_AssignsEveryHitAtOrAboveTheFloor(t *testing.T) {
	hits := []scored{{"a", 0.9}, {"b", 0.8}, {"c", 0.7}, {"d", 0.6}, {"e", 0.5}, {"f", 0.4}, {"g", 0.3}}
	exp := toSet([]string{"a", "c", "g"})
	tp, fp, fn := decisions(hits, exp, 0.4)
	if tp != 2 || fp != 4 || fn != 1 {
		t.Errorf("decisions = %d %d %d, want 2 4 1 (six tags assigned, a and c right, g left out)", tp, fp, fn)
	}
}

func TestCounts_Cut(t *testing.T) {
	var c counts
	c.add(1, 1, 1)
	got := c.cut(0.4)
	if !near(got.Precision, 0.5) || !near(got.Recall, 0.5) || !near(got.F1, 0.5) || got.Floor != 0.4 {
		t.Errorf("cut = %+v", got)
	}
	c = counts{}
	c.add(2, 1, 0)
	got = c.cut(0.2)
	if !near(got.Precision, 2.0/3) || !near(got.Recall, 1) || !near(got.F1, 0.8) {
		t.Errorf("cut = %+v", got)
	}
	if got := (counts{}).cut(0.5); got.Precision != 0 || got.Recall != 0 || got.F1 != 0 {
		t.Errorf("empty counts = %+v, want zeros", got)
	}
}

func TestSweepFloorsAndBestCut(t *testing.T) {
	floors := sweepFloors()
	if len(floors) != 91 || !near(floors[0], 0.05) || !near(floors[90], 0.95) || !near(floors[1]-floors[0], 0.01) {
		t.Fatalf("sweepFloors = %d values from %f to %f", len(floors), floors[0], floors[len(floors)-1])
	}
	cuts := []Cut{{Floor: 0.1, F1: 0.5}, {Floor: 0.2, F1: 0.7}, {Floor: 0.3, F1: 0.7}, {Floor: 0.4, F1: 0.6}}
	if best := bestCut(cuts); best.Floor != 0.2 {
		t.Errorf("bestCut = %+v, want the lowest floor among the 0.7 ties", best)
	}
}

func TestTop1(t *testing.T) {
	exp := toSet([]string{"x"})
	if !top1([]scored{{"x", 0.9}, {"y", 0.8}}, exp) || top1([]scored{{"y", 0.9}, {"x", 0.8}}, exp) || top1(nil, exp) {
		t.Error("top1 must be true only when the best-scoring hit is expected")
	}
}

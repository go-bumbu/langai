package eval

import "math"

// Cut is a tagging measurement at one floor: micro-averaged precision,
// recall and F1 over every (document, tag) decision.
type Cut struct {
	Floor     float64
	Precision float64
	Recall    float64
	F1        float64
}

// relevantSet is the set of ids a query should find, or the tags a document
// should get.
type relevantSet map[string]bool

func toSet(ids []string) relevantSet {
	out := make(relevantSet, len(ids))
	for _, id := range ids {
		out[id] = true
	}
	return out
}

// recallAt is the share of relevant ids found within the first k of ranked.
func recallAt(ranked []string, relevant relevantSet, k int) float64 {
	if len(relevant) == 0 {
		return 0
	}
	found := 0
	for i, id := range ranked {
		if i == k {
			break
		}
		if relevant[id] {
			found++
		}
	}
	return float64(found) / float64(len(relevant))
}

// reciprocalRank is 1/rank of the first relevant id in ranked, 0 when none.
func reciprocalRank(ranked []string, relevant relevantSet) float64 {
	for i, id := range ranked {
		if relevant[id] {
			return 1 / float64(i+1)
		}
	}
	return 0
}

// ndcgAt is the normalized discounted cumulative gain over the first k of
// ranked with binary gains: the ideal ranking has every relevant id first.
func ndcgAt(ranked []string, relevant relevantSet, k int) float64 {
	if len(relevant) == 0 {
		return 0
	}
	var dcg float64
	for i, id := range ranked {
		if i == k {
			break
		}
		if relevant[id] {
			dcg += 1 / math.Log2(float64(i+2))
		}
	}
	var ideal float64
	for i := 0; i < min(k, len(relevant)); i++ {
		ideal += 1 / math.Log2(float64(i+2))
	}
	return dcg / ideal
}

// scored is one tag with its score for a document; a hit list is sorted
// best first, as the store returns it.
type scored struct {
	Tag   string
	Score float64
}

// decisions counts, for one document, the tags assigned at floor (every hit
// scoring at least floor) that are expected (tp), the assigned ones that are
// not (fp), and the expected ones left unassigned (fn). The floor is
// inclusive, like autotag's NameMinScore, and nothing else limits how
// many tags a document gets.
func decisions(hits []scored, expected relevantSet, floor float64) (tp, fp, fn int) {
	for _, h := range hits {
		if h.Score < floor {
			break
		}
		if expected[h.Tag] {
			tp++
		} else {
			fp++
		}
	}
	return tp, fp, len(expected) - tp
}

// counts accumulates decisions over documents.
type counts struct{ tp, fp, fn int }

func (c *counts) add(tp, fp, fn int) {
	c.tp += tp
	c.fp += fp
	c.fn += fn
}

// cut reports the micro-averaged precision, recall and F1 of the
// accumulated decisions, labelled with the floor they were taken at.
func (c counts) cut(floor float64) Cut {
	out := Cut{Floor: floor}
	if c.tp+c.fp > 0 {
		out.Precision = float64(c.tp) / float64(c.tp+c.fp)
	}
	if c.tp+c.fn > 0 {
		out.Recall = float64(c.tp) / float64(c.tp+c.fn)
	}
	if out.Precision+out.Recall > 0 {
		out.F1 = 2 * out.Precision * out.Recall / (out.Precision + out.Recall)
	}
	return out
}

// sweepFloors are the cutoffs the tagging sweep evaluates: 0.05 to 0.95 in
// steps of 0.01, so a candidate model's useful floor is found to two
// decimals. Built from integers so the values print cleanly.
func sweepFloors() []float64 {
	out := make([]float64, 0, 91)
	for i := 5; i <= 95; i++ {
		out = append(out, float64(i)/100)
	}
	return out
}

// bestCut is the cut with the highest F1; on ties the lowest floor, which
// assigns the most tags for the same quality.
func bestCut(cuts []Cut) Cut {
	best := cuts[0]
	for _, c := range cuts[1:] {
		if c.F1 > best.F1 {
			best = c
		}
	}
	return best
}

// top1 reports whether the best-scoring hit is an expected tag.
func top1(hits []scored, expected relevantSet) bool {
	return len(hits) > 0 && expected[hits[0].Tag]
}

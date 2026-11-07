package eval

import (
	"math"
	"strings"
	"unicode/utf8"
)

// minContainRunes is the shortest alias that can match a tag by containment.
const minContainRunes = 4

// LongScore is how well a tag list covers the topics of one long document.
type LongScore struct {
	// Recall is over main and side topics together.
	Recall, MainRecall, SideRecall float64
	// LateSideRecall is over the side topics past LateOffset.
	LateSideRecall float64
	// MentionRate is the share of mention topics covered.
	MentionRate float64
	// Precision is the share of tags assigned to a main or side topic.
	Precision float64
	Tags      int
}

// F1 is the harmonic mean of Precision and Recall, 0 when both are 0.
func (s LongScore) F1() float64 {
	if s.Precision+s.Recall == 0 {
		return 0
	}
	return 2 * s.Precision * s.Recall / (s.Precision + s.Recall)
}

// aliasSim is the similarity of a tag to an alias: 1 on fold-case containment
// either way (alias of at least minContainRunes), else sim.
func aliasSim(tag, alias string, sim func(tag, alias string) float64) float64 {
	t := strings.ToLower(strings.TrimSpace(tag))
	a := strings.ToLower(strings.TrimSpace(alias))
	if utf8.RuneCountInString(a) >= minContainRunes && t != "" &&
		(strings.Contains(t, a) || strings.Contains(a, t)) {
		return 1
	}
	return sim(tag, alias)
}

// MatchTopics assigns each tag to the topic with its best alias similarity
// when that reaches tau (ties go to the topic listed first); a topic is
// covered when a tag went to it.
func MatchTopics(tags []string, topics []Topic, sim func(tag, alias string) float64, tau float64) LongScore {
	covered := make([]bool, len(topics))
	useful := 0
	for _, tag := range tags {
		best, bestSim := -1, math.Inf(-1)
		for i, t := range topics {
			for _, a := range t.Aliases {
				if v := aliasSim(tag, a, sim); v > bestSim {
					best, bestSim = i, v
				}
			}
		}
		if best < 0 || bestSim < tau {
			continue
		}
		covered[best] = true
		if topics[best].Kind != TopicMention {
			useful++
		}
	}
	var main, side, late, mention [2]int // total, covered
	count := func(c *[2]int, i int) {
		c[0]++
		if covered[i] {
			c[1]++
		}
	}
	for i, t := range topics {
		switch t.Kind {
		case TopicMain:
			count(&main, i)
		case TopicSide:
			count(&side, i)
			if t.Late() {
				count(&late, i)
			}
		case TopicMention:
			count(&mention, i)
		}
	}
	s := LongScore{
		Recall:         ratio(main[1]+side[1], main[0]+side[0]),
		MainRecall:     ratio(main[1], main[0]),
		SideRecall:     ratio(side[1], side[0]),
		LateSideRecall: ratio(late[1], late[0]),
		MentionRate:    ratio(mention[1], mention[0]),
		Precision:      ratio(useful, len(tags)),
		Tags:           len(tags),
	}
	return s
}

func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}

// MeanLongScores is the field-wise mean of ss; Tags is rounded. An empty
// slice gives the zero score.
func MeanLongScores(ss []LongScore) LongScore {
	if len(ss) == 0 {
		return LongScore{}
	}
	var m LongScore
	var tags float64
	for _, s := range ss {
		m.Recall += s.Recall
		m.MainRecall += s.MainRecall
		m.SideRecall += s.SideRecall
		m.LateSideRecall += s.LateSideRecall
		m.MentionRate += s.MentionRate
		m.Precision += s.Precision
		tags += float64(s.Tags)
	}
	n := float64(len(ss))
	m.Recall /= n
	m.MainRecall /= n
	m.SideRecall /= n
	m.LateSideRecall /= n
	m.MentionRate /= n
	m.Precision /= n
	m.Tags = int(math.Round(tags / n))
	return m
}

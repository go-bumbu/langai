package eval

import (
	"math"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/cases"
	"golang.org/x/text/unicode/norm"
)

// minContainRunes is the shortest text that can match by containment: of a
// tag and a topic's name, the one found inside the other needs at least this
// many runes, so "ai" matches neither a longer alias nor a longer tag.
const minContainRunes = 4

// LongScore is how well a tag list covers the topics of one long document.
// A rate over nothing (the side recall of a document without side topics,
// the precision of an empty tag list) is 0; the counts say when that is so,
// and MeanLongScores leaves such a document out of that rate's mean.
type LongScore struct {
	// Recall is over main and side topics together.
	Recall, MainRecall, SideRecall float64
	// LateSideRecall is over the side topics past LateOffset.
	LateSideRecall float64
	// MentionRate is the share of mention topics covered.
	MentionRate float64
	// Precision is the share of tags assigned to a main or side topic.
	Precision float64
	// Tags is the number of tags scored.
	Tags int
	// Main, Side, LateSide and Mentions are the number of the document's
	// topics of each kind; the late side topics are counted in Side too.
	Main, Side, LateSide, Mentions int
}

// F1 is the harmonic mean of Precision and Recall, 0 when both are 0. A
// document that got no tags covers no topic, so its F1 is 0.
func (s LongScore) F1() float64 {
	if s.Precision+s.Recall == 0 {
		return 0
	}
	return 2 * s.Precision * s.Recall / (s.Precision + s.Recall)
}

// words are the words of s as two texts are compared by: its runs of letters,
// digits and combining marks, case-folded and in NFC, so "Smart-Grid" and
// "smart grid" are the same two words and an accent written as a combining
// mark stays in its word.
func words(s string) []string {
	s = norm.NFC.String(cases.Fold().String(norm.NFC.String(s)))
	return strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r) && !unicode.IsMark(r)
	})
}

// textMatch grades how the words of a tag match those of a topic's name (its
// label or an alias): 1 when they are the same words; when one is a run of
// whole words of the other and has at least minContainRunes runes, the share
// of the longer one's runes it covers, which is higher the closer the two
// are; 0 otherwise. Words are compared whole, so "art" is not in "smart grid"
// and "cooperative" is not in "housing cooperatives".
func textMatch(tag, name []string) float64 {
	short, long := tag, name
	if len(short) > len(long) {
		short, long = long, short
	}
	if len(short) == 0 || !hasRun(long, short) {
		return 0
	}
	if len(short) == len(long) {
		return 1
	}
	n := runeLen(short)
	if n < minContainRunes {
		return 0
	}
	return float64(n) / float64(runeLen(long))
}

// hasRun reports whether run is in ws as consecutive words.
func hasRun(ws, run []string) bool {
	for i := 0; i+len(run) <= len(ws); i++ {
		j := 0
		for j < len(run) && ws[i+j] == run[j] {
			j++
		}
		if j == len(run) {
			return true
		}
	}
	return false
}

// runeLen is the length in runes of ws joined by single spaces.
func runeLen(ws []string) int {
	n := len(ws) - 1
	for _, w := range ws {
		n += utf8.RuneCountInString(w)
	}
	return n
}

// topicMatch is how well a tag fits one topic.
type topicMatch struct {
	// sim is 1 for a text match (see textMatch), else the tag's best
	// similarity to an alias, at most 1.
	sim float64
	// label is set when the tag is the topic's label, word for word.
	label bool
	// cover is textMatch's grade of the text match, 0 without one.
	cover float64
}

// better reports whether m is a closer fit than o: by similarity, then a
// label over anything else, then the closer text match.
func (m topicMatch) better(o topicMatch) bool {
	switch {
	case m.sim != o.sim:
		return m.sim > o.sim
	case m.label != o.label:
		return m.label
	default:
		return m.cover > o.cover
	}
}

// topicNames are the words of a topic's label and of each of its aliases.
type topicNames struct {
	label   []string
	aliases [][]string
}

func namesOf(topics []Topic) []topicNames {
	out := make([]topicNames, len(topics))
	for i, t := range topics {
		out[i].label = words(t.Label)
		out[i].aliases = make([][]string, len(t.Aliases))
		for j, a := range t.Aliases {
			out[i].aliases[j] = words(a)
		}
	}
	return out
}

// fit is the tag's match with topic t: a text match with its label or an
// alias, or else its best similarity to an alias. sim is not asked about the
// label, and a similarity that is not a number counts as none.
func (t Topic) fit(tag string, tagWords []string, names topicNames, sim func(tag, alias string) float64) topicMatch {
	best := topicMatch{sim: math.Inf(-1)}
	if c := textMatch(tagWords, names.label); c > 0 {
		best = topicMatch{sim: 1, label: c == 1, cover: c}
	}
	for i, a := range t.Aliases {
		m := topicMatch{sim: 1, cover: textMatch(tagWords, names.aliases[i])}
		if m.cover == 0 {
			v := sim(tag, a)
			if math.IsNaN(v) {
				continue
			}
			m.sim = math.Min(v, 1)
		}
		if m.better(best) {
			best = m
		}
	}
	return best
}

// MatchTopics assigns each tag to the topic it fits best, when that fit
// reaches tau; a topic is covered when a tag went to it. A tag fits a topic
// with 1 when its words are those of the topic's label or of an alias, or
// when one holds the other as whole words (see textMatch), and otherwise with
// its best sim to an alias. Among topics that fit alike, a tag that is a
// topic's label goes to that topic, then the closer text match wins (the
// longer name found in the tag, the shorter name the tag is found in), so
// "housing cooperatives" goes to the side topic of that label and not to the
// main topic "cooperatives"; what is still even goes to the topic listed
// first.
func MatchTopics(tags []string, topics []Topic, sim func(tag, alias string) float64, tau float64) LongScore {
	names := namesOf(topics)
	covered := make([]bool, len(topics))
	useful := 0
	for _, tag := range tags {
		tagWords := words(tag)
		at, best := -1, topicMatch{sim: math.Inf(-1)}
		for i, t := range topics {
			if m := t.fit(tag, tagWords, names[i], sim); m.better(best) {
				at, best = i, m
			}
		}
		if at < 0 || best.sim < tau {
			continue
		}
		covered[at] = true
		if topics[at].Kind != TopicMention {
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
	return LongScore{
		Recall:         ratio(main[1]+side[1], main[0]+side[0]),
		MainRecall:     ratio(main[1], main[0]),
		SideRecall:     ratio(side[1], side[0]),
		LateSideRecall: ratio(late[1], late[0]),
		MentionRate:    ratio(mention[1], mention[0]),
		Precision:      ratio(useful, len(tags)),
		Tags:           len(tags),
		Main:           main[0],
		Side:           side[0],
		LateSide:       late[0],
		Mentions:       mention[0],
	}
}

func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}

// LongMean is the mean of the LongScores of a set of documents. Every rate
// is the mean over the documents it exists for: a recall over those with
// topics of that kind, Precision over those that got a tag.
type LongMean struct {
	// Documents is the number of scores averaged.
	Documents int
	// Recall, MainRecall, SideRecall, LateSideRecall and MentionRate are the
	// means of the fields of the same name.
	Recall, MainRecall, SideRecall, LateSideRecall, MentionRate float64
	// Precision is the mean over the documents with at least one tag.
	Precision float64
	// MeanF1 is the mean of each document's own F1, over the documents with a
	// main or side topic; one without tags counts as 0. It is not the F1 of
	// Precision and Recall above.
	MeanF1 float64
	// Untagged is the number of documents that got no tag: they are left out
	// of Precision and count in every recall and in MeanF1.
	Untagged int
	// Tags is the mean number of tags per document.
	Tags float64
}

// rate sums one rate over the documents it exists for.
type rate struct {
	sum float64
	n   int
}

func (r *rate) add(v float64, exists bool) {
	if exists {
		r.sum += v
		r.n++
	}
}

func (r rate) mean() float64 { return mean(r.sum, float64(r.n)) }

// MeanLongScores averages ss as LongMean describes. An empty slice gives the
// zero mean.
func MeanLongScores(ss []LongScore) LongMean {
	var recall, main, side, late, mention, precision, f1, tags rate
	untagged := 0
	for _, s := range ss {
		recall.add(s.Recall, s.Main+s.Side > 0)
		main.add(s.MainRecall, s.Main > 0)
		side.add(s.SideRecall, s.Side > 0)
		late.add(s.LateSideRecall, s.LateSide > 0)
		mention.add(s.MentionRate, s.Mentions > 0)
		precision.add(s.Precision, s.Tags > 0)
		f1.add(s.F1(), s.Main+s.Side > 0)
		tags.add(float64(s.Tags), true)
		if s.Tags == 0 {
			untagged++
		}
	}
	return LongMean{
		Documents:      len(ss),
		Recall:         recall.mean(),
		MainRecall:     main.mean(),
		SideRecall:     side.mean(),
		LateSideRecall: late.mean(),
		MentionRate:    mention.mean(),
		Precision:      precision.mean(),
		MeanF1:         f1.mean(),
		Untagged:       untagged,
		Tags:           tags.mean(),
	}
}

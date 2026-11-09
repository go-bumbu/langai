package eval

import (
	"math"
	"slices"
	"testing"
)

func matchDoc() []Topic {
	return []Topic{
		{Label: "Main", Kind: TopicMain, Aliases: []string{"quantum computing"}},
		{Label: "Early", Kind: TopicSide, Offset: 100, Aliases: []string{"error correction"}},
		{Label: "Late", Kind: TopicSide, Offset: 3000, Aliases: []string{"cryogenics"}},
		{Label: "Passing", Kind: TopicMention, Offset: 4000, Aliases: []string{"superconductors"}},
	}
}

func noSim(string, string) float64 { return 0 }

func TestWords(t *testing.T) {
	cases := []struct {
		name, in string
		want     []string
	}{
		{"case and punctuation", "Smart-Grid: (Basics)", []string{"smart", "grid", "basics"}},
		{"digits are words", "Web 2.0", []string{"web", "2", "0"}},
		{"accents stay in the word", "Energía Solar", []string{"energía", "solar"}},
		{"decomposed accents are composed", "café cultura", []string{"café", "cultura"}},
		{"full case folding", "STRASSE Straße", []string{"strasse", "strasse"}},
		{"nothing but marks", " -- / ", nil},
		{"empty", "", nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := words(c.in); !slices.Equal(got, c.want) {
				t.Errorf("words(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestTextMatch(t *testing.T) {
	cases := []struct {
		name      string
		tag, text string
		want      float64
	}{
		{"same words", "Smart Grid", "smart-grid", 1},
		{"same short word", "AI", "ai", 1},
		{"name is whole words of the tag", "smart grid basics", "smart grid", 10.0 / 17},
		{"tag is whole words of the name", "grid", "smart grid", 4.0 / 10},
		{"inside a word is no match", "ai", "container shipping", 0},
		{"inside a word is no match, three letters", "art", "smart grid", 0},
		{"a plural is another word", "housing cooperatives", "cooperative", 0},
		{"the contained tag is too short", "ai", "ai research", 0},
		{"the contained name is too short", "ai research", "AI", 0},
		{"accented words", "energía solar térmica", "Energía Solar", 13.0 / 21},
		{"the words must be next to each other", "smart energy grid", "smart grid", 0},
		{"the words must be in order", "grid smart", "smart grid", 0},
		{"an empty tag", "", "smart grid", 0},
		{"an empty name", "smart grid", " ", 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := textMatch(words(c.tag), words(c.text)); !near(got, c.want) {
				t.Errorf("textMatch(%q, %q) = %v, want %v", c.tag, c.text, got, c.want)
			}
		})
	}
}

// coop is the shape of de-genossenschaften: the main topic's label is a word
// of every side label.
func coop() []Topic {
	return []Topic{
		{Label: "cooperatives", Kind: TopicMain, Aliases: []string{"Genossenschaft", "cooperative"}},
		{Label: "housing cooperatives", Kind: TopicSide, Offset: 3000, Aliases: []string{"Wohnungsgenossenschaft", "housing cooperative"}},
		{Label: "cooperative banking", Kind: TopicSide, Offset: 5000, Aliases: []string{"Volksbank", "cooperative bank"}},
	}
}

func TestMatchTopics(t *testing.T) {
	half := func(wantTag, wantAlias string) func(string, string) float64 {
		return func(tag, alias string) float64 {
			if tag == wantTag && alias == wantAlias {
				return 0.5
			}
			return 0
		}
	}
	cases := []struct {
		name   string
		tags   []string
		topics []Topic
		sim    func(tag, alias string) float64
		want   LongScore
	}{
		{
			name: "an alias inside the tag", tags: []string{"Quantum Computing Basics"}, topics: matchDoc(), sim: noSim,
			want: LongScore{Recall: 1.0 / 3, MainRecall: 1, Precision: 1, Tags: 1},
		},
		{
			name: "a side topic picked by its label is not the main topic's", tags: []string{"cooperatives", "housing cooperatives", "cooperative banking"},
			topics: coop(), sim: noSim,
			want: LongScore{Recall: 1, MainRecall: 1, SideRecall: 1, LateSideRecall: 1, Precision: 1, Tags: 3},
		},
		{
			name: "the longer contained name wins", tags: []string{"social housing cooperatives"}, topics: coop(), sim: noSim,
			want: LongScore{Recall: 1.0 / 3, SideRecall: 0.5, LateSideRecall: 0.5, Precision: 1, Tags: 1},
		},
		{
			name: "the name closest to a contained tag wins", tags: []string{"cooperative"},
			topics: []Topic{
				{Label: "banks", Kind: TopicSide, Aliases: []string{"cooperative bank"}},
				{Label: "co-ops", Kind: TopicMain, Aliases: []string{"cooperative"}},
			},
			sim:  noSim,
			want: LongScore{Recall: 0.5, MainRecall: 1, Precision: 1, Tags: 1},
		},
		{
			name: "a label beats the same words as an earlier topic's alias", tags: []string{"Alpha Beta"},
			topics: []Topic{
				{Label: "first", Kind: TopicMain, Aliases: []string{"alpha beta"}},
				{Label: "alpha beta", Kind: TopicSide, Aliases: []string{"other"}},
			},
			sim:  noSim,
			want: LongScore{Recall: 0.5, SideRecall: 1, Precision: 1, Tags: 1},
		},
		{
			name: "a label beats a similarity above 1", tags: []string{"alpha beta"},
			topics: []Topic{
				{Label: "first", Kind: TopicMain, Aliases: []string{"gamma"}},
				{Label: "alpha beta", Kind: TopicSide, Aliases: []string{"other"}},
			},
			sim:  func(string, string) float64 { return 1.0000001 },
			want: LongScore{Recall: 0.5, SideRecall: 1, Precision: 1, Tags: 1},
		},
		{
			name: "an even match goes to the topic listed first", tags: []string{"alpha beta"},
			topics: []Topic{
				{Label: "one", Kind: TopicMain, Aliases: []string{"alpha beta"}},
				{Label: "two", Kind: TopicSide, Aliases: []string{"alpha beta"}},
			},
			sim:  noSim,
			want: LongScore{Recall: 0.5, MainRecall: 1, Precision: 1, Tags: 1},
		},
		{
			name: "a short tag does not match inside a word", tags: []string{"ai", "art"},
			topics: []Topic{{Label: "freight", Kind: TopicMain, Aliases: []string{"container shipping", "smart grid"}}},
			sim:    noSim,
			want:   LongScore{Tags: 2},
		},
		{
			name: "a short alias does not match as a word of the tag", tags: []string{"main ai thing"},
			topics: []Topic{{Label: "machine learning", Kind: TopicMain, Aliases: []string{"AI"}}},
			sim:    noSim,
			want:   LongScore{Tags: 1},
		},
		{
			name: "a short tag matches the same short alias", tags: []string{"ai"},
			topics: []Topic{{Label: "machine learning", Kind: TopicMain, Aliases: []string{"AI"}}},
			sim:    noSim,
			want:   LongScore{Recall: 1, MainRecall: 1, Precision: 1, Tags: 1},
		},
		{
			name: "similarity at tau", tags: []string{"qec"}, topics: matchDoc(), sim: half("qec", "error correction"),
			want: LongScore{Recall: 1.0 / 3, SideRecall: 0.5, Precision: 1, Tags: 1},
		},
		{
			name: "similarity below tau", tags: []string{"qec"}, topics: matchDoc(),
			sim:  func(string, string) float64 { return 0.49 },
			want: LongScore{Tags: 1},
		},
		{
			name: "a similarity that is not a number matches nothing", tags: []string{"qec"}, topics: matchDoc(),
			sim:  func(string, string) float64 { return math.NaN() },
			want: LongScore{Tags: 1},
		},
		{
			name: "a mention lowers precision", tags: []string{"superconductors", "quantum computing"}, topics: matchDoc(), sim: noSim,
			want: LongScore{Recall: 1.0 / 3, MainRecall: 1, MentionRate: 1, Precision: 0.5, Tags: 2},
		},
		{
			name: "a late side topic", tags: []string{"cryogenics"}, topics: matchDoc(), sim: noSim,
			want: LongScore{Recall: 1.0 / 3, SideRecall: 0.5, LateSideRecall: 1, Precision: 1, Tags: 1},
		},
		{
			name: "no tags", tags: nil, topics: matchDoc(), sim: noSim,
			want: LongScore{},
		},
		{
			name: "no topics", tags: []string{"alpha"}, topics: nil, sim: noSim,
			want: LongScore{Tags: 1},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := MatchTopics(c.tags, c.topics, c.sim, 0.5)
			want := c.want
			for _, tp := range c.topics { // the topic counts follow from the topics
				switch tp.Kind {
				case TopicMain:
					want.Main++
				case TopicSide:
					want.Side++
					if tp.Late() {
						want.LateSide++
					}
				case TopicMention:
					want.Mentions++
				}
			}
			if !nearScore(got, want) {
				t.Errorf("score = %+v\nwant    %+v", got, want)
			}
		})
	}
}

func nearScore(a, b LongScore) bool {
	return near(a.Recall, b.Recall) && near(a.MainRecall, b.MainRecall) && near(a.SideRecall, b.SideRecall) &&
		near(a.LateSideRecall, b.LateSideRecall) && near(a.MentionRate, b.MentionRate) && near(a.Precision, b.Precision) &&
		a.Tags == b.Tags && a.Main == b.Main && a.Side == b.Side && a.LateSide == b.LateSide && a.Mentions == b.Mentions
}

func TestLongScore_F1(t *testing.T) {
	cases := []struct {
		name string
		s    LongScore
		want float64
	}{
		{"zero score", LongScore{}, 0},
		{"no tags", LongScore{Main: 1, Side: 3}, 0},
		{"harmonic mean", LongScore{Precision: 0.5, Recall: 1, Tags: 2, Main: 1}, 2.0 / 3},
		{"tags that cover nothing", LongScore{Tags: 4, Main: 1}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.s.F1(); !near(got, c.want) {
				t.Errorf("F1 = %v, want %v", got, c.want)
			}
		})
	}
}

func TestMeanLongScores(t *testing.T) {
	// full is a document with every kind of topic and three tags
	full := func(s LongScore) LongScore {
		s.Main, s.Side, s.LateSide, s.Mentions = 1, 4, 2, 2
		if s.Tags == 0 {
			s.Tags = 3
		}
		return s
	}
	cases := []struct {
		name string
		in   []LongScore
		want LongMean
	}{
		{"no documents", nil, LongMean{}},
		{
			name: "field-wise mean",
			in: []LongScore{
				full(LongScore{Recall: 1, MainRecall: 0.2, SideRecall: 0.4, LateSideRecall: 0.6, MentionRate: 0.8, Precision: 0.1, Tags: 3}),
				full(LongScore{Recall: 0, MainRecall: 0.4, SideRecall: 0.8, LateSideRecall: 0, MentionRate: 0.2, Precision: 0.3, Tags: 4}),
			},
			want: LongMean{
				Documents: 2, Recall: 0.5, MainRecall: 0.3, SideRecall: 0.6, LateSideRecall: 0.3, MentionRate: 0.5,
				Precision: 0.2, MeanF1: (2 * 0.1 / 1.1) / 2, Tags: 3.5,
			},
		},
		{
			name: "F1 is the mean of each document's, not that of the mean precision and recall",
			in: []LongScore{
				full(LongScore{Recall: 0.2, Precision: 1}),
				full(LongScore{Recall: 1, Precision: 0.2}),
			},
			// each document's F1 is 1/3; the F1 of the means (0.6, 0.6) would be 0.6
			want: LongMean{Documents: 2, Recall: 0.6, Precision: 0.6, MeanF1: 1.0 / 3, Tags: 3},
		},
		{
			name: "a document without tags is left out of precision and counts as F1 0",
			in: []LongScore{
				full(LongScore{Recall: 1, Precision: 0.5, Tags: 2}),
				{Main: 1, Side: 4, LateSide: 2, Mentions: 2},
			},
			want: LongMean{Documents: 2, Untagged: 1, Recall: 0.5, Precision: 0.5, MeanF1: (2.0 / 3) / 2, Tags: 1},
		},
		{
			name: "no document has tags",
			in:   []LongScore{{Main: 1, Side: 4}, {Main: 1, Side: 3}},
			want: LongMean{Documents: 2, Untagged: 2},
		},
		{
			name: "a rate is the mean over the documents that have such topics",
			in: []LongScore{
				{Recall: 1, MainRecall: 1, SideRecall: 1, LateSideRecall: 1, MentionRate: 1, Precision: 1, Tags: 2, Main: 1, Side: 2, LateSide: 1, Mentions: 1},
				{Recall: 1, MainRecall: 1, Precision: 1, Tags: 2, Main: 1}, // no side, late or mention topics
			},
			want: LongMean{
				Documents: 2, Recall: 1, MainRecall: 1, SideRecall: 1, LateSideRecall: 1, MentionRate: 1,
				Precision: 1, MeanF1: 1, Tags: 2,
			},
		},
		{
			name: "a document without topics is left out of recall and F1",
			in: []LongScore{
				full(LongScore{Recall: 0.5, MainRecall: 1, Precision: 0.5, Tags: 2}),
				{Tags: 2},
			},
			want: LongMean{Documents: 2, Recall: 0.5, MainRecall: 1, Precision: 0.25, MeanF1: 0.5, Tags: 2},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := MeanLongScores(c.in)
			w := c.want
			if got.Documents != w.Documents || got.Untagged != w.Untagged ||
				!near(got.Recall, w.Recall) || !near(got.MainRecall, w.MainRecall) || !near(got.SideRecall, w.SideRecall) ||
				!near(got.LateSideRecall, w.LateSideRecall) || !near(got.MentionRate, w.MentionRate) ||
				!near(got.Precision, w.Precision) || !near(got.MeanF1, w.MeanF1) || !near(got.Tags, w.Tags) {
				t.Errorf("mean = %+v\nwant   %+v", got, w)
			}
		})
	}
}

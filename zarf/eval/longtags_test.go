package eval

import (
	"math"
	"testing"
)

func TestLongStrategies_Values(t *testing.T) {
	c := Candidate{Name: "x", ProposedBy: 3, Cos: []float64{0.2, 0.6, 0.5}, Mean: 0.4}
	third := (0.6 + 0.5 + 0.2) / 3
	want := []struct {
		name       string
		gate, rank float64
	}{
		{"mean", 0.4, 0.4},
		{"max", 0.6, 0.6},
		{"top2", 0.55, 0.55},
		{"top3", third, third},
		{"max>top2", 0.6, 0.55},
		{"max>top3", 0.6, third},
		{"max>mean", 0.6, 0.4},
		{"max>top2+rec", 0.6, 0.57},
	}
	got := LongStrategies()
	if len(got) != len(want) {
		t.Fatalf("got %d strategies, want %d", len(got), len(want))
	}
	for i, w := range want {
		s := got[i]
		if s.Name != w.name {
			t.Fatalf("strategy %d is %q, want %q", i, s.Name, w.name)
		}
		if g := s.Gate(c); !near(g, w.gate) {
			t.Errorf("%s gate = %v, want %v", w.name, g, w.gate)
		}
		if r := s.Rank(c); !near(r, w.rank) {
			t.Errorf("%s rank = %v, want %v", w.name, r, w.rank)
		}
	}
}

func TestLongStrategies_FewerChunksThanK(t *testing.T) {
	c := Candidate{Cos: []float64{0.4}, ProposedBy: 1}
	for _, s := range LongStrategies() {
		if s.Name == "top2" || s.Name == "top3" {
			if g := s.Gate(c); !near(g, 0.4) {
				t.Errorf("%s gate = %v, want 0.4", s.Name, g)
			}
		}
	}
}

func TestPick_GatesSortsAndCaps(t *testing.T) {
	s := Strategy{
		Name: "t",
		Gate: func(c Candidate) float64 { return c.Mean },
		Rank: func(c Candidate) float64 { return float64(c.ProposedBy) },
	}
	cands := []Candidate{
		{Name: "a", Mean: 0.5, ProposedBy: 1},
		{Name: "nan", Mean: math.NaN(), ProposedBy: 9},
		{Name: "low", Mean: 0.1, ProposedBy: 9},
		{Name: "b", Mean: 0.5, ProposedBy: 2},
		{Name: "c", Mean: 0.6, ProposedBy: 1},
		{Name: "d", Mean: 0.7, ProposedBy: 1},
	}
	got := Pick(cands, s, 0.5, 10)
	names := ""
	for _, c := range got {
		names += c.Name + " "
	}
	if names != "b a c d " {
		t.Errorf("order = %q, want %q", names, "b a c d ")
	}
	if got := Pick(cands, s, 0.5, 2); len(got) != 2 || got[0].Name != "b" || got[1].Name != "a" {
		t.Errorf("limit 2 = %+v", got)
	}
}

func matchDoc() []Topic {
	return []Topic{
		{Label: "Main", Kind: TopicMain, Aliases: []string{"quantum computing"}},
		{Label: "Early", Kind: TopicSide, Offset: 100, Aliases: []string{"error correction"}},
		{Label: "Late", Kind: TopicSide, Offset: 3000, Aliases: []string{"cryogenics"}},
		{Label: "Passing", Kind: TopicMention, Offset: 4000, Aliases: []string{"superconductors"}},
	}
}

func noSim(string, string) float64 { return 0 }

func TestMatchTopics(t *testing.T) {
	topics := matchDoc()

	t.Run("containment", func(t *testing.T) {
		s := MatchTopics([]string{"Quantum Computing Basics"}, topics, noSim, 0.5)
		if !near(s.MainRecall, 1) || !near(s.Recall, 1.0/3) || !near(s.Precision, 1) || s.Tags != 1 {
			t.Errorf("score = %+v", s)
		}
	})
	t.Run("short alias does not contain", func(t *testing.T) {
		tp := []Topic{{Kind: TopicMain, Aliases: []string{"AI"}}}
		s := MatchTopics([]string{"main ai thing"}, tp, noSim, 0.5)
		if s.Recall != 0 {
			t.Errorf("recall = %v, want 0", s.Recall)
		}
	})
	t.Run("sim at tau", func(t *testing.T) {
		sim := func(tag, alias string) float64 {
			if tag == "qec" && alias == "error correction" {
				return 0.5
			}
			return 0
		}
		s := MatchTopics([]string{"qec"}, topics, sim, 0.5)
		if !near(s.SideRecall, 0.5) || !near(s.Precision, 1) {
			t.Errorf("score = %+v", s)
		}
	})
	t.Run("below tau", func(t *testing.T) {
		sim := func(string, string) float64 { return 0.49 }
		s := MatchTopics([]string{"qec"}, topics, sim, 0.5)
		if s.Recall != 0 || s.Precision != 0 {
			t.Errorf("score = %+v", s)
		}
	})
	t.Run("mention lowers precision", func(t *testing.T) {
		s := MatchTopics([]string{"superconductors", "quantum computing"}, topics, noSim, 0.5)
		if !near(s.MentionRate, 1) || !near(s.Precision, 0.5) {
			t.Errorf("score = %+v", s)
		}
	})
	t.Run("late side", func(t *testing.T) {
		s := MatchTopics([]string{"cryogenics"}, topics, noSim, 0.5)
		if !near(s.LateSideRecall, 1) || !near(s.SideRecall, 0.5) {
			t.Errorf("score = %+v", s)
		}
	})
	t.Run("no tags", func(t *testing.T) {
		s := MatchTopics(nil, topics, noSim, 0.5)
		if s != (LongScore{}) {
			t.Errorf("score = %+v, want zero", s)
		}
	})
	t.Run("tie goes to first topic", func(t *testing.T) {
		tp := []Topic{
			{Kind: TopicSide, Aliases: []string{"alpha beta"}},
			{Kind: TopicSide, Aliases: []string{"alpha beta"}},
		}
		s := MatchTopics([]string{"alpha beta"}, tp, noSim, 0.5)
		if !near(s.SideRecall, 0.5) {
			t.Errorf("side recall = %v, want 0.5", s.SideRecall)
		}
	})
}

func TestLongScore_F1AndMean(t *testing.T) {
	if (LongScore{}).F1() != 0 {
		t.Error("F1 of zero score should be 0")
	}
	if f := (LongScore{Precision: 0.5, Recall: 1}).F1(); !near(f, 2.0/3) {
		t.Errorf("F1 = %v", f)
	}
	if MeanLongScores(nil) != (LongScore{}) {
		t.Error("mean of nothing should be zero")
	}
	m := MeanLongScores([]LongScore{{Recall: 1, Tags: 3}, {Recall: 0, Tags: 4}})
	if !near(m.Recall, 0.5) || m.Tags != 4 {
		t.Errorf("mean = %+v", m)
	}
}

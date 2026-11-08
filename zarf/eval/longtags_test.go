package eval

import "testing"

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
			{Kind: TopicMain, Aliases: []string{"alpha beta"}},
			{Kind: TopicSide, Aliases: []string{"alpha beta"}},
		}
		s := MatchTopics([]string{"alpha beta"}, tp, noSim, 0.5)
		if !near(s.MainRecall, 1) || !near(s.SideRecall, 0) {
			t.Errorf("main %v side %v, want 1 and 0", s.MainRecall, s.SideRecall)
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
	m := MeanLongScores([]LongScore{
		{Recall: 1, MainRecall: 0.2, SideRecall: 0.4, LateSideRecall: 0.6, MentionRate: 0.8, Precision: 0.1, Tags: 3},
		{Recall: 0, MainRecall: 0.4, SideRecall: 0.8, LateSideRecall: 0, MentionRate: 0.2, Precision: 0.3, Tags: 4},
	})
	want := LongScore{Recall: 0.5, MainRecall: 0.3, SideRecall: 0.6, LateSideRecall: 0.3, MentionRate: 0.5, Precision: 0.2, Tags: 4}
	if !near(m.Recall, want.Recall) || !near(m.MainRecall, want.MainRecall) ||
		!near(m.SideRecall, want.SideRecall) || !near(m.LateSideRecall, want.LateSideRecall) ||
		!near(m.MentionRate, want.MentionRate) || !near(m.Precision, want.Precision) || m.Tags != want.Tags {
		t.Errorf("mean = %+v, want %+v", m, want)
	}
}

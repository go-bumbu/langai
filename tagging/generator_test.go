package tagging

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/go-bumbu/langai"
)

var errBoom = errors.New("boom")

type prompt struct{ system, user string }

// stubLLM answers a prompt with byUser[user] when it holds the user message
// and with answer otherwise, or fails with err, and records what it was asked.
// hook, when set, runs first with the prompt's number (1 for the first), and
// an error from it fails that prompt.
type stubLLM struct {
	answer string
	byUser map[string]string
	err    error
	hook   func(n int) error
	asked  []prompt
}

func (s *stubLLM) Complete(_ context.Context, system, user string) (string, error) {
	s.asked = append(s.asked, prompt{system, user})
	if s.hook != nil {
		if err := s.hook(len(s.asked)); err != nil {
			return "", err
		}
	}
	if a, ok := s.byUser[user]; ok {
		return a, s.err
	}
	return s.answer, s.err
}

type embedCall struct {
	role  langai.Role
	texts []string
}

// vecEmbedder gives each known text a fixed vector, so a test can state the
// exact score it expects, and records what it was asked to embed. An unknown
// text is an error: the test and the code disagree on what gets embedded.
type vecEmbedder struct {
	vecs  map[string][]float32
	err   error
	drop  bool // return one vector too few
	calls []embedCall
}

func (e *vecEmbedder) Dimensions() int { return 3 }

func (e *vecEmbedder) Embed(_ context.Context, role langai.Role, texts []string) ([][]float32, error) {
	e.calls = append(e.calls, embedCall{role, slices.Clone(texts)})
	if e.err != nil {
		return nil, e.err
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, ok := e.vecs[t]
		if !ok {
			return nil, fmt.Errorf("no vector for %q", t)
		}
		out[i] = v
	}
	if e.drop {
		out = out[:len(out)-1]
	}
	return out, nil
}

// bread is the text of most tests. angled gives it and its tags the vectors
// that make each tag's score the cosine written next to it.
const bread = "Baking sourdough bread at home"

func angled() map[string][]float32 {
	return map[string][]float32{
		bread:   {1, 0, 0},
		"alpha": {0.6, 0.8, 0}, // 0.6
		"beta":  {1, 0, 0},     // 1
		"gamma": {0, 1, 0},     // 0
		"delta": {0.8, 0.6, 0}, // 0.8
	}
}

func newGenerator(t *testing.T, answer string, vecs map[string][]float32, cfg GeneratorConfig) (*Generator, *stubLLM, *vecEmbedder) {
	t.Helper()
	llm := &stubLLM{answer: answer}
	emb := &vecEmbedder{vecs: vecs}
	g, err := NewGenerator(llm, emb, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return g, llm, emb
}

func requireTags(t *testing.T, got, want []Tag) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i].Name != want[i].Name || math.Abs(got[i].Score-want[i].Score) > 1e-6 {
			t.Fatalf("tag %d: got %+v, want %+v (all: %v)", i, got[i], want[i], got)
		}
	}
}

func TestNew_RequiresCompleterAndEmbedder(t *testing.T) {
	if _, err := NewGenerator(nil, &vecEmbedder{}, GeneratorConfig{}); err == nil {
		t.Error("nil Completer accepted")
	}
	if _, err := NewGenerator(&stubLLM{}, nil, GeneratorConfig{}); err == nil {
		t.Error("nil Embedder accepted")
	}
}

func TestNew_RejectsNegativeSettings(t *testing.T) {
	for _, cfg := range []GeneratorConfig{{Candidates: -1}, {MaxBytes: -1}} {
		if _, err := NewGenerator(&stubLLM{}, &vecEmbedder{}, cfg); err == nil {
			t.Errorf("%+v accepted", cfg)
		}
	}
}

func TestNew_FillsInTheDefaults(t *testing.T) {
	if DefaultMaxBytes != 65536 || DefaultChunkBytes != 2048 {
		t.Fatalf("DefaultMaxBytes %d, DefaultChunkBytes %d, want 65536 and 2048", DefaultMaxBytes, DefaultChunkBytes)
	}
	g, _, _ := newGenerator(t, "alpha", angled(), GeneratorConfig{})
	want := GeneratorConfig{Candidates: DefaultCandidates, MaxBytes: DefaultMaxBytes, ChunkBytes: DefaultChunkBytes}
	if g.cfg != want {
		t.Errorf("config %+v, want %+v", g.cfg, want)
	}
}

func TestGenerate_ScoresTagsAndReturnsTheBestFirst(t *testing.T) {
	g, _, _ := newGenerator(t, "alpha\nbeta\ngamma\ndelta", angled(), GeneratorConfig{})
	got, err := g.Generate(context.Background(), bread, -1)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"beta", 1}, {"delta", 0.8}, {"alpha", 0.6}, {"gamma", 0}})
}

func TestGenerate_DropsTagsBelowTheCutoff(t *testing.T) {
	g, _, _ := newGenerator(t, "alpha\nbeta\ngamma\ndelta", angled(), GeneratorConfig{})
	got, err := g.Generate(context.Background(), bread, 0.7)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"beta", 1}, {"delta", 0.8}})
}

func TestGenerate_KeepsATagScoringExactlyTheCutoff(t *testing.T) {
	g, _, _ := newGenerator(t, "alpha\ndelta", angled(), GeneratorConfig{})
	at := cosine(angled()[bread], angled()["delta"])
	got, err := g.Generate(context.Background(), bread, at)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"delta", at}})
}

func TestGenerate_ReturnsNothingForANaNCutoff(t *testing.T) {
	g, _, _ := newGenerator(t, "alpha\nbeta", angled(), GeneratorConfig{})
	got, err := g.Generate(context.Background(), bread, math.NaN())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("a NaN cutoff returned %v", got)
	}
}

func TestGenerate_KeepsTheModelsOrderForEqualScores(t *testing.T) {
	vecs := map[string][]float32{bread: {1, 0, 0}, "zeta": {0, 1, 0}, "alpha": {0, 1, 0}, "beta": {1, 0, 0}}
	g, _, _ := newGenerator(t, "zeta\nalpha\nbeta", vecs, GeneratorConfig{})
	got, err := g.Generate(context.Background(), bread, -1)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"beta", 1}, {"zeta", 0}, {"alpha", 0}})
}

// A Tagger embeds an item's text as a query and a tag's name as a document; a
// Generator does the same, so its scores sit on the scale of the Tagger's
// NameMinScore.
func TestGenerate_EmbedsTheTextAsAQueryAndTheTagsAsDocuments(t *testing.T) {
	g, _, emb := newGenerator(t, "alpha\nbeta", angled(), GeneratorConfig{})
	if _, err := g.Generate(context.Background(), bread, 0); err != nil {
		t.Fatal(err)
	}
	want := []embedCall{
		{langai.RoleQuery, []string{bread}},
		{langai.RoleDocument, []string{"alpha", "beta"}},
	}
	if !reflect.DeepEqual(emb.calls, want) {
		t.Errorf("embedded %v, want %v", emb.calls, want)
	}
}

func TestGenerate_EmbedsTheCleanedNamesOnce(t *testing.T) {
	answer := "Here are the tags:\n1. Alpha\n2) \"beta\"\n- #gamma\n* alpha\n"
	vecs := map[string][]float32{bread: {1, 0, 0}, "Alpha": {1, 0, 0}, "beta": {1, 0, 0}, "gamma": {1, 0, 0}}
	g, _, emb := newGenerator(t, answer, vecs, GeneratorConfig{})
	got, err := g.Generate(context.Background(), bread, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"Alpha", 1}, {"beta", 1}, {"gamma", 1}})
	if want := []string{"Alpha", "beta", "gamma"}; !slices.Equal(emb.calls[1].texts, want) {
		t.Errorf("embedded %v, want %v", emb.calls[1].texts, want)
	}
}

func TestGenerate_ScoresOnlyTheFirstCandidates(t *testing.T) {
	g, _, emb := newGenerator(t, "alpha\nbeta\ngamma\ndelta", angled(), GeneratorConfig{Candidates: 2})
	got, err := g.Generate(context.Background(), bread, -1)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"beta", 1}, {"alpha", 0.6}})
	if want := []string{"alpha", "beta"}; !slices.Equal(emb.calls[1].texts, want) {
		t.Errorf("embedded %v, want %v", emb.calls[1].texts, want)
	}
}

func TestGenerate_AsksTheModelForTheConfiguredNumberOfTags(t *testing.T) {
	for _, tc := range []struct {
		cfg  GeneratorConfig
		want string
	}{
		{GeneratorConfig{}, "up to 10 tags"},
		{GeneratorConfig{Candidates: 3}, "up to 3 tags"},
	} {
		g, llm, _ := newGenerator(t, "alpha", angled(), tc.cfg)
		if _, err := g.Generate(context.Background(), bread, 0); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(llm.asked[0].system, tc.want) {
			t.Errorf("%+v: the system prompt does not say %q:\n%s", tc.cfg, tc.want, llm.asked[0].system)
		}
		if llm.asked[0].user != bread {
			t.Errorf("the user message is %q, want the text", llm.asked[0].user)
		}
	}
}

// The model and the embedder must see the same text: a tag about words only
// the model read would score low for a reason the caller cannot see.
func TestGenerate_CutsTheTextForTheModelAndTheEmbedderAlike(t *testing.T) {
	text := strings.Repeat("é", 10) // 20 bytes; a cut at 5 falls inside a rune
	vecs := map[string][]float32{"éé": {1, 0, 0}, "alpha": {1, 0, 0}}
	g, llm, emb := newGenerator(t, "alpha", vecs, GeneratorConfig{MaxBytes: 5})
	if _, err := g.Generate(context.Background(), text, 0); err != nil {
		t.Fatal(err)
	}
	if llm.asked[0].user != "éé" {
		t.Errorf("the model read %q, want %q", llm.asked[0].user, "éé")
	}
	if emb.calls[0].texts[0] != "éé" {
		t.Errorf("the embedder read %q, want %q", emb.calls[0].texts[0], "éé")
	}
}

func TestGenerate_ReplacesInvalidUTF8(t *testing.T) {
	vecs := map[string][]float32{"bread \uFFFD": {1, 0, 0}, "alpha": {1, 0, 0}}
	g, llm, _ := newGenerator(t, "alpha", vecs, GeneratorConfig{})
	if _, err := g.Generate(context.Background(), "bread \xff", 0); err != nil {
		t.Fatal(err)
	}
	if llm.asked[0].user != "bread \uFFFD" {
		t.Errorf("the model read %q", llm.asked[0].user)
	}
}

func TestGenerate_AsksNobodyAboutABlankText(t *testing.T) {
	g, llm, emb := newGenerator(t, "alpha", angled(), GeneratorConfig{})
	got, err := g.Generate(context.Background(), " \n\t ", 0)
	if err != nil || got != nil {
		t.Fatalf("got %v, %v", got, err)
	}
	if len(llm.asked) != 0 || len(emb.calls) != 0 {
		t.Errorf("a blank text reached the model (%d) or the embedder (%d)", len(llm.asked), len(emb.calls))
	}
}

func TestGenerate_AnswerWithoutTagsIsNotAnError(t *testing.T) {
	g, _, emb := newGenerator(t, "\n \n", angled(), GeneratorConfig{})
	got, err := g.Generate(context.Background(), bread, 0)
	if err != nil || got != nil {
		t.Fatalf("got %v, %v", got, err)
	}
	if len(emb.calls) != 0 {
		t.Errorf("embedded %v although the model wrote no tag", emb.calls)
	}
}

func TestGenerate_ReportsAModelFailure(t *testing.T) {
	g, llm, emb := newGenerator(t, "alpha", angled(), GeneratorConfig{})
	llm.err = errBoom
	_, err := g.Generate(context.Background(), bread, 0)
	if !errors.Is(err, errBoom) {
		t.Fatalf("got %v, want it to wrap %v", err, errBoom)
	}
	if len(emb.calls) != 0 {
		t.Errorf("embedded %v after the model failed", emb.calls)
	}
}

func TestGenerate_ReportsAnEmbedderFailure(t *testing.T) {
	g, _, emb := newGenerator(t, "alpha", angled(), GeneratorConfig{})
	emb.err = errBoom
	if _, err := g.Generate(context.Background(), bread, 0); !errors.Is(err, errBoom) {
		t.Fatalf("got %v, want it to wrap %v", err, errBoom)
	}
}

func TestGenerate_RejectsAnEmbedderThatReturnsTooFewVectors(t *testing.T) {
	g, _, emb := newGenerator(t, "alpha\nbeta", angled(), GeneratorConfig{})
	emb.drop = true
	if _, err := g.Generate(context.Background(), bread, 0); err == nil {
		t.Fatal("a missing vector went unnoticed")
	}
}

func TestGenerate_RejectsATagVectorOfAnotherLength(t *testing.T) {
	vecs := angled()
	vecs["alpha"] = []float32{1, 0}
	g, _, _ := newGenerator(t, "alpha", vecs, GeneratorConfig{})
	if _, err := g.Generate(context.Background(), bread, 0); err == nil {
		t.Fatal("a vector of another length was scored")
	}
}

func TestGenerate_StopsOnACanceledContext(t *testing.T) {
	g, llm, _ := newGenerator(t, "alpha", angled(), GeneratorConfig{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := g.Generate(ctx, bread, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if len(llm.asked) != 0 {
		t.Error("the model was asked after the context was canceled")
	}
}

func TestParseTags(t *testing.T) {
	for _, tc := range []struct {
		name   string
		answer string
		limit  int
		want   []string
	}{
		{"one per line", "a\nb\nc", 10, []string{"a", "b", "c"}},
		{"windows line endings", "a\r\nb\r\n", 10, []string{"a", "b"}},
		{"comma and semicolon separated", "a, b; c", 10, []string{"a", "b", "c"}},
		{"numbers and bullets", "1. a\n2) b\n- c\n* d\n• e", 10, []string{"a", "b", "c", "d", "e"}},
		{"hashtags, quotes and emphasis", "#a\n\"b\"\n`c`\n**d**\n“e”", 10, []string{"a", "b", "c", "d", "e"}},
		{"trailing punctuation", "a.\nb!\nc?", 10, []string{"a", "b", "c"}},
		{"punctuation inside a tag", "node.js\nc#\nc++", 10, []string{"node.js", "c#", "c++"}},
		{"leading digits that are not a list number", "3d printing\n2024 elections", 10, []string{"3d printing", "2024 elections"}},
		{"extra spaces", "  sourdough    bread  ", 10, []string{"sourdough bread"}},
		{"a line ending in a colon is a heading", "Here are the tags:\na\nb", 10, []string{"a", "b"}},
		{"duplicates ignoring case", "Bread\nbread\nBREAD\nbaking", 10, []string{"Bread", "baking"}},
		{"a sentence is not a tag", "one two three four five\nok", 10, []string{"ok"}},
		{"a long word run is not a tag", strings.Repeat("x", 41) + "\nok", 10, []string{"ok"}},
		{"four words are a tag", "one two three four", 10, []string{"one two three four"}},
		{"limit", "a\nb\nc", 2, []string{"a", "b"}},
		{"nothing", " \n\n ", 10, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := parseTags(tc.answer, tc.limit, cleanTag); !slices.Equal(got, tc.want) {
				t.Errorf("parseTags(%q) = %q, want %q", tc.answer, got, tc.want)
			}
		})
	}
}

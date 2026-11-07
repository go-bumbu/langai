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
	"unicode/utf8"

	"github.com/go-bumbu/langai"
)

// words returns n distinct words of six bytes, first letter then a number
// from from on, joined by spaces: 7n-1 bytes.
func words(first byte, from, n int) string {
	ws := make([]string, n)
	for i := range ws {
		ws[i] = fmt.Sprintf("%c%05d", first, from+i)
	}
	return strings.Join(ws, " ")
}

// threeWindows is a text the long path reads in three windows.
var threeWindows = words('w', 0, 2500)

// flatEmbedder gives every text the vector vec, except short, which gets a
// vector of two dimensions, and records what it was asked to embed. With
// dropAt n, the n-th call (1 for the first) returns one vector too few.
type flatEmbedder struct {
	vec    []float32
	short  string
	dropAt int
	err    error
	calls  []embedCall
}

func (e *flatEmbedder) Dimensions() int { return len(e.vec) }

func (e *flatEmbedder) Embed(_ context.Context, role langai.Role, texts []string) ([][]float32, error) {
	e.calls = append(e.calls, embedCall{role, slices.Clone(texts)})
	if e.err != nil {
		return nil, e.err
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = e.vec
		if t == e.short {
			out[i] = []float32{1, 0}
		}
	}
	if len(e.calls) == e.dropAt {
		out = out[:len(out)-1]
	}
	return out, nil
}

func newFlat() *flatEmbedder { return &flatEmbedder{vec: []float32{1, 0, 0}} }

func newLongGenerator(t *testing.T, llm *stubLLM, emb langai.Embedder, cfg GeneratorConfig, opts ...GeneratorOption) *Generator {
	t.Helper()
	g, err := NewGenerator(llm, emb, cfg, opts...)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// windowsOf returns the windows the long path reads text in, and fails the
// test unless there are n.
func windowsOf(t *testing.T, text string, n int) []string {
	t.Helper()
	ws := Chunk(text, windowBytes)
	if len(ws) != n {
		t.Fatalf("the text makes %d windows, want %d", len(ws), n)
	}
	return ws
}

func users(asked []prompt) []string {
	var out []string
	for _, p := range asked {
		out = append(out, p.user)
	}
	return out
}

// scored is a unit vector whose cosine to {1, 0, 0} is s.
func scored(s float64) []float32 {
	return []float32{float32(s), float32(math.Sqrt(1 - s*s)), 0}
}

func TestNew_RejectsNegativeChunkBytes(t *testing.T) {
	if _, err := NewGenerator(&stubLLM{}, &vecEmbedder{}, GeneratorConfig{ChunkBytes: -1}); err == nil {
		t.Error("a negative ChunkBytes accepted")
	}
}

func TestLongScore(t *testing.T) {
	if got := longScore([]float64{0.2, 0.6, 0.5}); got != 0.6 {
		t.Errorf("longScore = %v, want 0.6", got)
	}
}

func TestGenerate_ShortTextAsksOnce(t *testing.T) {
	text := strings.Repeat("x", DefaultChunkBytes)
	llm, emb := &stubLLM{answer: "alpha\nbeta"}, newFlat()
	g := newLongGenerator(t, llm, emb, GeneratorConfig{})
	if _, err := g.Generate(context.Background(), text, 0); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(users(llm.asked), []string{text}) {
		t.Errorf("asked %d times, want once with the text", len(llm.asked))
	}
	want := []embedCall{{langai.RoleQuery, []string{text}}, {langai.RoleDocument, []string{"alpha", "beta"}}}
	if !reflect.DeepEqual(emb.calls, want) {
		t.Errorf("embedded %v, want %v", emb.calls, want)
	}
}

func TestGenerate_LongTextAsksOncePerWindow(t *testing.T) {
	windows := windowsOf(t, threeWindows, 3)
	llm, emb := &stubLLM{answer: "alpha\nbeta"}, newFlat()
	g := newLongGenerator(t, llm, emb, GeneratorConfig{})
	if _, err := g.Generate(context.Background(), threeWindows, 0); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(users(llm.asked), windows) {
		t.Errorf("the user messages are not the text's windows, as they are: got %d messages", len(llm.asked))
	}
	for _, p := range llm.asked {
		if p.system != g.prompt {
			t.Errorf("a window was asked with another system prompt: %q", p.system)
		}
	}
	want := []embedCall{
		{langai.RoleQuery, Chunk(threeWindows, DefaultChunkBytes)},
		{langai.RoleDocument, []string{"alpha", "beta"}},
	}
	if !reflect.DeepEqual(emb.calls, want) {
		var got []string
		for _, c := range emb.calls {
			got = append(got, fmt.Sprintf("%d texts as %s", len(c.texts), c.role))
		}
		t.Errorf("embedded %v, want the %d chunks as queries in one call and the 2 names as documents in another", got, len(want[0].texts))
	}
}

func TestGenerate_LongTextMergesNamesAcrossWindows(t *testing.T) {
	ws := windowsOf(t, threeWindows, 3)
	llm := &stubLLM{byUser: map[string]string{ws[0]: "Go\nrust", ws[1]: "go\npython", ws[2]: "RUST"}}
	emb := newFlat()
	g := newLongGenerator(t, llm, emb, GeneratorConfig{})
	got, err := g.Generate(context.Background(), threeWindows, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"Go", 1}, {"rust", 1}, {"python", 1}})
	if want := []string{"Go", "rust", "python"}; !slices.Equal(emb.calls[1].texts, want) {
		t.Errorf("embedded %v, want %v", emb.calls[1].texts, want)
	}
}

// A name scores by the chunk it fits best: a topic of one section is not
// diluted by the sections that are about something else.
func TestGenerate_LongTextScoresEachChunk(t *testing.T) {
	chunks := Chunk(threeWindows, DefaultChunkBytes)
	if len(chunks) < 3 {
		t.Fatalf("%d chunks, want at least 3", len(chunks))
	}
	vecs := map[string][]float32{
		"volcano": {0.1, 0.8, float32(math.Sqrt(0.35))}, // 0.8 to chunk 3, 0.1 to the others
		"noise":   {0.3, 0.3, float32(math.Sqrt(0.82))}, // 0.3 to every chunk
	}
	for i, c := range chunks {
		vecs[c] = []float32{1, 0, 0}
		if i == 2 {
			vecs[c] = []float32{0, 1, 0}
		}
	}
	llm := &stubLLM{answer: "noise\nvolcano"}
	g := newLongGenerator(t, llm, &vecEmbedder{vecs: vecs}, GeneratorConfig{})
	got, err := g.Generate(context.Background(), threeWindows, DefaultGeneratorMinScore)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"volcano", 0.8}})
	got, err = g.Generate(context.Background(), threeWindows, -1)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"volcano", 0.8}, {"noise", 0.3}})
}

func TestGenerate_LongTextReturnsNothingForANaNCutoff(t *testing.T) {
	g := newLongGenerator(t, &stubLLM{answer: "alpha"}, newFlat(), GeneratorConfig{})
	got, err := g.Generate(context.Background(), threeWindows, math.NaN())
	if err != nil || got != nil {
		t.Fatalf("got %v, %v", got, err)
	}
}

func TestGenerate_LongTextCapsAtCandidates(t *testing.T) {
	ws := windowsOf(t, threeWindows, 3)
	llm := &stubLLM{byUser: map[string]string{
		ws[0]: "a1\na2\na3\na4",
		ws[1]: "b1\nb2\nb3\nb4",
		ws[2]: "c1\nc2\nc3\nc4",
	}}
	vecs := map[string][]float32{
		"a1": scored(0.1), "a2": scored(0.5), "a3": scored(0.2), "a4": scored(0.3),
		"b1": scored(0.6), "b2": scored(0.15), "b3": scored(0.7), "b4": scored(0.25),
		"c1": scored(0.9), "c2": scored(0.05), "c3": scored(0.4), "c4": scored(0.35),
	}
	for _, c := range Chunk(threeWindows, DefaultChunkBytes) {
		vecs[c] = []float32{1, 0, 0}
	}
	emb := &vecEmbedder{vecs: vecs}
	g := newLongGenerator(t, llm, emb, GeneratorConfig{Candidates: 4})
	got, err := g.Generate(context.Background(), threeWindows, -1)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"c1", 0.9}, {"b3", 0.7}, {"b1", 0.6}, {"a2", 0.5}})
	if n := len(emb.calls[1].texts); n != 12 {
		t.Errorf("embedded %d names, want all 12 the windows proposed", n)
	}
}

func TestGenerate_LongTextKeepsTheProposedOrderForEqualScores(t *testing.T) {
	ws := windowsOf(t, threeWindows, 3)
	llm := &stubLLM{byUser: map[string]string{ws[0]: "zeta", ws[1]: "alpha\nbeta", ws[2]: "gamma"}}
	vecs := map[string][]float32{"zeta": scored(0.5), "alpha": scored(0.9), "beta": scored(0.5), "gamma": scored(0.5)}
	for _, c := range Chunk(threeWindows, DefaultChunkBytes) {
		vecs[c] = []float32{1, 0, 0}
	}
	g := newLongGenerator(t, llm, &vecEmbedder{vecs: vecs}, GeneratorConfig{})
	got, err := g.Generate(context.Background(), threeWindows, -1)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"alpha", 0.9}, {"zeta", 0.5}, {"beta", 0.5}, {"gamma", 0.5}})
}

func TestGenerate_LongTextReadsAtMostMaxBytes(t *testing.T) {
	head := words('w', 0, 1500)
	text := head + " " + words('t', 0, 500)
	llm, emb := &stubLLM{answer: "alpha"}, newFlat()
	g := newLongGenerator(t, llm, emb, GeneratorConfig{MaxBytes: len(head)})
	if _, err := g.Generate(context.Background(), text, 0); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(users(llm.asked), Chunk(head, windowBytes)) {
		t.Errorf("the model read %d windows, want the windows of the first MaxBytes", len(llm.asked))
	}
	for _, c := range emb.calls {
		for _, s := range c.texts {
			if strings.Contains(s, "t00") {
				t.Fatalf("the embedder read words past MaxBytes: %q", s)
			}
		}
	}
}

func TestGenerate_JustOverChunkBytesTakesTheLongPath(t *testing.T) {
	text := strings.Repeat("x ", DefaultChunkBytes/2) + "y" // 2049 bytes
	llm, emb := &stubLLM{answer: "alpha"}, newFlat()
	g := newLongGenerator(t, llm, emb, GeneratorConfig{})
	if _, err := g.Generate(context.Background(), text, 0); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(users(llm.asked), []string{text}) {
		t.Errorf("asked %d times, want once with the one window", len(llm.asked))
	}
	if chunks := emb.calls[0].texts; len(chunks) != 2 || !slices.Equal(chunks, Chunk(text, DefaultChunkBytes)) {
		t.Errorf("embedded %d texts as queries, want the text's 2 chunks", len(chunks))
	}
}

func TestGenerate_ChunkBytesSetsWhereTheLongPathStarts(t *testing.T) {
	text := words('w', 0, 50) // 349 bytes
	llm, emb := &stubLLM{answer: "alpha"}, newFlat()
	g := newLongGenerator(t, llm, emb, GeneratorConfig{ChunkBytes: 100})
	if _, err := g.Generate(context.Background(), text, 0); err != nil {
		t.Fatal(err)
	}
	if len(llm.asked) != 1 || !slices.Equal(emb.calls[0].texts, Chunk(text, 100)) {
		t.Errorf("asked %d times and embedded %d queries, want once and the text's chunks of 100 bytes", len(llm.asked), len(emb.calls[0].texts))
	}
}

func TestGenerate_MaxBytesBelowChunkBytesStaysShort(t *testing.T) {
	llm, emb := &stubLLM{answer: "alpha"}, newFlat()
	g := newLongGenerator(t, llm, emb, GeneratorConfig{MaxBytes: 1000})
	if _, err := g.Generate(context.Background(), threeWindows, 0); err != nil {
		t.Fatal(err)
	}
	if len(llm.asked) != 1 || len(llm.asked[0].user) > 1000 {
		t.Fatalf("asked %d times, want once with at most 1000 bytes", len(llm.asked))
	}
	if q := emb.calls[0].texts; !slices.Equal(q, []string{llm.asked[0].user}) {
		t.Errorf("embedded %d queries, want the text the model read", len(q))
	}
}

func TestGenerate_LongTextSkipsWindowsWithoutTags(t *testing.T) {
	ws := windowsOf(t, threeWindows, 3)
	llm := &stubLLM{byUser: map[string]string{ws[0]: "\n \n", ws[1]: "alpha", ws[2]: "Here are the tags:\n"}}
	g := newLongGenerator(t, llm, newFlat(), GeneratorConfig{})
	got, err := g.Generate(context.Background(), threeWindows, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"alpha", 1}})
	if len(llm.asked) != 3 {
		t.Errorf("asked %d times, want once per window", len(llm.asked))
	}
}

func TestGenerate_LongTextReturnsNilWhenNoWindowHasTags(t *testing.T) {
	llm, emb := &stubLLM{answer: "\n"}, newFlat()
	g := newLongGenerator(t, llm, emb, GeneratorConfig{})
	got, err := g.Generate(context.Background(), threeWindows, 0)
	if err != nil || got != nil {
		t.Fatalf("got %v, %v", got, err)
	}
	if len(llm.asked) != 3 || len(emb.calls) != 0 {
		t.Errorf("asked %d times and embedded %v, want 3 and nothing", len(llm.asked), emb.calls)
	}
}

func TestGenerate_LongTextWritesTreeTags(t *testing.T) {
	g := newLongGenerator(t, &stubLLM{answer: "cooking > baking"}, newFlat(), GeneratorConfig{}, WithTreeTags())
	got, err := g.Generate(context.Background(), threeWindows, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"cooking / baking", 1}})
}

func TestGenerate_LongTextKeepsUTF8Valid(t *testing.T) {
	text := strings.Repeat("é€\xff", 2000) // no whitespace: every cut is on a rune
	llm, emb := &stubLLM{answer: "alpha"}, newFlat()
	g := newLongGenerator(t, llm, emb, GeneratorConfig{MaxBytes: 9001})
	if _, err := g.Generate(context.Background(), text, 0); err != nil {
		t.Fatal(err)
	}
	if len(llm.asked) < 2 {
		t.Fatalf("asked %d times, want the long path", len(llm.asked))
	}
	for _, p := range llm.asked {
		if !utf8.ValidString(p.user) {
			t.Errorf("the model read invalid UTF-8: %q", p.user)
		}
	}
	for _, c := range emb.calls {
		for _, s := range c.texts {
			if !utf8.ValidString(s) {
				t.Errorf("the embedder read invalid UTF-8: %q", s)
			}
		}
	}
}

func TestGenerate_LongTextReportsAModelFailure(t *testing.T) {
	llm, emb := &stubLLM{answer: "alpha"}, newFlat()
	llm.hook = func(n int) error {
		if n == 2 {
			return errBoom
		}
		return nil
	}
	g := newLongGenerator(t, llm, emb, GeneratorConfig{})
	_, err := g.Generate(context.Background(), threeWindows, 0)
	if !errors.Is(err, errBoom) || !strings.HasPrefix(err.Error(), "tagging: write tags: ") {
		t.Fatalf("got %v, want a write-tags error wrapping %v", err, errBoom)
	}
	if len(llm.asked) != 2 || len(emb.calls) != 0 {
		t.Errorf("asked %d times and embedded %v after the model failed", len(llm.asked), emb.calls)
	}
}

func TestGenerate_LongTextStopsOnACanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	llm, emb := &stubLLM{answer: "alpha"}, newFlat()
	llm.hook = func(int) error { cancel(); return nil }
	g := newLongGenerator(t, llm, emb, GeneratorConfig{})
	if _, err := g.Generate(ctx, threeWindows, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if len(llm.asked) != 1 || len(emb.calls) != 0 {
		t.Errorf("asked %d times and embedded %v after the context was canceled", len(llm.asked), emb.calls)
	}
}

func TestGenerate_LongTextReportsAnEmbedderFailure(t *testing.T) {
	emb := newFlat()
	emb.err = errBoom
	g := newLongGenerator(t, &stubLLM{answer: "alpha"}, emb, GeneratorConfig{})
	if _, err := g.Generate(context.Background(), threeWindows, 0); !errors.Is(err, errBoom) {
		t.Fatalf("got %v, want it to wrap %v", err, errBoom)
	}
}

func TestGenerate_LongTextRejectsWrongVectorCounts(t *testing.T) {
	for _, tc := range []struct {
		name   string
		dropAt int
	}{{"chunks", 1}, {"names", 2}} {
		t.Run(tc.name, func(t *testing.T) {
			emb := newFlat()
			emb.dropAt = tc.dropAt
			g := newLongGenerator(t, &stubLLM{answer: "alpha\nbeta"}, emb, GeneratorConfig{})
			if _, err := g.Generate(context.Background(), threeWindows, 0); err == nil {
				t.Fatal("a missing vector went unnoticed")
			}
		})
	}
}

func TestGenerate_LongTextRejectsVectorsOfAnotherLength(t *testing.T) {
	for _, tc := range []struct{ name, short string }{
		{"a name", "beta"},
		{"a chunk", Chunk(threeWindows, DefaultChunkBytes)[1]},
	} {
		t.Run(tc.name, func(t *testing.T) {
			emb := newFlat()
			emb.short = tc.short
			g := newLongGenerator(t, &stubLLM{answer: "alpha\nbeta"}, emb, GeneratorConfig{})
			if _, err := g.Generate(context.Background(), threeWindows, 0); err == nil {
				t.Fatal("a vector of another length was scored")
			}
		})
	}
}

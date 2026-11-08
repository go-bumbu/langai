package tagging

import (
	"context"
	"slices"
	"strings"
	"testing"
)

// askedPrompt returns the system prompt a Generator built with opts sends.
func askedPrompt(t *testing.T, opts ...GeneratorOption) string {
	t.Helper()
	llm := &stubLLM{answer: "alpha"}
	g, err := NewGenerator(llm, &vecEmbedder{vecs: angled()}, GeneratorConfig{}, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Generate(context.Background(), bread, 0); err != nil {
		t.Fatal(err)
	}
	return llm.asked[0].system
}

func TestGeneratorOptions_Language(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts []GeneratorOption
		want string
	}{
		{"default", nil, "Write every tag in the language of the text."},
		{"native", []GeneratorOption{WithNativeLanguageTags()}, "Write every tag in the language of the text."},
		{"native tree rule", []GeneratorOption{WithTreeTags()}, "Every tag must be in the language of the text, even though these instructions"},
		{"native tree reminder", []GeneratorOption{WithTreeTags()}, "Remember: write every tag in the language of the text."},
		{"spanish rule", []GeneratorOption{WithLanguageTags("es")}, "Every tag must be in Spanish (es). The text may be in another language: translate"},
		{"spanish reminder", []GeneratorOption{WithLanguageTags("es")}, "Remember: write every tag in Spanish (es), whatever the language of the text."},
		{"region", []GeneratorOption{WithLanguageTags("pt-BR")}, "in Brazilian Portuguese (pt-BR),"},
		{"later wins", []GeneratorOption{WithLanguageTags("en"), WithNativeLanguageTags()}, "in the language of the text"},
		{"later wins again", []GeneratorOption{WithNativeLanguageTags(), WithLanguageTags("de")}, "in German (de)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := askedPrompt(t, tc.opts...); !strings.Contains(got, tc.want) {
				t.Errorf("prompt does not say %q:\n%s", tc.want, got)
			}
		})
	}
}

func TestWithLanguageTags_RefusesABadTag(t *testing.T) {
	for _, lang := range []string{"", "not a language!", "x"} {
		if _, err := NewGenerator(&stubLLM{}, &vecEmbedder{}, GeneratorConfig{}, WithLanguageTags(lang)); err == nil {
			t.Errorf("WithLanguageTags(%q): want an error", lang)
		}
	}
}

func TestGeneratorOptions_TreeAsksForPathsAndShowsTheExistingTree(t *testing.T) {
	flat := askedPrompt(t)
	if strings.Contains(flat, " / ") || !strings.Contains(flat, "one to three words") {
		t.Errorf("flat prompt:\n%s", flat)
	}
	own := askedPrompt(t, WithTreeTags())
	if !strings.Contains(own, `separated by " / "`) || strings.Contains(own, "exist already") {
		t.Errorf("tree prompt without a tree:\n%s", own)
	}
	ext := askedPrompt(t, WithTreeTags("dev / go", "  ", "cooking / baking"), WithLanguageTags("en"))
	for _, want := range []string{"exist already, one per line:\ndev / go\ncooking / baking\n", "written exactly as above", "in English (en)", "Existing paths keep their words"} {
		if !strings.Contains(ext, want) {
			t.Errorf("tree prompt does not say %q:\n%s", want, ext)
		}
	}
	native := askedPrompt(t, WithTreeTags("dev / go"))
	if want := "the levels you add are in the language of the text"; !strings.Contains(native, want) {
		t.Errorf("native tree prompt does not say %q:\n%s", want, native)
	}
	if flat := askedPrompt(t); strings.Contains(flat, "levels you add") || strings.Contains(flat, "Every tag must be") {
		t.Errorf("the flat prompt in the text's language states the rule only last:\n%s", flat)
	}
}

func TestTreeShown_CapsTheExistingTree(t *testing.T) {
	var paths []string
	for range 400 {
		paths = append(paths, "category / subcategory") // 23 bytes with its newline
	}
	shown := treeShown(paths)
	size := 0
	for _, p := range shown {
		size += len(p) + 1
	}
	if len(shown) == 0 || size > MaxTreeBytes || len(shown) == len(paths) {
		t.Errorf("shown %d paths, %d bytes", len(shown), size)
	}
}

func TestCleanPath(t *testing.T) {
	for in, want := range map[string]string{
		"dev / go / error handling":                 "dev / go / error handling",
		"1. Cooking/Baking":                         "Cooking / Baking",
		"- travel > city guides > Basel.":           "travel / city guides / Basel",
		"**events › hackathons**":                   "events / hackathons",
		"a / b / c / d / e / f":                     "a / b / c / d",
		"dev // go /":                               "dev / go",
		"single":                                    "single",
		"dev / this level is far too long a phrase": "",
		"   ":                                       "",
	} {
		if got := cleanPath(in); got != want {
			t.Errorf("cleanPath(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestGenerate_TreeTagsAreScoredByTheirWholePath(t *testing.T) {
	answer := "Here are the tags:\n1. cooking / baking / sourdough\n2. cooking/baking\n3. Cooking / Baking\n"
	vecs := map[string][]float32{
		bread:                          {1, 0, 0},
		"cooking / baking / sourdough": {0.8, 0.6, 0},
		"cooking / baking":             {0.6, 0.8, 0},
	}
	llm := &stubLLM{answer: answer}
	emb := &vecEmbedder{vecs: vecs}
	g, err := NewGenerator(llm, emb, GeneratorConfig{}, WithTreeTags())
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.Generate(context.Background(), bread, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"cooking / baking / sourdough", 0.8}, {"cooking / baking", 0.6}})
	if docs := emb.calls[1].texts; !slices.Equal(docs, []string{"cooking / baking / sourdough", "cooking / baking"}) {
		t.Errorf("embedded %q", docs)
	}

	// the same reply without WithTreeTags: a three-level path has too many
	// words to be a flat tag, and the other two are flat tags as written
	vecs["cooking/baking"] = []float32{0.6, 0.8, 0}
	vecs["Cooking / Baking"] = []float32{0, 1, 0}
	g, _ = NewGenerator(&stubLLM{answer: answer}, &vecEmbedder{vecs: vecs}, GeneratorConfig{})
	got, err = g.Generate(context.Background(), bread, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{"cooking/baking", 0.6}, {"Cooking / Baking", 0}})
}

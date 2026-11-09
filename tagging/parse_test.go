package tagging

import (
	"context"
	"slices"
	"testing"

	"github.com/go-bumbu/sense/sensetest"
)

// TestParseTags_DropsTheScaffoldingOfAReply: a chat model wraps its tags in
// JSON, a label, a reasoning block, a code fence or Markdown as often as it
// writes them one per line. None of that is part of a tag.
func TestParseTags_DropsTheScaffoldingOfAReply(t *testing.T) {
	bread := []string{"sourdough", "baking", "bread"}
	for _, tc := range []struct {
		name   string
		answer string
		want   []string
	}{
		{"a JSON array", `["sourdough", "baking", "bread"]`, bread},
		{"a JSON array over several lines", "[\n  \"sourdough\",\n  \"baking\",\n  \"bread\"\n]", bread},
		{"a JSON array and a closing remark", "[\"sourdough\", \"baking\", \"bread\"]\nI hope these tags help!", bread},
		{"a JSON object of tags", `{"tags": ["sourdough", "baking", "bread"]}`, bread},
		{"a JSON object under another key", `{"Keywords": ["sourdough", "baking", "bread"]}`, bread},
		{"a JSON array in a code fence", "```json\n[\"sourdough\", \"baking\", \"bread\"]\n```", bread},
		{"a JSON array cut short", `["sourdough", "baking", "bread`, bread},
		{"a JSON object under a key no tag word names, alone", `{"items": ["sourdough", "baking", "bread"]}`, bread},
		{"a JSON object gives the member a tag word names only", `{"topic": "sourdough", "kind": "baking"}`, []string{"sourdough"}},
		{"a JSON array of objects gives their tags", `[{"tag": "sourdough"}, {"tag": "baking"}]`, []string{"sourdough", "baking"}},
		{"tags in JSON are cleaned too", `["1. Sourdough.", "one two three four five", "**baking**"]`, []string{"Sourdough", "baking"}},
		{"an inline label", "Tags: sourdough, baking, bread", bread},
		{"an inline label in bold", "**Tags:** sourdough, baking, bread", bread},
		{"an inline label before one tag", "Keywords: sourdough", []string{"sourdough"}},
		{"a sentence before a list", "Here are the tags: sourdough, baking, bread", bread},
		{"a colon inside a tag stays", "Star Trek: Voyager\nsourdough", []string{"Star Trek: Voyager", "sourdough"}},
		{"a reasoning block", "<think>\nThe text is about bread, flour and ovens.\nSo: baking.\n</think>\n\nsourdough\nbaking\nbread", bread},
		{"a reasoning block on one line", "<think>bread, ovens</think>sourdough, baking, bread", bread},
		{"a reasoning block never closed", "<think>\nThe text is about bread, so I would say sourdough", nil},
		{"a reasoning block never opened", "The text is about bread.\n</think>\nsourdough\nbaking\nbread", bread},
		{"a code fence", "```\nsourdough\nbaking\nbread\n```", bread},
		{"markdown emphasis and a remark in brackets", "- **sourdough** (bread)\n- **baking** (a technique, mostly)\n- *bread*", bread},
		{"emphasis inside a tag", "**sourdough** bread", []string{"sourdough bread"}},
		{"inline code", "`sourdough`, `baking`, `bread`", bread},
		{"brackets around a tag", "[sourdough]\n(baking)\n{bread}\n<bread>", bread},
		{"a markdown link", "[sourdough](https://example.com/sourdough)", []string{"sourdough"}},
		{"markup left open is no tag", "sour(dough\nbaking\nbr\"ead\n<b>bread</b>\nbread", []string{"baking", "bread"}},
		{"a remark in brackets on its own keeps its words", "sourdough (starter) bread", []string{"sourdough (starter) bread"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := parseTags(tc.answer, 10, cleanTag); !slices.Equal(got, tc.want) {
				t.Errorf("parseTags(%q) = %q, want %q", tc.answer, got, tc.want)
			}
		})
	}
}

// TestCleanPath_KeepsTheApplicationsSpellingOfItsOwnPaths: the model is asked
// to reuse the existing paths, written exactly as they are. When it does, the
// path is the application's own: it is returned in the application's spelling
// whatever the model's case or spacing, and its levels are not held to the
// limits of a tag the model makes up, so a long level no longer drops it.
func TestCleanPath_KeepsTheApplicationsSpellingOfItsOwnPaths(t *testing.T) {
	existing := newPathIndex([]string{
		"Recetas / Cocina tradicional de la abuela",
		"Cooking / Baking",
		"dev/go",
		" Travel / City Guides / Basel ",
		"cooking / grilling",
	})
	for in, want := range map[string]string{
		"Recetas / Cocina tradicional de la abuela":         "Recetas / Cocina tradicional de la abuela",
		"recetas / cocina tradicional de la abuela":         "Recetas / Cocina tradicional de la abuela",
		"1. **Recetas > Cocina  tradicional de la abuela**": "Recetas / Cocina tradicional de la abuela",
		"recetas / cocina tradicional de la abuela / pan":   "Recetas / Cocina tradicional de la abuela / pan",
		"recetas":                      "Recetas",
		"cooking / baking":             "Cooking / Baking",
		"COOKING/BAKING/Sourdough":     "Cooking / Baking / Sourdough",
		"Dev / Go":                     "dev/go",
		"dev / go / error handling":    "dev/go / error handling",
		"travel / city guides":         "Travel / City Guides",
		"travel / city guides / basel": "Travel / City Guides / Basel",
		"Cooking / Grilling":           "cooking / grilling",
		"cooking":                      "Cooking",
		"gardening / roses":            "gardening / roses",
		"gardening / this level is far too long a phrase":        "",
		"Cooking / Baking / this level is far too long a phrase": "",
		"cooking / baking / a / b / c":                           "Cooking / Baking / a / b",
	} {
		if got := cleanPath(in, existing); got != want {
			t.Errorf("cleanPath(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestGenerate_TreeTagsComeBackInTheExistingSpelling: through Generate, a
// reused path is scored and returned as the application wrote it.
func TestGenerate_TreeTagsComeBackInTheExistingSpelling(t *testing.T) {
	const long = "Recetas / Cocina tradicional de la abuela"
	vecs := map[string][]float32{bread: {1, 0, 0}, long: {0.8, 0.6, 0}, "Cooking / Baking": {0.6, 0.8, 0}}
	llm := &sensetest.FakeCompleter{Replies: []string{"recetas / cocina tradicional de la abuela\ncooking / baking\nCOOKING/BAKING"}}
	g, err := NewGenerator(llm, &vecEmbedder{vecs: vecs}, GeneratorConfig{}, WithTreeTags(long, "Cooking / Baking"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.Generate(context.Background(), bread, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireTags(t, got, []Tag{{long, 0.8}, {"Cooking / Baking", 0.6}})
}

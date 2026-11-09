package tagging

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai/langaitest"
)

// TestParseTags_SplitsKeepsAndDrops is the table of what the parser makes of
// the ways a model writes its tags: what is split into several tags, what is
// scaffolding and goes, and what is part of a tag and stays. tree, when not
// nil, is what the same reply gives with WithTreeTags; it is the same
// otherwise.
func TestParseTags_SplitsKeepsAndDrops(t *testing.T) {
	list := func(names ...string) []string { return names }
	same := []string(nil)
	for _, tc := range []struct {
		name   string
		answer string
		want   []string
		tree   []string
	}{
		// several tags on a line
		{"hashtags on one line", "#go #rust #devops", list("go", "rust", "devops"), same},
		{"two hashtags", "#golang #rust", list("golang", "rust"), same},
		{"hashtags and commas", "#go #rust, #devops", list("go", "rust", "devops"), same},
		{"a hash inside a line of words is no list", "learning #go the hard", list("learning #go the hard"), same},
		{"a markdown heading is not split", "# Tags\nsourdough", list("Tags", "sourdough"), same},
		{"ideographic commas", "烘焙、酸面团、面包", list("烘焙", "酸面团", "面包"), same},
		{"full-width commas", "烘焙，酸面团，面包", list("烘焙", "酸面团", "面包"), same},
		{"a full-width semicolon", "烘焙；酸面团", list("烘焙", "酸面团"), same},
		{"an Arabic comma", "خبز، عجين مخمر", list("خبز", "عجين مخمر"), same},
		{"an Arabic semicolon", "خبز؛ عجين مخمر", list("خبز", "عجين مخمر"), same},

		// labels and headings
		{"a label with a full-width colon", "标签：烘焙", list("烘焙"), same},
		{"a Japanese label and list", "タグ：寿司、天ぷら", list("寿司", "天ぷら"), same},
		{"a heading with a full-width colon", "标签：\n烘焙\n面包", list("烘焙", "面包"), same},
		{"a label is cut", "Tags: sourdough, baking", list("sourdough", "baking"), same},
		{"a label led in by a few words is cut", "Here are the tags: sourdough, baking", list("sourdough", "baking"), same},
		{"a label with a count is cut", "Top 3 tags: sourdough, baking\nTags (2): bread, butter", list("sourdough", "baking", "bread", "butter"), same},
		{"labels in Spanish and German are cut", "Las etiquetas son: pan, masa\nHier sind die Schlagwörter: Brot", list("pan", "masa", "Brot"), same},
		{"an answer word labels a line", "Answer: sourdough\nThe answer is: baking", list("sourdough", "baking"), same},
		{"a tag that starts with a label word is whole", "Topic modeling: LDA", list("Topic modeling: LDA"), same},
		{"another one", "Keyword research: SEO", list("Keyword research: SEO"), same},
		{"a brand that starts with one", "tag heuer: watches", list("tag heuer: watches"), same},
		{"a phrase that ends in one", "Record label: Motown", list("Record label: Motown"), same},
		{"a colon inside a tag", "Star Trek: Voyager", list("Star Trek: Voyager"), same},

		// JSON
		{"pretty-printed objects with scores", "{\n  \"tags\": [\n    {\"name\": \"go\", \"score\": 0.9},\n    {\"name\": \"rust\", \"score\": 0.8}\n  ]\n}", list("go", "rust"), same},
		{"an object with an unrelated string member", `{"keywords": ["a1", "b1"], "language": "en"}`, list("a1", "b1"), same},
		{"an object with an unrelated number member", `{"tags": ["go", "rust"], "confidence": 0.9}`, list("go", "rust"), same},
		{"an array with numbers, bools and null", `[2024, "go", true, null]`, list("go"), same},
		{"an array in an array", `["go", ["rust"]]`, list("go", "rust"), same},
		{"a list in a string", `{"tags": "go, rust"}`, list("go", "rust"), same},
		{"objects of tag and label", `[{"tag": "go"}, {"label": "rust"}, {"id": "x7"}]`, list("go", "rust"), same},
		{"an object wrapped in another", `{"result": {"tags": ["go"]}}`, list("go"), same},
		{"an object of scores holds no tags", `{"go": 0.9, "rust": 0.8}`, nil, same},
		{"an object of something else holds no tags", `{"language": "en", "summary": "about bread"}`, nil, same},
		{"JSON without tags is not read by lines", "{\"language\": \"en\"}\nbread", nil, same},
		{"a key and a number on a line", `"go": 0.9`, nil, same},
		{"keys with numbers, bools and null by lines", "\"name\": \"go\",\n\"score\": 0.9,\n\"ok\": true,\n\"note\": null,\n\"year\": \"2024\"", list("go", "2024"), same},

		// no tags, said in words
		{"no tags", "No tags.", nil, same},
		{"none", "None", nil, same},
		{"none in brackets", "(none)", nil, same},
		{"not applicable", "N/A", nil, same},
		{"no tags in Spanish", "Sin etiquetas.", nil, same},
		{"no tags in German", "Keine Tags", nil, same},
		{"a tag about nothing is a tag", "no tags\nbaking", list("no tags", "baking"), same},
		{"a longer sentence is not guessed at", "No tags apply", list("No tags apply"), same},

		// marks that are part of a tag
		{"a bracket that closes inside the tag", "(re)insurance", list("(re)insurance"), same},
		{"an inch mark", `5" floppy`, list(`5" floppy`), same},
		{"another inch mark", `12" vinyl`, list(`12" vinyl`), same},
		{"a less-than sign", "x < y", list("x < y"), same},
		{"an arrow", "a -> b", list("a -> b"), list("a / b")},
		{"an arrow in one rune", "a → b", list("a → b"), list("a / b")},
		{"angle brackets that pair up", "vector<int>", list("vector<int>"), list("vector<int")},
		{"brackets inside a tag", "state [of] the art", list("state [of] the art"), same},
		{"plus signs", "C++", list("C++"), same},
		{"a hash at the end", "C#", list("C#"), same},
		{"a dot at the start", ".NET", list(".NET"), same},
		{"an ampersand", "R&D\nQ&A", list("R&D", "Q&A"), same},
		{"a percent sign", "50% off", list("50% off"), same},
		{"a number in quotes", `"2024"`, list("2024"), same},
		{"an emoji", "🍞 baking", list("🍞 baking"), same},

		// marks around a tag
		{"brackets around a tag", "[sourdough]\n(baking)\n{bread}\n<butter>", list("sourdough", "baking", "bread", "butter"), same},
		{"a bracket never closed", "[sourdough", list("sourdough"), same},
		{"a bracket never opened", "sourdough]", list("sourdough"), same},
		{"a number in brackets before a tag", "(1) sourdough\n(2) baking", list("sourdough", "baking"), same},
		{"a checkbox before a tag", "- [ ] sourdough\n- [x] baking", list("sourdough", "baking"), same},
		{"a remark in brackets after a tag", "sourdough (bread)", list("sourdough"), same},

		// markup that does not close is no tag
		{"a bracket open inside", "sour(dough\nbread", list("bread"), same},
		{"a bracket closed inside", "re)insurance\nbread", list("bread"), same},
		{"a quote open inside", "br\"ead\nbread", list("bread"), same},
		{"an HTML element", "<b>bread</b>\nbutter", list("butter"), same},

		// paths
		{"a bracket that closes inside a level", "(re)insurance / life", list("(re)insurance / life"), same},
		{"hashtags of paths", "#dev/go #dev/rust", list("dev/go", "dev/rust"), list("dev / go", "dev / rust")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := parseTags(tc.answer, 10, cleanTag); !slices.Equal(got, tc.want) {
				t.Errorf("parseTags(%q) = %q, want %q", tc.answer, got, tc.want)
			}
			want := tc.tree
			if want == nil {
				want = tc.want
			}
			path := func(s string) string { return cleanPath(s, nil) }
			if got := parseTags(tc.answer, 10, path); !slices.Equal(got, want) {
				t.Errorf("as tree tags, parseTags(%q) = %q, want %q", tc.answer, got, want)
			}
		})
	}
}

// TestParseTags_ReadsALongReplyInLinearTime: a reply is the model's, as long
// and as odd as it likes. Brackets by the hundred thousand must not make the
// parser quadratic.
func TestParseTags_ReadsALongReplyInLinearTime(t *testing.T) {
	for name, answer := range map[string]string{
		"opening brackets":   strings.Repeat("(", 400_000),
		"closing brackets":   strings.Repeat("]", 400_000),
		"nested brackets":    strings.Repeat("[", 200_000) + "x" + strings.Repeat("]", 200_000),
		"pairs":              strings.Repeat("(a)", 150_000),
		"remarks":            "x" + strings.Repeat(" (y)", 100_000),
		"commas":             strings.Repeat("a、", 200_000),
		"hashtags":           strings.Repeat("#a ", 150_000),
		"deeply nested JSON": strings.Repeat("[", 5_000) + `"a"` + strings.Repeat("]", 5_000),
	} {
		t.Run(name, func(t *testing.T) {
			start := time.Now()
			parseTags(answer, 10, cleanTag)
			parseTags(answer, 10, func(s string) string { return cleanPath(s, nil) })
			if d := time.Since(start); d > 5*time.Second {
				t.Errorf("parsing %d bytes took %v", len(answer), d)
			}
		})
	}
}

// TestGenerate_AnAnswerOfNoTagsInWordsIsNoTags: a model that says in words
// that it has no tag for a text returns none, not its sentence as a tag.
func TestGenerate_AnAnswerOfNoTagsInWordsIsNoTags(t *testing.T) {
	for _, answer := range []string{"No tags.", "none", "N/A", `{"tags": []}`, `{"language": "en"}`} {
		llm := &langaitest.FakeCompleter{Replies: []string{answer}}
		g, err := NewGenerator(llm, langaitest.FakeEmbedder{}, GeneratorConfig{})
		if err != nil {
			t.Fatal(err)
		}
		if tags, err := g.Generate(context.Background(), "a text about nothing in particular", -1); err != nil || tags != nil {
			t.Errorf("answer %q: Generate = %+v, %v; want no tags and no error", answer, tags, err)
		}
	}
}

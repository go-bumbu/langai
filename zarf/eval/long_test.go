package eval

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadLong_ReadsAndFillsOffsets(t *testing.T) {
	docs, err := LoadLong("testdata/longfixture")
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 {
		t.Fatalf("got %d documents, want 1", len(docs))
	}
	doc := docs[0]
	sides, late := 0, 0
	for _, tp := range doc.Topics {
		switch tp.Kind {
		case TopicMain:
			if tp.Offset != 0 {
				t.Errorf("main topic %q offset = %d, want 0", tp.Label, tp.Offset)
			}
		case TopicSide:
			sides++
			if want := strings.Index(doc.Text, "## "+tp.Section); tp.Offset != want || want <= 0 {
				t.Errorf("side topic %q offset = %d, want %d", tp.Label, tp.Offset, want)
			}
			if tp.Late() {
				late++
			}
		}
	}
	if sides < 3 || late == 0 {
		t.Errorf("fixture has %d sides, %d late", sides, late)
	}
}

func TestLoadLong_RejectsUnknownKeysAndSortsByID(t *testing.T) {
	raw, err := os.ReadFile("testdata/longfixture/one.yaml")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("a.yaml", strings.Replace(string(raw), "id: fixture-one", "id: zzz", 1))
	write("b.yaml", strings.Replace(string(raw), "id: fixture-one", "id: aaa", 1))
	docs, err := LoadLong(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 || docs[0].ID != "aaa" || docs[1].ID != "zzz" {
		t.Fatalf("not sorted by id: %+v", docs)
	}
	write("c.yaml", "id: ccc\nbogus: 1\n")
	if _, err := LoadLong(dir); err == nil {
		t.Error("unknown key accepted")
	}
}

func TestLoadLong_DuplicateIDAcrossFiles(t *testing.T) {
	raw, err := os.ReadFile("testdata/longfixture/one.yaml")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	for _, n := range []string{"a.yaml", "b.yaml"} {
		if err := os.WriteFile(filepath.Join(dir, n), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	_, err = LoadLong(dir)
	if err == nil || !strings.Contains(err.Error(), "fixture-one") {
		t.Errorf("err = %v, want duplicate naming fixture-one", err)
	}
}

// validLong builds a document that passes every rule; cases break one thing.
func validLong() LongDocument {
	pad := strings.Repeat("Filler sentence that pads the text out. ", 30) + "\n\n"
	text := "Intro.\n\n" + pad
	for _, s := range []string{"Early", "Middle", "Late", "Closing", "MentionA", "MentionB"} {
		text += "## " + s + "\n\n" + pad
	}
	return LongDocument{
		ID: "doc", Lang: "en", Title: "Title", Text: text,
		Topics: []Topic{
			{Label: "Main", Kind: TopicMain, Aliases: []string{"main"}},
			{Label: "E", Kind: TopicSide, Section: "Early", Aliases: []string{"e"}},
			{Label: "M", Kind: TopicSide, Section: "Middle", Aliases: []string{"m"}},
			{Label: "L", Kind: TopicSide, Section: "Late", Aliases: []string{"l"}},
			{Label: "C", Kind: TopicSide, Section: "Closing", Aliases: []string{"c"}},
			{Label: "A", Kind: TopicMention, Section: "MentionA", Aliases: []string{"a"}},
			{Label: "B", Kind: TopicMention, Section: "MentionB", Aliases: []string{"b"}},
		},
	}
}

func TestLongDocument_Validate(t *testing.T) {
	if err := func() error { d := validLong(); return d.validate(map[string]bool{}) }(); err != nil {
		t.Fatalf("base document invalid: %v", err)
	}
	cases := []struct {
		name   string
		mutate func(d *LongDocument)
		seen   map[string]bool
		id     string // id expected in the error; empty means d.ID
		want   string
	}{
		{"empty id", func(d *LongDocument) { d.ID = "" }, nil, "", "empty id"},
		{"duplicate id", func(d *LongDocument) {}, map[string]bool{"doc": true}, "", "duplicate"},
		{"bad lang", func(d *LongDocument) { d.Lang = "fr" }, nil, "", "lang"},
		{"blank title", func(d *LongDocument) { d.Title = "  " }, nil, "", "title"},
		{"text too short", func(d *LongDocument) { d.Text = "## Early\n## Middle\n## Late\n## Closing\n## MentionA\n## MentionB\n" }, nil, "", "text"},
		{"text too long", func(d *LongDocument) { d.Text += strings.Repeat("x", 65536) }, nil, "", "text"},
		{"no main", func(d *LongDocument) { d.Topics = d.Topics[1:] }, nil, "", "main"},
		{"too few sides", func(d *LongDocument) { d.Topics = append(d.Topics[:3:3], d.Topics[5:]...) }, nil, "", "side"},
		{"too few mentions", func(d *LongDocument) { d.Topics = d.Topics[:6] }, nil, "", "mention"},
		{"blank label", func(d *LongDocument) { d.Topics[1].Label = " " }, nil, "", "label"},
		{"no aliases", func(d *LongDocument) { d.Topics[1].Aliases = nil }, nil, "", "alias"},
		{"blank alias", func(d *LongDocument) { d.Topics[1].Aliases = []string{"ok", " "} }, nil, "", "alias"},
		{"bad kind", func(d *LongDocument) { d.Topics[1].Kind = "other" }, nil, "", "kind"},
		{"side section missing", func(d *LongDocument) { d.Topics[1].Section = "Nowhere" }, nil, "", "Nowhere"},
		{"mention section missing", func(d *LongDocument) { d.Topics[5].Section = "Nowhere" }, nil, "", "Nowhere"},
		{"main with section", func(d *LongDocument) { d.Topics[0].Section = "Early" }, nil, "", "main"},
		{"no late side", func(d *LongDocument) {
			for _, i := range []int{2, 3, 4} {
				d.Topics[i].Section = "Early"
			}
		}, nil, "", "late"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := validLong()
			tc.mutate(&d)
			seen := tc.seen
			if seen == nil {
				seen = map[string]bool{}
			}
			err := d.validate(seen)
			if err == nil {
				t.Fatal("expected an error")
			}
			msg := err.Error()
			if !strings.HasPrefix(msg, "eval: long ") {
				t.Errorf("error %q lacks the eval: long prefix", msg)
			}
			if d.ID != "" && !strings.Contains(msg, d.ID) {
				t.Errorf("error %q does not name the doc id %q", msg, d.ID)
			}
			if !strings.Contains(msg, tc.want) {
				t.Errorf("error %q does not mention %q", msg, tc.want)
			}
		})
	}
}

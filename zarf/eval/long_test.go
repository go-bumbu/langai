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
	// Otherwise valid, so only KnownFields can reject it.
	write("c.yaml", strings.Replace(string(raw), "id: fixture-one", "id: ccc\nbogus: 1", 1))
	if _, err := LoadLong(dir); err == nil || !strings.Contains(err.Error(), "bogus") {
		t.Errorf("err = %v, want unknown key bogus rejected", err)
	}
}

func TestLoadLong_NoDocumentsIsAnError(t *testing.T) {
	for _, dir := range []string{t.TempDir(), filepath.Join(t.TempDir(), "missing")} {
		if docs, err := LoadLong(dir); err == nil || !strings.Contains(err.Error(), dir) {
			t.Errorf("LoadLong(%s) = %d documents, %v; want an error naming the directory", dir, len(docs), err)
		}
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

// TestLongGoldenComposition checks the shape of testdata/long: 24 documents,
// 8 per language, each language with 3 texts of 6-10 KB, 3 of 14-20 KB and
// 2 of 28-36 KB, and every id written as <lang>-<slug>.
func TestLongGoldenComposition(t *testing.T) {
	docs, err := LoadLong("testdata/long")
	if err != nil {
		t.Fatalf("LoadLong(testdata/long): %v", err)
	}
	if len(docs) != 24 {
		t.Errorf("got %d documents, want 24 (8 per language)", len(docs))
	}

	byLang := map[string][]LongDocument{}
	for _, d := range docs {
		byLang[d.Lang] = append(byLang[d.Lang], d)
		if !strings.HasPrefix(d.ID, d.Lang+"-") || len(d.ID) == len(d.Lang)+1 {
			t.Errorf("doc %q: id must be <lang>-<slug> with the lang prefix %q", d.ID, d.Lang)
		}
	}

	tiers := []struct {
		name     string
		min, max int // bytes of Text, inclusive
		want     int
	}{
		{"6-10 KB", 6000, 10000, 3},
		{"14-20 KB", 14000, 20000, 3},
		{"28-36 KB", 28000, 36000, 2},
	}
	for _, lang := range []string{"en", "es", "de"} {
		ds := byLang[lang]
		if len(ds) != 8 {
			t.Errorf("lang %s: %d documents, want 8 (short by %d)", lang, len(ds), 8-len(ds))
		}
		for _, tr := range tiers {
			n := 0
			for _, d := range ds {
				if l := len(d.Text); l >= tr.min && l <= tr.max {
					n++
				}
			}
			switch {
			case n < tr.want:
				t.Errorf("lang %s: %d documents of %s bytes %d-%d, want %d (short by %d)",
					lang, n, tr.name, tr.min, tr.max, tr.want, tr.want-n)
			case n > tr.want:
				t.Errorf("lang %s: %d documents of %s bytes %d-%d, want %d (over by %d)",
					lang, n, tr.name, tr.min, tr.max, tr.want, n-tr.want)
			}
		}
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
		{"two topics of one label", func(d *LongDocument) { d.Topics[2].Label = "e" }, nil, "", `share the name "e"`},
		{"an alias of two topics", func(d *LongDocument) {
			d.Topics[5].Aliases = []string{"x", "The-Main"}
			d.Topics[0].Aliases = []string{"the main"}
		}, nil, "", `share the name "The-Main"`},
		{"an alias that is another topic's label", func(d *LongDocument) { d.Topics[3].Aliases = []string{"main"} }, nil, "", `share the name "main"`},
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

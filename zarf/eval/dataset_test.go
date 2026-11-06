package eval

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

// validDataset is the smallest dataset that passes Validate; the mutation
// cases below each break one rule of it.
func validDataset() Dataset {
	return Dataset{
		Tags: []string{"dev", "dev / go"},
		Documents: []Document{
			{ID: "go-doc", Lang: "en", Title: "Go modules", Excerpt: "Dependencies.", Site: "go.dev", Text: "go modules pin versions", Tags: []string{"dev / go"}},
			{ID: "dev-doc", Lang: "es", Title: "Revisar código", Text: "revisiones de código", Tags: []string{"dev"}},
		},
		Queries: []Query{
			{ID: "q-go", Lang: "en", Text: "go modules", Relevant: []string{"go-doc"}},
		},
	}
}

func TestValidate_AcceptsValidDataset(t *testing.T) {
	ds := validDataset()
	if err := ds.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

func TestValidate_RejectsBrokenReferences(t *testing.T) {
	cases := []struct {
		name   string
		break_ func(*Dataset)
		want   string
	}{
		{"empty tag", func(d *Dataset) { d.Tags = append(d.Tags, " ") }, "tags[2]"},
		{"duplicate tag", func(d *Dataset) { d.Tags = append(d.Tags, "dev") }, "duplicate tag path"},
		{"document with unknown tag spelling", func(d *Dataset) { d.Documents[0].Tags = []string{"dev/go"} }, `documents[0] "go-doc": tag "dev/go" not in tags`},
		{"document without tags", func(d *Dataset) { d.Documents[0].Tags = nil }, "no expected tags"},
		{"document without id", func(d *Dataset) { d.Documents[1].ID = "" }, "documents[1]: empty id"},
		{"duplicate document id", func(d *Dataset) { d.Documents[1].ID = "go-doc" }, "duplicate id"},
		{"document without lang", func(d *Dataset) { d.Documents[0].Lang = "" }, "empty lang"},
		{"document without title", func(d *Dataset) { d.Documents[0].Title = " " }, "empty title"},
		{"document without text", func(d *Dataset) { d.Documents[0].Text = "" }, "empty text"},
		{"document text over the cap", func(d *Dataset) { d.Documents[0].Text = strings.Repeat("x", maxTextBytes+1) }, "2001 bytes"},
		{"query without id", func(d *Dataset) { d.Queries[0].ID = "" }, "queries[0]: empty id"},
		{"query id clashing with a document", func(d *Dataset) { d.Queries[0].ID = "go-doc" }, "duplicate id"},
		{"query without lang", func(d *Dataset) { d.Queries[0].Lang = "" }, "empty lang"},
		{"query without text", func(d *Dataset) { d.Queries[0].Text = "" }, "empty text"},
		{"query without relevant", func(d *Dataset) { d.Queries[0].Relevant = nil }, "no relevant documents"},
		{"query with unknown relevant id", func(d *Dataset) { d.Queries[0].Relevant = []string{"nope"} }, `relevant "nope" is not a document id`},
		{"query relevant to another query", func(d *Dataset) {
			d.Queries = append(d.Queries, Query{ID: "q2", Lang: "en", Text: "x", Relevant: []string{"q-go"}})
		}, `relevant "q-go" is not a document id`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ds := validDataset()
			c.break_(&ds)
			err := ds.Validate()
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("Validate = %v, want an error containing %q", err, c.want)
			}
		})
	}
}

func TestParse_RejectsUnknownKeys(t *testing.T) {
	const doc = "tags: [dev]\ndocuments:\n  - id: a\n    lang: en\n    title: T\n    text: body\n    tagz: [dev]\nqueries: []\n"
	_, err := Parse(strings.NewReader(doc))
	if err == nil || !strings.Contains(err.Error(), "tagz") {
		t.Fatalf("Parse = %v, want an error naming the unknown key", err)
	}
}

func TestLoad_ReadsAndValidates(t *testing.T) {
	const doc = `tags:
  - dev
documents:
  - id: a
    lang: en
    title: A title
    excerpt: An excerpt.
    site: example.org
    text: |
      First paragraph.

      Second paragraph.
    tags: [dev]
queries:
  - id: q
    lang: de
    text: eine frage
    relevant: [a]
`
	path := filepath.Join(t.TempDir(), "ds.yaml")
	if err := os.WriteFile(path, []byte(doc), 0o600); err != nil {
		t.Fatal(err)
	}
	ds, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(ds.Documents) != 1 || ds.Documents[0].Text != "First paragraph.\n\nSecond paragraph.\n" {
		t.Fatalf("documents = %+v", ds.Documents)
	}
	if len(ds.Queries) != 1 || ds.Queries[0].Relevant[0] != "a" {
		t.Fatalf("queries = %+v", ds.Queries)
	}
	if _, err := Load(filepath.Join(t.TempDir(), "missing.yaml")); err == nil {
		t.Fatal("Load of a missing file must fail")
	}
}

// TestSplitQueries pins the cross-lingual rule: a query crosses a language
// when a document it should find is in another language, whatever the query's
// own language. Of the three Spanish queries only the one for the English
// document and the one relevant to both documents cross a language; a rule
// keyed on the query's language alone ("anything not English") would move all
// three. Each slice keeps the dataset's order.
func TestSplitQueries(t *testing.T) {
	ds := validDataset() // go-doc is English, dev-doc is Spanish
	ds.Queries = []Query{
		{ID: "q-en-en", Lang: "en", Text: "go modules", Relevant: []string{"go-doc"}},
		{ID: "q-es-en", Lang: "es", Text: "módulos de go", Relevant: []string{"go-doc"}},
		{ID: "q-es-es", Lang: "es", Text: "revisiones", Relevant: []string{"dev-doc"}},
		{ID: "q-es-both", Lang: "es", Text: "programación", Relevant: []string{"dev-doc", "go-doc"}},
	}
	if err := ds.Validate(); err != nil {
		t.Fatal(err)
	}
	same, cross := ds.SplitQueries()
	if got := queryIDs(same); got != "q-en-en,q-es-es" {
		t.Errorf("same-language = %s, want q-en-en,q-es-es", got)
	}
	if got := queryIDs(cross); got != "q-es-en,q-es-both" {
		t.Errorf("cross-lingual = %s, want q-es-en,q-es-both", got)
	}
}

func queryIDs(qs []Query) string {
	ids := make([]string, len(qs))
	for i, q := range qs {
		ids[i] = q.ID
	}
	return strings.Join(ids, ",")
}

// TestEssence_JoinsTitleExcerptSiteAndText pins the join rule callers build
// their pages' essences by: title, excerpt, site, text on their own lines,
// trimmed, so the tagging task embeds what an application hands its tagger.
func TestEssence_JoinsTitleExcerptSiteAndText(t *testing.T) {
	doc := Document{Title: "Title", Excerpt: "Excerpt", Site: "site.org", Text: "Body text.\n"}
	if got := doc.Essence(); got != "Title\nExcerpt\nsite.org\nBody text." {
		t.Fatalf("Essence = %q", got)
	}
	bare := Document{Title: "Title", Text: "Body"}
	if got := bare.Essence(); got != "Title\n\n\nBody" {
		t.Fatalf("Essence without excerpt/site = %q, want the empty lines kept", got)
	}
}

// TestEssence_CutsTextWithoutSplittingARune covers a text past maxTextBytes,
// which Validate keeps out of a dataset but a page built by a test (two
// documents stitched into one) can reach. The cut is a byte offset, so it can
// land inside a multi-byte rune; what comes out must still be valid UTF-8.
func TestEssence_CutsTextWithoutSplittingARune(t *testing.T) {
	head := "Title\nExcerpt\nSite\n"
	for _, tc := range []struct {
		name string
		text string
	}{
		// the leading "x" shifts the 2-byte runes so the cut lands mid-rune
		{"2-byte runes", "x" + strings.Repeat("é", maxTextBytes)},
		{"3-byte runes", strings.Repeat("€", maxTextBytes)},
		{"3-byte runes offset", "x" + strings.Repeat("€", maxTextBytes)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Document{Title: "Title", Excerpt: "Excerpt", Site: "Site", Text: tc.text}.Essence()
			if !utf8.ValidString(got) {
				t.Errorf("essence is not valid UTF-8: %q", got[len(got)-8:])
			}
			if len(got) > len(head)+maxTextBytes || len(got) < len(head)+maxTextBytes-3 {
				t.Errorf("essence is %d bytes, want the header and up to %d bytes of text, at most one rune short", len(got), maxTextBytes)
			}
			if !strings.HasPrefix(got, head) {
				t.Errorf("essence lost its title/excerpt/site header: %q", got[:len(head)])
			}
		})
	}
}

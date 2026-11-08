package eval

import (
	"testing"
)

// goldenPath is the checked-in dataset, relative to this package.
const goldenPath = "testdata/golden.yaml"

// TestGoldenSet pins the composition the design asks for: three languages
// with enough documents and same-language queries each that "does search
// work in German" rests on something, every tag with at least two documents,
// prose-length texts, and enough cross-lingual queries for that number to
// mean something too. Load already enforces the structural rules.
func TestGoldenSet(t *testing.T) {
	ds, err := Load(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(ds.Documents) < 80 || len(ds.Queries) < 60 || len(ds.Tags) < 12 {
		t.Fatalf("golden set has %d documents, %d queries, %d tags; want >= 80, 60, 12", len(ds.Documents), len(ds.Queries), len(ds.Tags))
	}
	docsPerLang, docsPerTag := map[string]int{}, map[string]int{}
	for _, doc := range ds.Documents {
		docsPerLang[doc.Lang]++
		for _, tag := range doc.Tags {
			docsPerTag[tag]++
		}
		if len(doc.Text) < 300 {
			t.Errorf("document %q: text is %d bytes, want prose of at least 300", doc.ID, len(doc.Text))
		}
	}
	same, cross := ds.SplitQueries()
	samePerLang := map[string]int{}
	for _, q := range same {
		samePerLang[q.Lang]++
	}
	for _, l := range []string{"en", "es", "de"} {
		if docsPerLang[l] < 20 || samePerLang[l] < 12 {
			t.Errorf("language %s: %d documents, %d same-language queries; want >= 20 and >= 12", l, docsPerLang[l], samePerLang[l])
		}
	}
	if len(docsPerLang) != 3 {
		t.Errorf("document languages %v; want exactly en, es, de", docsPerLang)
	}
	for _, q := range ds.Queries {
		if docsPerLang[q.Lang] == 0 {
			t.Errorf("query %q is in %s, a language without documents", q.ID, q.Lang)
		}
	}
	for _, tag := range ds.Tags {
		if docsPerTag[tag] < 2 {
			t.Errorf("tag %q has %d documents, want >= 2", tag, docsPerTag[tag])
		}
	}
	if len(cross) < 8 {
		t.Errorf("%d cross-lingual queries, want >= 8", len(cross))
	}
}

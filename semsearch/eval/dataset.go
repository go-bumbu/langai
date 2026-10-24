package eval

import (
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"

	"go.yaml.in/yaml/v3"
)

// Dataset is one labelled corpus: a tag taxonomy, documents with their
// expected tags, and queries with their relevant documents.
type Dataset struct {
	Tags      []string   `yaml:"tags"`
	Documents []Document `yaml:"documents"`
	Queries   []Query    `yaml:"queries"`
}

// Document is one retrieval-corpus entry and one tagging input. Its four
// text fields are a web page's, joined into the essence a tagger is handed
// (see Essence); Tags are its expected labels, as full tag paths.
type Document struct {
	ID      string   `yaml:"id"`
	Lang    string   `yaml:"lang"`
	Title   string   `yaml:"title"`
	Excerpt string   `yaml:"excerpt"`
	Site    string   `yaml:"site"`
	Text    string   `yaml:"text"`
	Tags    []string `yaml:"tags"`
}

// Query is one free-text search with the ids of the documents it should find.
type Query struct {
	ID       string   `yaml:"id"`
	Lang     string   `yaml:"lang"`
	Text     string   `yaml:"text"`
	Relevant []string `yaml:"relevant"`
}

// maxTextBytes is how much of a document's body text its essence keeps (see
// Essence). Validate refuses longer texts, so every dataset document is
// measured on the whole text it holds.
const maxTextBytes = 2000

// Load reads and validates a dataset file.
func Load(path string) (*Dataset, error) {
	f, err := os.Open(path) //nolint:gosec // path is a test fixture or an operator's flag, read once
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return Parse(f)
}

// Parse decodes and validates a dataset. Unknown keys are an error, so a
// misspelt field cannot silently drop labels.
func Parse(r io.Reader) (*Dataset, error) {
	dec := yaml.NewDecoder(r)
	dec.KnownFields(true)
	var ds Dataset
	if err := dec.Decode(&ds); err != nil {
		return nil, fmt.Errorf("eval: parse dataset: %w", err)
	}
	if err := ds.Validate(); err != nil {
		return nil, err
	}
	return &ds, nil
}

// Validate checks the structural rules every dataset must meet: unique,
// non-empty ids across documents and queries; non-empty language, title and
// text; text within what an essence keeps; expected tags and relevant ids
// that exist. Errors name the offending entry and field.
func (d *Dataset) Validate() error {
	tags := make(map[string]bool, len(d.Tags))
	for i, t := range d.Tags {
		if strings.TrimSpace(t) == "" {
			return fmt.Errorf("eval: tags[%d]: empty tag path", i)
		}
		if tags[t] {
			return fmt.Errorf("eval: tags[%d]: duplicate tag path %q", i, t)
		}
		tags[t] = true
	}
	ids := make(map[string]bool, len(d.Documents)+len(d.Queries))
	docIDs := make(map[string]bool, len(d.Documents))
	for i, doc := range d.Documents {
		if err := doc.validate(i, ids, tags); err != nil {
			return err
		}
		ids[doc.ID], docIDs[doc.ID] = true, true
	}
	for i, q := range d.Queries {
		if err := q.validate(i, ids, docIDs); err != nil {
			return err
		}
		ids[q.ID] = true
	}
	return nil
}

func (doc Document) validate(i int, ids, tags map[string]bool) error {
	where := fmt.Sprintf("eval: documents[%d] %q", i, doc.ID)
	switch {
	case doc.ID == "":
		return fmt.Errorf("eval: documents[%d]: empty id", i)
	case ids[doc.ID]:
		return fmt.Errorf("%s: duplicate id", where)
	case doc.Lang == "":
		return fmt.Errorf("%s: empty lang", where)
	case strings.TrimSpace(doc.Title) == "":
		return fmt.Errorf("%s: empty title", where)
	case strings.TrimSpace(doc.Text) == "":
		return fmt.Errorf("%s: empty text", where)
	case len(doc.Text) > maxTextBytes:
		return fmt.Errorf("%s: text is %d bytes, an essence keeps at most %d", where, len(doc.Text), maxTextBytes)
	case len(doc.Tags) == 0:
		return fmt.Errorf("%s: no expected tags", where)
	}
	for _, t := range doc.Tags {
		if !tags[t] {
			return fmt.Errorf("%s: tag %q not in tags", where, t)
		}
	}
	return nil
}

func (q Query) validate(i int, ids, docIDs map[string]bool) error {
	where := fmt.Sprintf("eval: queries[%d] %q", i, q.ID)
	switch {
	case q.ID == "":
		return fmt.Errorf("eval: queries[%d]: empty id", i)
	case ids[q.ID]:
		return fmt.Errorf("%s: duplicate id", where)
	case q.Lang == "":
		return fmt.Errorf("%s: empty lang", where)
	case strings.TrimSpace(q.Text) == "":
		return fmt.Errorf("%s: empty text", where)
	case len(q.Relevant) == 0:
		return fmt.Errorf("%s: no relevant documents", where)
	}
	for _, id := range q.Relevant {
		if !docIDs[id] {
			return fmt.Errorf("%s: relevant %q is not a document id", where, id)
		}
	}
	return nil
}

// Essence is the text the tagging task embeds for a document: its title,
// excerpt and site (each a strong, short signal) and the first maxTextBytes
// bytes of its body text, on their own lines and trimmed. The golden
// numbers measure tagging over essences built this way, so a caller that
// hands an autotag.Tagger its pages' essences built by the same rule gets
// what the numbers say.
func (doc Document) Essence() string {
	return strings.TrimSpace(doc.Title + "\n" + doc.Excerpt + "\n" + doc.Site + "\n" + cutText(doc.Text))
}

// cutText keeps the first maxTextBytes bytes of text. The cut is a byte
// offset, so it can land inside a multi-byte rune: that partial rune is
// dropped too, so the essence handed to a tokenizer stays valid UTF-8.
func cutText(text string) string {
	if len(text) <= maxTextBytes {
		return text
	}
	text = text[:maxTextBytes]
	for len(text) > 0 {
		if r, size := utf8.DecodeLastRuneInString(text); r != utf8.RuneError || size > 1 {
			break
		}
		text = text[:len(text)-1]
	}
	return text
}

// SplitQueries partitions the queries by whether they must cross a language.
// A query is cross-lingual when a document it should find is in another
// language than the query itself; every other query is same-language. The
// query's own language never decides alone: a Spanish query for a Spanish
// document is same-language, a Spanish query for an English one is not. Both
// slices keep the dataset's order.
func (d *Dataset) SplitQueries() (same, cross []Query) {
	lang := make(map[string]string, len(d.Documents))
	for _, doc := range d.Documents {
		lang[doc.ID] = doc.Lang
	}
	for _, q := range d.Queries {
		if q.crossLingual(lang) {
			cross = append(cross, q)
		} else {
			same = append(same, q)
		}
	}
	return same, cross
}

// crossLingual reports whether one of q's relevant documents is in another
// language than q, given each document's language by id.
func (q Query) crossLingual(lang map[string]string) bool {
	for _, id := range q.Relevant {
		if lang[id] != q.Lang {
			return true
		}
	}
	return false
}

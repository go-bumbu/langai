package eval

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"go.yaml.in/yaml/v3"
)

// Topic kinds of a long document.
const (
	// TopicMain is what the whole document is about.
	TopicMain = "main"
	// TopicSide is a topic one section of the document is about.
	TopicSide = "side"
	// TopicMention is a topic only named in passing in one section.
	TopicMention = "mention"
)

// LateOffset is the byte offset past which a topic counts as late: beyond
// what a tagger that reads only the first 2 KB of a text would ever see.
const LateOffset = 2048

const (
	minLongBytes   = 6000
	maxLongBytes   = 65536
	minSideTopics  = 3
	minMentionTops = 2
)

// LongDocument is one entry of the long golden set: a long text and the
// topics a tagger should find in it, by where in the text they live.
type LongDocument struct {
	ID     string  `yaml:"id"`
	Lang   string  `yaml:"lang"`
	Title  string  `yaml:"title"`
	Text   string  `yaml:"text"`
	Topics []Topic `yaml:"topics"`
}

// Topic is one expected subject of a long document. Aliases are the strings
// that count as finding it; Section names the "## " heading of the part of
// the text it lives in (empty for the main topic).
type Topic struct {
	Label   string   `yaml:"label"`
	Kind    string   `yaml:"kind"`
	Section string   `yaml:"section"`
	Aliases []string `yaml:"aliases"`
	// Offset is the byte offset of the section heading in the document's
	// text, filled on load; 0 for main topics.
	Offset int `yaml:"-"`
}

// Late reports whether the topic lives past LateOffset.
func (t Topic) Late() bool { return t.Offset > LateOffset }

// LoadLong reads every *.yaml file in dir, one document per file, validates
// them and returns them sorted by id. Unknown keys are an error.
func LoadLong(dir string) ([]LongDocument, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.yaml"))
	if err != nil {
		return nil, err
	}
	var docs []LongDocument
	seen := map[string]bool{}
	for _, p := range paths {
		doc, err := loadLongFile(p)
		if err != nil {
			return nil, err
		}
		if err := doc.validate(seen); err != nil {
			return nil, err
		}
		seen[doc.ID] = true
		docs = append(docs, doc)
	}
	sort.Slice(docs, func(i, j int) bool { return docs[i].ID < docs[j].ID })
	return docs, nil
}

func loadLongFile(path string) (LongDocument, error) {
	f, err := os.Open(path) //nolint:gosec // path comes from a directory the operator named
	if err != nil {
		return LongDocument{}, err
	}
	defer func() { _ = f.Close() }()
	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)
	var doc LongDocument
	if err := dec.Decode(&doc); err != nil {
		return LongDocument{}, fmt.Errorf("eval: parse long document %s: %w", path, err)
	}
	return doc, nil
}

// validate checks the structural rules of a long document and fills each
// topic's Offset. seen holds the ids already loaded.
func (d *LongDocument) validate(seen map[string]bool) error {
	if d.ID == "" {
		return fmt.Errorf("eval: long %q: empty id", d.Title)
	}
	fail := func(format string, a ...any) error {
		return fmt.Errorf("eval: long %s: %s", d.ID, fmt.Sprintf(format, a...))
	}
	switch {
	case seen[d.ID]:
		return fail("duplicate id")
	case d.Lang != "en" && d.Lang != "es" && d.Lang != "de":
		return fail("lang %q is not en, es or de", d.Lang)
	case strings.TrimSpace(d.Title) == "":
		return fail("empty title")
	case len(d.Text) < minLongBytes || len(d.Text) > maxLongBytes:
		return fail("text is %d bytes, want %d to %d", len(d.Text), minLongBytes, maxLongBytes)
	}
	var c topicCounts
	for i := range d.Topics {
		if err := d.Topics[i].validate(d, i, &c); err != nil {
			return err
		}
	}
	switch {
	case c.main == 0:
		return fail("no main topic")
	case c.side < minSideTopics:
		return fail("%d side topics, want at least %d", c.side, minSideTopics)
	case c.mention < minMentionTops:
		return fail("%d mention topics, want at least %d", c.mention, minMentionTops)
	case c.late == 0:
		return fail("no side topic is late (past byte %d)", LateOffset)
	}
	return nil
}

// topicCounts tallies the topics of a document by role.
type topicCounts struct{ main, side, mention, late int }

// validate checks topic i of d, fills its Offset and counts it in c.
func (t *Topic) validate(d *LongDocument, i int, c *topicCounts) error {
	fail := func(format string, a ...any) error {
		return fmt.Errorf("eval: long %s: %s", d.ID, fmt.Sprintf(format, a...))
	}
	if strings.TrimSpace(t.Label) == "" {
		return fail("topics[%d]: empty label", i)
	}
	if len(t.Aliases) == 0 {
		return fail("topic %q: no aliases", t.Label)
	}
	for _, a := range t.Aliases {
		if strings.TrimSpace(a) == "" {
			return fail("topic %q: blank alias", t.Label)
		}
	}
	switch t.Kind {
	case TopicMain:
		c.main++
		if t.Section != "" {
			return fail("main topic %q has a section", t.Label)
		}
		t.Offset = 0
		return nil
	case TopicSide:
		c.side++
	case TopicMention:
		c.mention++
	default:
		return fail("topic %q: kind %q is not main, side or mention", t.Label, t.Kind)
	}
	off := indexHeading(d.Text, t.Section)
	if off < 0 {
		return fail("topic %q: no \"## %s\" line in text", t.Label, t.Section)
	}
	t.Offset = off
	if t.Kind == TopicSide && t.Late() {
		c.late++
	}
	return nil
}

// indexHeading returns the byte offset of the line "## section" in text, or
// -1 when there is none.
func indexHeading(text, section string) int {
	if strings.TrimSpace(section) == "" {
		return -1
	}
	want := "## " + section
	off := 0
	for _, line := range strings.SplitAfter(text, "\n") {
		if strings.TrimRight(line, "\r\n") == want {
			return off
		}
		off += len(line)
	}
	return -1
}

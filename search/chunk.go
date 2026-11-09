package search

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

// Chunker splits a document into the pieces that get embedded individually.
// Embedding models handle short passages far better than whole documents, so
// the quality of search hinges on sensible chunks.
type Chunker interface {
	Chunk(text string) []string
}

// MarkdownChunker splits markdown text into heading-delimited sections and
// packs their paragraphs into chunks of at most MaxChars. Each chunk is
// prefixed with its heading path ("Title › Section") so the embedding keeps
// the surrounding context even when the paragraph alone is ambiguous; the
// prefix counts towards MaxChars and is cut to a quarter of it. A heading
// with no text under it and no subheading is a chunk of its heading path
// alone, so its words stay searchable. Headings are ATX lines ("# Title")
// indented by at most three spaces and outside fenced code blocks.
//
// Paragraphs are separated by blank lines, which may hold spaces or tabs;
// CRLF and CR line endings are read as LF. A paragraph longer than a chunk
// is split between lines, so code blocks and lists keep their structure,
// and a line longer than a chunk between words.
type MarkdownChunker struct {
	// MaxChars caps the chunk size in bytes (len, not runes), defaulting to
	// 1200 (roughly 300 tokens of English, comfortably within small
	// embedding models' input windows). A single rune wider than a chunk
	// still makes a chunk of its own.
	MaxChars int
}

const defaultMaxChars = 1200

// Chunk implements Chunker.
func (c *MarkdownChunker) Chunk(text string) []string {
	maxChars := c.maxChars()
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	var out []string
	var headings []string // heading text per level, index 0 = h1
	var section []string  // content lines of the current section
	var open fence        // the fenced code block the line is in, if any
	bodiless := false     // the last heading has no text under it yet

	// emitPath makes a chunk of the heading path alone, for a heading with
	// no text of its own: its words stay searchable, in a short chunk that
	// embeds as just that topic, instead of being lost or folded into the
	// text of a sibling section they do not describe
	emitPath := func() {
		if path := strings.TrimSpace(truncate(headingPath(headings), maxChars)); path != "" {
			out = append(out, path)
		}
	}
	flush := func() {
		body := strings.TrimSpace(strings.Join(section, "\n"))
		section = section[:0]
		if body == "" {
			return
		}
		bodiless = false
		// the prefix is part of every chunk: cut it to a quarter of the
		// chunk and pack the text into what is left
		prefix := strings.TrimSpace(truncate(headingPath(headings), maxChars/4))
		budget := maxChars
		if prefix != "" {
			budget -= len(prefix) + len("\n\n")
		}
		for _, part := range packParagraphs(body, budget) {
			if prefix != "" {
				part = prefix + "\n\n" + part
			}
			out = append(out, part)
		}
	}

	for _, line := range strings.Split(text, "\n") {
		if open.char != 0 {
			if open.closedBy(line) {
				open = fence{}
			}
			section = append(section, line)
			continue
		}
		if f, ok := openingFence(line); ok {
			open = f
			section = append(section, line)
			continue
		}
		level, title := heading(line)
		if level == 0 {
			section = append(section, line)
			continue
		}
		flush()
		// a bodiless heading is in the path of a deeper one that follows,
		// so only one followed by a sibling or a shallower heading needs a
		// chunk of its own
		if bodiless && level <= len(headings) {
			emitPath()
		}
		headings = setHeading(headings, level, title)
		bodiless = true
	}
	flush()
	if bodiless {
		emitPath()
	}
	return out
}

// Fingerprint implements Fingerprinter: the algorithm's version and the
// effective MaxChars, so the zero value and the default spelled out agree.
// Version 2 gives a heading without text a chunk of its own and reads an
// ATX heading only with at most three spaces of indent and outside fenced
// code.
func (c *MarkdownChunker) Fingerprint() string {
	return fmt.Sprintf("markdown/2 max=%d", c.maxChars())
}

// maxChars is MaxChars or its default.
func (c *MarkdownChunker) maxChars() int {
	if c.MaxChars <= 0 {
		return defaultMaxChars
	}
	return c.MaxChars
}

// maxIndent is the most spaces an ATX heading or a code fence may be
// indented by; a line indented further (or by a tab) is indented code.
const maxIndent = 3

// unindent returns line without its leading spaces, false when there are
// more than maxIndent of them or a tab among them.
func unindent(line string) (string, bool) {
	rest := strings.TrimLeft(line, " ")
	if len(line)-len(rest) > maxIndent || strings.HasPrefix(rest, "\t") {
		return "", false
	}
	return rest, true
}

// heading returns the ATX heading level of a line and its text, or 0 if it
// is not a heading.
func heading(line string) (int, string) {
	line, ok := unindent(line)
	if !ok {
		return 0, ""
	}
	level := 0
	for level < len(line) && line[level] == '#' {
		level++
	}
	if level == 0 || level > 6 {
		return 0, ""
	}
	if level == len(line) || line[level] == ' ' || line[level] == '\t' {
		return level, strings.TrimSpace(line[level:])
	}
	return 0, ""
}

// fence is an open fenced code block: the character of its fence (` or ~)
// and the fence's length; the zero value is no block.
type fence struct {
	char byte
	n    int
}

// openingFence returns the fence a line opens: at least three backticks or
// tildes after at most maxIndent spaces.
func openingFence(line string) (fence, bool) {
	line, ok := unindent(line)
	if !ok || len(line) < 3 || (line[0] != '`' && line[0] != '~') {
		return fence{}, false
	}
	n := len(line) - len(strings.TrimLeft(line, line[:1]))
	if n < 3 {
		return fence{}, false
	}
	return fence{char: line[0], n: n}, true
}

// closedBy reports whether line closes the block: a fence of the same
// character, at least as long, after at most maxIndent spaces and with
// nothing but blanks after it.
func (f fence) closedBy(line string) bool {
	line, ok := unindent(line)
	if !ok {
		return false
	}
	rest := strings.TrimLeft(line, string(f.char))
	return len(line)-len(rest) >= f.n && strings.TrimSpace(rest) == ""
}

// setHeading truncates the heading stack to level and records text there,
// padding skipped levels (an h3 straight under an h1) with empty entries.
func setHeading(headings []string, level int, text string) []string {
	if level > len(headings) {
		for len(headings) < level-1 {
			headings = append(headings, "")
		}
		return append(headings, text)
	}
	headings = headings[:level]
	headings[level-1] = text
	return headings
}

// headingPath renders the heading stack as a single context line.
func headingPath(headings []string) string {
	parts := make([]string, 0, len(headings))
	for _, h := range headings {
		if h != "" {
			parts = append(parts, h)
		}
	}
	return strings.Join(parts, " › ")
}

// blankLine separates paragraphs: an empty line, or one of spaces and tabs.
var blankLine = regexp.MustCompile(`\n[ \t]*\n`)

// packParagraphs greedily joins the blank-line-separated paragraphs of body
// into pieces of at most maxChars bytes, splitting a paragraph longer than
// that between its lines.
func packParagraphs(body string, maxChars int) []string {
	var paras []string
	for _, para := range blankLine.Split(body, -1) {
		if para = strings.TrimSpace(para); para != "" {
			paras = append(paras, para)
		}
	}
	return pack(paras, "\n\n", maxChars, splitLines)
}

// splitLines packs the lines of an over-long paragraph, keeping their
// indentation, and splits a line longer than maxChars between its words.
func splitLines(para string, maxChars int) []string {
	lines := strings.Split(para, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	return pack(lines, "\n", maxChars, splitWords)
}

// splitWords packs the words of an over-long line and cuts a word longer
// than maxChars at rune boundaries.
func splitWords(line string, maxChars int) []string {
	return pack(strings.Fields(line), " ", maxChars, splitRunes)
}

// splitRunes cuts a word into pieces of at most maxChars bytes at rune
// boundaries; a rune wider than maxChars is a piece of its own.
func splitRunes(word string, maxChars int) []string {
	var out []string
	for word != "" {
		head := truncate(word, maxChars)
		if head == "" {
			_, size := utf8.DecodeRuneInString(word)
			head = word[:size]
		}
		out = append(out, head)
		word = word[len(head):]
	}
	return out
}

// pack greedily joins pieces with sep into parts of at most maxChars bytes;
// a piece longer than that goes to split, whose parts are kept as they come.
func pack(pieces []string, sep string, maxChars int, split func(string, int) []string) []string {
	var out []string
	var buf strings.Builder
	flush := func() {
		if buf.Len() > 0 {
			out = append(out, buf.String())
			buf.Reset()
		}
	}
	for _, p := range pieces {
		if len(p) > maxChars {
			flush()
			out = append(out, split(p, maxChars)...)
			continue
		}
		if buf.Len() > 0 && buf.Len()+len(sep)+len(p) > maxChars {
			flush()
		}
		if buf.Len() > 0 {
			buf.WriteString(sep)
		}
		buf.WriteString(p)
	}
	flush()
	return out
}

// truncate cuts s to at most n bytes at a rune boundary.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

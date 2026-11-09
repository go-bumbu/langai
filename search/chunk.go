package search

import (
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
// prefix counts towards MaxChars and is cut to a quarter of it.
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
	inFence := false

	flush := func() {
		body := strings.TrimSpace(strings.Join(section, "\n"))
		section = section[:0]
		if body == "" {
			return
		}
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
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			section = append(section, line)
			continue
		}
		if level := headingLevel(trimmed); level > 0 && !inFence {
			flush()
			headings = setHeading(headings, level, strings.TrimSpace(trimmed[level:]))
			continue
		}
		section = append(section, line)
	}
	flush()

	// a document of nothing but headings still deserves one searchable chunk
	if len(out) == 0 {
		if path := strings.TrimSpace(truncate(headingPath(headings), maxChars)); path != "" {
			out = append(out, path)
		}
	}
	return out
}

// maxChars is MaxChars or its default.
func (c *MarkdownChunker) maxChars() int {
	if c.MaxChars <= 0 {
		return defaultMaxChars
	}
	return c.MaxChars
}

// headingLevel returns the ATX heading level of a line, or 0 if it is not a
// heading.
func headingLevel(line string) int {
	level := 0
	for level < len(line) && line[level] == '#' {
		level++
	}
	if level == 0 || level > 6 {
		return 0
	}
	if level == len(line) || line[level] == ' ' || line[level] == '\t' {
		return level
	}
	return 0
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

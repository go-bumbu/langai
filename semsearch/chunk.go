package semsearch

import "strings"

// Chunker splits a document into the pieces that get embedded individually.
// Embedding models handle short passages far better than whole documents, so
// the quality of search hinges on sensible chunks.
type Chunker interface {
	Chunk(text string) []string
}

// MarkdownChunker splits markdown text into heading-delimited sections and
// packs their paragraphs into chunks of at most MaxChars. Each chunk is
// prefixed with its heading path ("Title › Section") so the embedding keeps
// the surrounding context even when the paragraph alone is ambiguous.
type MarkdownChunker struct {
	// MaxChars caps the chunk size, defaulting to 1200 (roughly 300 tokens,
	// comfortably within small embedding models' input windows).
	MaxChars int
}

const defaultMaxChars = 1200

// Chunk implements Chunker.
func (c *MarkdownChunker) Chunk(text string) []string {
	maxChars := c.MaxChars
	if maxChars <= 0 {
		maxChars = defaultMaxChars
	}

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
		prefix := headingPath(headings)
		for _, part := range packParagraphs(body, maxChars) {
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
		if path := headingPath(headings); path != "" {
			out = append(out, path)
		}
	}
	return out
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

// packParagraphs greedily joins blank-line-separated paragraphs into pieces
// of at most maxChars, hard-splitting single paragraphs that are longer than
// that on their own.
func packParagraphs(body string, maxChars int) []string {
	var out []string
	var buf strings.Builder
	for _, para := range strings.Split(body, "\n\n") {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		if buf.Len() > 0 && buf.Len()+2+len(para) > maxChars {
			out = append(out, buf.String())
			buf.Reset()
		}
		if len(para) > maxChars {
			if buf.Len() > 0 {
				out = append(out, buf.String())
				buf.Reset()
			}
			out = append(out, splitWords(para, maxChars)...)
			continue
		}
		if buf.Len() > 0 {
			buf.WriteString("\n\n")
		}
		buf.WriteString(para)
	}
	if buf.Len() > 0 {
		out = append(out, buf.String())
	}
	return out
}

// splitWords hard-splits an over-long paragraph at word boundaries, falling
// back to a rune cut for a single "word" longer than maxChars.
func splitWords(para string, maxChars int) []string {
	var out []string
	var buf strings.Builder
	for _, word := range strings.Fields(para) {
		if buf.Len() > 0 && buf.Len()+1+len(word) > maxChars {
			out = append(out, buf.String())
			buf.Reset()
		}
		for len(word) > maxChars {
			head, tail := runePrefix(word, maxChars)
			out = append(out, head)
			word = tail
		}
		if buf.Len() > 0 {
			buf.WriteString(" ")
		}
		buf.WriteString(word)
	}
	if buf.Len() > 0 {
		out = append(out, buf.String())
	}
	return out
}

// runePrefix cuts s after the largest rune boundary within maxBytes.
func runePrefix(s string, maxBytes int) (head, tail string) {
	cut := 0
	for i := range s {
		if i > maxBytes {
			break
		}
		cut = i
	}
	if cut == 0 { // first rune alone exceeds maxBytes; take it anyway
		for i := range s {
			if i > 0 {
				cut = i
				break
			}
		}
		if cut == 0 {
			return s, ""
		}
	}
	return s[:cut], s[cut:]
}

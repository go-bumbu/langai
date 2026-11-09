package search

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestMarkdownChunkerSections(t *testing.T) {
	c := &MarkdownChunker{}
	text := "# Guide\n\nIntro paragraph.\n\n## Install\n\nRun the installer.\n\n## Use\n\nOpen the app."
	got := c.Chunk(text)
	if len(got) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %#v", len(got), got)
	}
	if got[0] != "Guide\n\nIntro paragraph." {
		t.Errorf("unexpected first chunk: %q", got[0])
	}
	if got[1] != "Guide › Install\n\nRun the installer." {
		t.Errorf("expected heading path prefix, got %q", got[1])
	}
	if got[2] != "Guide › Use\n\nOpen the app." {
		t.Errorf("expected heading path prefix, got %q", got[2])
	}
}

func TestMarkdownChunkerPlainText(t *testing.T) {
	c := &MarkdownChunker{}
	got := c.Chunk("just one paragraph, no headings at all")
	if len(got) != 1 || got[0] != "just one paragraph, no headings at all" {
		t.Fatalf("unexpected chunks: %#v", got)
	}
}

func TestMarkdownChunkerEmpty(t *testing.T) {
	c := &MarkdownChunker{}
	if got := c.Chunk(""); len(got) != 0 {
		t.Fatalf("expected no chunks for empty text, got %#v", got)
	}
	if got := c.Chunk("   \n\n  "); len(got) != 0 {
		t.Fatalf("expected no chunks for blank text, got %#v", got)
	}
}

func TestMarkdownChunkerHeadingOnly(t *testing.T) {
	c := &MarkdownChunker{}
	got := c.Chunk("# Lonely title")
	if len(got) != 1 || got[0] != "Lonely title" {
		t.Fatalf("expected the heading itself as chunk, got %#v", got)
	}
}

func TestMarkdownChunkerSplitsLongSections(t *testing.T) {
	c := &MarkdownChunker{MaxChars: 40}
	paras := []string{
		"first paragraph with some words",
		"second paragraph with more words",
		"third one",
	}
	got := c.Chunk("# T\n\n" + strings.Join(paras, "\n\n"))
	if len(got) < 2 {
		t.Fatalf("expected the section split into several chunks, got %#v", got)
	}
	for _, chunk := range got {
		body := strings.TrimPrefix(chunk, "T\n\n")
		if len(body) > 40 {
			t.Errorf("chunk body exceeds MaxChars: %q", body)
		}
	}
}

func TestMarkdownChunkerHardSplitsLongParagraph(t *testing.T) {
	c := &MarkdownChunker{MaxChars: 30}
	got := c.Chunk(strings.Repeat("word ", 20))
	if len(got) < 3 {
		t.Fatalf("expected hard split of a long paragraph, got %d chunks", len(got))
	}
	for _, chunk := range got {
		if len(chunk) > 30 {
			t.Errorf("chunk exceeds MaxChars: %q", chunk)
		}
	}
}

func TestMarkdownChunkerIgnoresHeadingsInCodeFences(t *testing.T) {
	c := &MarkdownChunker{}
	text := "# Real\n\ntext\n\n```\n# not a heading\n```\n\nmore text"
	got := c.Chunk(text)
	for _, chunk := range got {
		if strings.Contains(chunk, "Real › not a heading") {
			t.Fatalf("fenced pseudo-heading leaked into heading path: %q", chunk)
		}
	}
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "# not a heading") {
		t.Fatalf("fenced content should stay in the chunk body: %#v", got)
	}
}

// TestMarkdownChunker_LineEndingsAndBlankLines: CRLF line endings and blank
// lines holding spaces or tabs separate paragraphs exactly as plain blank
// lines do, so such a document is chunked like its LF twin.
func TestMarkdownChunker_LineEndingsAndBlankLines(t *testing.T) {
	c := &MarkdownChunker{MaxChars: 40}
	lf := "# Title\n\nfirst paragraph with some words\n\nsecond paragraph with more words\n\n- a list item\n- another item"
	want := c.Chunk(lf)
	for name, text := range map[string]string{
		"CRLF":             strings.ReplaceAll(lf, "\n", "\r\n"),
		"CR":               strings.ReplaceAll(lf, "\n", "\r"),
		"spaces and tabs":  strings.ReplaceAll(lf, "\n\n", "\n \t \n"),
		"several blanks":   strings.ReplaceAll(lf, "\n\n", "\n\n  \n\n"),
		"CRLF with spaces": strings.ReplaceAll(strings.ReplaceAll(lf, "\n\n", "\n  \n"), "\n", "\r\n"),
	} {
		if got := c.Chunk(text); !slices.Equal(got, want) {
			t.Errorf("%s: %q\nwant %q", name, got, want)
		}
	}
	if len(want) < 3 || !strings.Contains(want[len(want)-1], "- a list item\n- another item") {
		t.Errorf("the list lost its lines: %q", want)
	}
}

// TestMarkdownChunker_LongHeadingCountsTowardsMaxChars: the heading path
// prefixed to every chunk is part of the chunk's size, and a heading longer
// than a quarter of MaxChars is cut, so it cannot crowd out the text.
func TestMarkdownChunker_LongHeadingCountsTowardsMaxChars(t *testing.T) {
	c := &MarkdownChunker{MaxChars: 120}
	heading := strings.Repeat("überlange Überschrift ", 20)
	body := strings.Repeat("some words in a paragraph. ", 20)
	got := c.Chunk("# " + heading + "\n\n" + body)
	if len(got) < 2 {
		t.Fatalf("got %d chunks, want the body split", len(got))
	}
	for _, chunk := range got {
		if len(chunk) > 120 {
			t.Errorf("chunk of %d bytes exceeds MaxChars: %q", len(chunk), chunk)
		}
		if !utf8.ValidString(chunk) {
			t.Errorf("chunk cut inside a rune: %q", chunk)
		}
		prefix, _, _ := strings.Cut(chunk, "\n\n")
		if len(prefix) > 30 || !strings.HasPrefix(heading, prefix) {
			t.Errorf("heading prefix %q is not the heading cut to a quarter of MaxChars", prefix)
		}
	}
}

// TestMarkdownChunker_LongCodeBlockKeepsItsLines: a fenced block too long
// for one chunk is split between lines, not flattened into words.
func TestMarkdownChunker_LongCodeBlockKeepsItsLines(t *testing.T) {
	c := &MarkdownChunker{MaxChars: 80}
	var code strings.Builder
	code.WriteString("```go\n")
	for i := range 10 {
		fmt.Fprintf(&code, "\tx%d := compute(%d)\n", i, i)
	}
	code.WriteString("```")
	got := c.Chunk(code.String())
	if len(got) < 2 {
		t.Fatalf("got %q, want the block split", got)
	}
	for _, chunk := range got {
		if len(chunk) > 80 {
			t.Errorf("chunk exceeds MaxChars: %q", chunk)
		}
	}
	if !strings.Contains(got[0], "```go\n\tx0 := compute(0)\n\tx1 := compute(1)") {
		t.Errorf("the code lost its line structure: %q", got[0])
	}
}

// TestMarkdownChunker_Properties: over generated documents and sizes, no
// chunk exceeds MaxChars (in bytes, the unit the chunker documents) and
// every word of the text, headings aside, is in the chunks in order.
func TestMarkdownChunker_Properties(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	words := []string{"go", "channel", "über", "naïve", "日本語", "x", "routine", "a", "longerword", "-", "*", "1."}
	pick := func() string { return words[rng.IntN(len(words))] }
	for doc := range 200 {
		var text, body strings.Builder
		for range 1 + rng.IntN(12) {
			switch rng.IntN(4) {
			case 0:
				fmt.Fprintf(&text, "%s %s\n", strings.Repeat("#", 1+rng.IntN(3)), pick())
			case 1:
				text.WriteString("```\n")
				for range 1 + rng.IntN(6) {
					line := "    " + pick() + " " + pick() + "\n"
					text.WriteString(line)
					body.WriteString(line)
				}
				text.WriteString("```\n")
			default:
				for range 1 + rng.IntN(30) {
					w := pick()
					text.WriteString(w + []string{" ", " ", "\n", "\t"}[rng.IntN(4)])
					body.WriteString(w + " ")
				}
			}
			text.WriteString([]string{"\n", "\r\n", "\n  \n", "\n\t\n"}[rng.IntN(4)])
		}
		for _, maxChars := range []int{16, 40, 100, 1200} {
			c := &MarkdownChunker{MaxChars: maxChars}
			got := c.Chunk(text.String())
			for _, chunk := range got {
				if len(chunk) > maxChars {
					t.Fatalf("doc %d, MaxChars %d: chunk of %d bytes: %q", doc, maxChars, len(chunk), chunk)
				}
			}
			if missing := missingWord(strings.Fields(body.String()), strings.Fields(strings.Join(got, " "))); missing != "" {
				t.Fatalf("doc %d, MaxChars %d: %q lost from %q\nchunks %q", doc, maxChars, missing, text.String(), got)
			}
		}
	}
}

// missingWord returns the first of want not found, in order, in got, or ""
// when every word is there. Headings prefixed to chunks add words to got,
// never remove any, so an in-order search is the right check; the generated
// words are all shorter than the smallest chunk, so none is ever cut.
func missingWord(want, got []string) string {
	i := 0
	for _, w := range want {
		for i < len(got) && got[i] != w {
			i++
		}
		if i == len(got) {
			return w
		}
		i++
	}
	return ""
}

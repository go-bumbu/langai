package semsearch

import (
	"strings"
	"testing"
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

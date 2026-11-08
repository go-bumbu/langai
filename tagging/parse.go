package tagging

import (
	"strings"
	"unicode/utf8"
)

const (
	// maxTagWords and maxTagRunes bound what counts as a tag: a longer line is
	// the model explaining itself, not tagging.
	maxTagWords = 4
	maxTagRunes = 40

	// wrapMarks are the emphasis, quotes and spaces a model wraps around a tag
	// or a heading. leadingMarks and trailingMarks add the list bullets and
	// hashes it puts in front of a tag and the punctuation it ends one with.
	// All are trimmed from the ends only, so "node.js", "c#" and "c++" survive.
	wrapMarks     = "*_\"'`“”‘’«» \t\r"
	leadingMarks  = "-•–—#>" + wrapMarks
	trailingMarks = ".:;!?" + wrapMarks
)

// parseTags reads the model's reply as a list of tag names: one per line, or
// several separated by commas or semicolons, each cleaned by clean (cleanTag,
// or cleanPath for tree tags), which drops an item that is not a tag. A line
// ending in a colon is a heading ("Here are the tags:") and is skipped. Names
// that differ only in case are one tag, the first spelling wins, and at most
// limit are returned, in reply order.
func parseTags(answer string, limit int, clean func(string) string) []string {
	var names []string
	seen := make(map[string]bool)
	for _, line := range strings.Split(answer, "\n") {
		if strings.HasSuffix(strings.TrimRight(line, wrapMarks), ":") {
			continue
		}
		for _, item := range strings.FieldsFunc(line, func(r rune) bool { return r == ',' || r == ';' }) {
			name := clean(item)
			key := strings.ToLower(name)
			if name == "" || seen[key] {
				continue
			}
			seen[key] = true
			names = append(names, name)
			if len(names) == limit {
				return names
			}
		}
	}
	return names
}

// cleanTag returns the tag s holds, or "" when it holds none.
func cleanTag(s string) string {
	s = strings.TrimLeft(strings.TrimSpace(s), leadingMarks)
	s = strings.TrimLeft(trimListNumber(s), leadingMarks)
	words := strings.Fields(strings.TrimRight(s, trailingMarks))
	name := strings.Join(words, " ")
	if len(words) > maxTagWords || utf8.RuneCountInString(name) > maxTagRunes {
		return ""
	}
	return name
}

// cleanPath returns the tree tag s holds, its levels joined by " / ", or ""
// when it holds none. Levels are split at "/", ">" and "›", which models use
// for paths, and each is cleaned as a tag (see cleanTag) after the list marks
// of the whole line are gone; blank levels are dropped, and a path deeper than
// MaxTreeLevels is cut to it. A level that is too long to be a tag drops the
// whole path, which is the model explaining itself.
func cleanPath(s string) string {
	s = strings.TrimLeft(strings.TrimSpace(s), leadingMarks)
	s = strings.TrimLeft(trimListNumber(s), leadingMarks)
	parts := strings.FieldsFunc(s, func(r rune) bool { return r == '/' || r == '>' || r == '›' })
	var levels []string
	for _, p := range parts {
		if strings.Trim(p, trailingMarks) == "" {
			continue
		}
		level := cleanTag(p)
		if level == "" {
			return ""
		}
		levels = append(levels, level)
	}
	if len(levels) > MaxTreeLevels {
		levels = levels[:MaxTreeLevels]
	}
	return strings.Join(levels, " / ")
}

// trimListNumber removes a list number such as "1. " or "2) " from the front
// of s. The marker has to be followed by a space, so "3.5 inch" and "3d" are
// kept.
func trimListNumber(s string) string {
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == 0 || i+1 >= len(s) || (s[i] != '.' && s[i] != ')') || s[i+1] != ' ' {
		return s
	}
	return s[i+1:]
}

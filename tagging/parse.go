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
// several separated by commas or semicolons, with list numbers, bullets,
// hashes, quotes and trailing punctuation removed. A line ending in a colon is
// a heading ("Here are the tags:") and is skipped, as is any item that is too
// long to be a tag. Names that differ only in case are one tag, the first
// spelling wins, and at most limit are returned, in reply order.
func parseTags(answer string, limit int) []string {
	var names []string
	seen := make(map[string]bool)
	for _, line := range strings.Split(answer, "\n") {
		if strings.HasSuffix(strings.TrimRight(line, wrapMarks), ":") {
			continue
		}
		for _, item := range strings.FieldsFunc(line, func(r rune) bool { return r == ',' || r == ';' }) {
			name := cleanTag(item)
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

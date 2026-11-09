package tagging

import (
	"encoding/json"
	"strings"
	"unicode/utf8"
)

const (
	// maxTagWords and maxTagRunes bound what counts as a tag: a longer line is
	// the model explaining itself, not tagging.
	maxTagWords = 4
	maxTagRunes = 40

	// wrapMarks are the emphasis, quotes and spaces a model wraps around a tag
	// or a heading, and bracketMarks the brackets. leadingMarks and
	// trailingMarks add the list bullets and hashes it puts in front of a tag
	// and the punctuation it ends one with. All are trimmed from the ends
	// only, so "node.js", "c#" and "c++" survive.
	wrapMarks     = "*_\"'`“”‘’«» \t\r"
	bracketMarks  = "[](){}<>"
	leadingMarks  = "-•–—#" + bracketMarks + wrapMarks
	trailingMarks = ".:;!?," + bracketMarks + wrapMarks
	// endMarks are the trailingMarks that are no bracket: what is trimmed
	// before a remark in brackets at the end of a tag is looked for.
	endMarks = ".:;!?," + wrapMarks
	// anyMarks are the leading and the trailing marks together: text made of
	// nothing else holds no word.
	anyMarks = "-•–—#" + endMarks + bracketMarks

	// maxLabelWords is the most words a label before a colon may have ("Here
	// are the tags: a, b"); a longer text is not a label (see cutLabel).
	maxLabelWords = 5
)

// labelWords are the words, lowercased, by which a label before the tags of a
// line says that tags follow ("Tags: a, b"; see cutLabel).
var labelWords = map[string]bool{
	"tag": true, "tags": true, "keyword": true, "keywords": true, "label": true, "labels": true,
	"topic": true, "topics": true, "answer": true, "output": true, "result": true,
	"etiqueta": true, "etiquetas": true, "schlagwort": true, "schlagwörter": true, "schlagworte": true,
}

// parseTags reads the model's reply as a list of tag names, each cleaned by
// clean (cleanTag, or cleanPath for tree tags), which drops an item that is
// not a tag. The reply is first rid of what is no part of the list: a
// reasoning block and code fences (see stripScaffolding). A reply that is
// JSON, an array of strings or an object holding one, gives its strings (see
// jsonTags). Any other reply is read by lines: one name per line, or several
// separated by commas or semicolons; a line ending in a colon is a heading
// ("Here are the tags:") and is skipped, and a label before the names of a
// line is cut ("Tags: a, b"). Names that differ only in case are one tag, the
// first spelling wins, and at most limit are returned, in reply order.
func parseTags(answer string, limit int, clean func(string) string) []string {
	answer = stripScaffolding(answer)
	items, ok := jsonTags(answer)
	if !ok {
		items = lineItems(answer)
	}
	var names []string
	seen := make(map[string]bool)
	for _, item := range items {
		name := clean(item)
		key := strings.ToLower(name)
		if name == "" || seen[key] {
			continue
		}
		seen[key] = true
		names = append(names, name)
		if len(names) == limit {
			break
		}
	}
	return names
}

// stripScaffolding removes what a model wraps its answer in: a reasoning
// block between <think> and </think>, which a reasoning model writes before
// its answer, and the lines of a Markdown code fence. A block that is never
// closed, a model cut off while it was still thinking, takes the rest of the
// reply with it; a closing tag with no opening one, which some chat templates
// leave, takes everything before it.
func stripScaffolding(answer string) string {
	if i := strings.LastIndex(answer, "</think>"); i >= 0 {
		answer = answer[i+len("</think>"):]
	}
	if i := strings.Index(answer, "<think>"); i >= 0 {
		answer = answer[:i]
	}
	if !strings.Contains(answer, "```") && !strings.Contains(answer, "~~~") {
		return answer
	}
	lines := strings.Split(answer, "\n")
	kept := lines[:0]
	for _, line := range lines {
		if t := strings.TrimSpace(line); strings.HasPrefix(t, "```") || strings.HasPrefix(t, "~~~") {
			continue
		}
		kept = append(kept, line)
	}
	return strings.Join(kept, "\n")
}

// jsonTags returns the strings of a reply that is JSON: an array of strings,
// or an object whose "tags" member (in any case), or whose only member, is
// one. Text after the JSON value is the model's closing remark and is
// ignored. Anything else, JSON of another shape or JSON cut short among it,
// is not ok, and is read by lines.
func jsonTags(answer string) (tags []string, ok bool) {
	answer = strings.TrimSpace(answer)
	if !strings.HasPrefix(answer, "[") && !strings.HasPrefix(answer, "{") {
		return nil, false
	}
	var raw json.RawMessage
	if err := json.NewDecoder(strings.NewReader(answer)).Decode(&raw); err != nil {
		return nil, false
	}
	if json.Unmarshal(raw, &tags) == nil {
		return tags, true
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil {
		return nil, false
	}
	for key, value := range object {
		if len(object) == 1 || strings.EqualFold(key, "tags") {
			if json.Unmarshal(value, &tags) == nil {
				return tags, true
			}
		}
	}
	return nil, false
}

// lineItems returns the items of a reply read by lines: each line without a
// heading's is cut of its label (see cutLabel) and split at the commas and
// semicolons outside brackets (see splitItems), and an item written as a
// "key": value pair, the rest of JSON that jsonTags could not read, is its
// value.
func lineItems(answer string) []string {
	var items []string
	for _, line := range strings.Split(answer, "\n") {
		if strings.HasSuffix(strings.TrimRight(line, wrapMarks), ":") {
			continue
		}
		for _, item := range splitItems(cutLabel(line)) {
			if key, value, found := strings.Cut(item, ":"); found && strings.HasSuffix(strings.TrimSpace(key), `"`) && strings.TrimSpace(value) != "" {
				item = value
			}
			items = append(items, item)
		}
	}
	return items
}

// cutLabel removes the label a model puts before the tags of a line: the text
// before the first colon when it is short and says that tags follow, by one of
// labelWords ("Tags: sourdough", "Here are the tags: a, b"). Any other text
// before a colon stays, so a tag with a colon of its own ("Star Trek:
// Voyager") is whole. A colon that follows a comma belongs to a later item,
// and one before "//" to a link.
func cutLabel(line string) string {
	label, rest, found := strings.Cut(line, ":")
	if !found || strings.ContainsAny(label, ",;") || strings.HasPrefix(rest, "//") {
		return line
	}
	words := strings.Fields(strings.ToLower(strings.Trim(label, anyMarks)))
	if len(words) > maxLabelWords {
		return line
	}
	for _, w := range words {
		if labelWords[w] {
			return rest
		}
	}
	return line
}

// splitItems splits a line at its commas and semicolons, except the ones
// inside round brackets: "baking (a technique, mostly)" is one item.
func splitItems(line string) []string {
	var items []string
	depth, start := 0, 0
	for i, r := range line {
		switch r {
		case '(':
			depth++
		case ')':
			depth = max(depth-1, 0)
		case ',', ';':
			if depth == 0 {
				items = append(items, line[start:i])
				start = i + 1
			}
		}
	}
	return append(items, line[start:])
}

// tidy returns s without what a model writes around a tag: list bullets and
// numbers, quotes and brackets, Markdown emphasis, trailing punctuation, and a
// remark in brackets after it ("sourdough (bread)"), with its words separated
// by single spaces. It returns "" for a name left with markup that does not
// close (see unbalanced): half a link or half a quotation is no tag.
func tidy(s string) string {
	s = strings.TrimLeft(strings.TrimSpace(s), leadingMarks)
	s = strings.TrimLeft(trimListNumber(s), leadingMarks)
	s = strings.NewReplacer("**", "", "__", "", "`", "").Replace(s)
	for {
		s = strings.TrimRight(s, endMarks)
		open := openingOf(s)
		if open <= 0 {
			break
		}
		s = s[:open]
	}
	name := strings.Join(strings.Fields(strings.TrimRight(strings.TrimLeft(s, leadingMarks), trailingMarks)), " ")
	if unbalanced(name) {
		return ""
	}
	return name
}

// openingOf returns where the round bracket that s ends with opens, or -1
// when s does not end with one or it never opens.
func openingOf(s string) int {
	if !strings.HasSuffix(s, ")") {
		return -1
	}
	depth := 0
	for i := len(s) - 1; i >= 0; i-- {
		switch s[i] {
		case ')':
			depth++
		case '(':
			if depth--; depth == 0 {
				return i
			}
		}
	}
	return -1
}

// unbalanced reports whether name holds a bracket that does not close or was
// never opened, or an odd number of double quotes.
func unbalanced(name string) bool {
	var open []rune
	quotes := 0
	closes := map[rune]rune{')': '(', ']': '[', '}': '{', '>': '<'}
	for _, r := range name {
		switch r {
		case '"':
			quotes++
		case '(', '[', '{', '<':
			open = append(open, r)
		case ')', ']', '}', '>':
			if len(open) == 0 || open[len(open)-1] != closes[r] {
				return true
			}
			open = open[:len(open)-1]
		}
	}
	return len(open) > 0 || quotes%2 == 1
}

// short returns name when it is short enough to be a tag, and "" when it is
// the model explaining itself (see maxTagWords).
func short(name string) string {
	if len(strings.Fields(name)) > maxTagWords || utf8.RuneCountInString(name) > maxTagRunes {
		return ""
	}
	return name
}

// cleanTag returns the tag s holds, or "" when it holds none.
func cleanTag(s string) string { return short(tidy(s)) }

// pathIndex holds the paths an application has already (see WithTreeTags),
// and each of their upper parts, by the key of their levels (see pathKey), so
// that a path the model reuses is found whatever its case and spacing.
type pathIndex map[string]existingPath

// existingPath is an existing path, or the upper levels of one, as the
// application spells it, and how many levels it has.
type existingPath struct {
	spelling string
	levels   int
}

// newPathIndex indexes existing. A whole path keeps the exact spelling the
// application gave, trimmed; the upper levels of a path that are no path of
// their own are spelled as in the first path that has them, joined by " / ".
func newPathIndex(existing []string) pathIndex {
	ix := pathIndex{}
	split := make([][]string, len(existing))
	for i, p := range existing {
		for l := range strings.SplitSeq(p, "/") {
			if l = strings.Join(strings.Fields(l), " "); l != "" {
				split[i] = append(split[i], l)
			}
		}
		if key := pathKey(split[i]); key != "" {
			if _, ok := ix[key]; !ok {
				ix[key] = existingPath{spelling: strings.TrimSpace(p), levels: len(split[i])}
			}
		}
	}
	for _, levels := range split {
		for n := 1; n < len(levels); n++ {
			if key := pathKey(levels[:n]); ix[key].levels == 0 {
				ix[key] = existingPath{spelling: strings.Join(levels[:n], " / "), levels: n}
			}
		}
	}
	return ix
}

// pathKey is what two spellings of one path share: its levels lowercased,
// their words separated by single spaces, joined by NULs.
func pathKey(levels []string) string {
	keys := make([]string, len(levels))
	for i, l := range levels {
		keys[i] = strings.ToLower(strings.Join(strings.Fields(l), " "))
	}
	return strings.Join(keys, "\x00")
}

// cleanPath returns the tree tag s holds, its levels joined by " / ", or ""
// when it holds none. Levels are split at "/", ">" and "›", which models use
// for paths, and each is tidied (see tidy) after the list marks of the whole
// line are gone; blank levels are dropped.
//
// The longest run of upper levels that is a path in existing, or the start of
// one, is the application's own: it comes back in the application's spelling
// and is not judged, so a path the model reused as asked is returned as the
// application knows it even where one of its levels is longer than a tag the
// model may make up. Each level below it has to be short enough to be a tag
// (see short), or the whole path is the model explaining itself and is
// dropped; levels beyond MaxTreeLevels in all are cut.
func cleanPath(s string, existing pathIndex) string {
	s = strings.TrimLeft(strings.TrimSpace(s), leadingMarks)
	s = strings.TrimLeft(trimListNumber(s), leadingMarks)
	var levels []string
	for _, p := range strings.FieldsFunc(s, func(r rune) bool { return r == '/' || r == '>' || r == '›' }) {
		if strings.Trim(p, anyMarks) == "" {
			continue
		}
		level := tidy(p)
		if level == "" {
			return ""
		}
		levels = append(levels, level)
	}
	var out []string
	depth, reused := 0, 0
	for n := len(levels); n > 0 && reused == 0; n-- {
		if e, ok := existing[pathKey(levels[:n])]; ok {
			out, depth, reused = append(out, e.spelling), e.levels, n
		}
	}
	for _, level := range levels[reused:] {
		if depth >= MaxTreeLevels {
			break
		}
		if short(level) == "" {
			return ""
		}
		out, depth = append(out, level), depth+1
	}
	return strings.Join(out, " / ")
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

package tagging

import (
	"encoding/json"
	"maps"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	// maxTagWords and maxTagRunes bound what counts as a tag: a longer line is
	// the model explaining itself, not tagging.
	maxTagWords = 4
	maxTagRunes = 40

	// wrapMarks are the emphasis, quotes and spaces a model wraps around a tag
	// or a heading. leadingMarks adds the list bullets and hashes it puts in
	// front of a tag, and endMarks the punctuation it ends one with. All are
	// trimmed from the ends only, so "node.js", "c#" and "c++" survive.
	wrapMarks    = "*_\"'`“”‘’«» \t\r"
	leadingMarks = "-•–—#" + wrapMarks
	endMarks     = ".:;!?," + wrapMarks
	// openMarks are the brackets a model puts around a tag and closeMarks the
	// ones that close them, in the same order. A bracket is trimmed only
	// where it stands around the tag and is no part of it (see unwrap).
	openMarks  = "([{<"
	closeMarks = ")]}>"
	// anyMarks are all the marks together: text made of nothing else holds no
	// word.
	anyMarks = "-•–—#" + endMarks + openMarks + closeMarks

	// separators are what a line of several tags is split at (see
	// splitItems): the comma and the semicolon, as they are written in
	// English, in Chinese and Japanese ("、", "，", "；") and in Arabic ("،",
	// "؛").
	separators = ",;、，；،؛"
	// colons end a label or a heading: the colon and the full-width one of
	// Chinese and Japanese.
	colons = ":："

	// maxLabelWords is the most words a label before a colon may have ("Here
	// are the tags: a, b"); a longer text is not a label (see isLabel).
	maxLabelWords = 5
)

var (
	// tagWords are the words, lowercased, that name a list of tags: the last
	// word of a label before the tags of a line ("Tags: a, b"; see isLabel)
	// and the key of the member of a JSON object that holds them (see
	// jsonItems).
	tagWords = map[string]bool{
		"tag": true, "tags": true, "keyword": true, "keywords": true, "label": true, "labels": true,
		"topic": true, "topics": true,
		"etiqueta": true, "etiquetas": true, "schlagwort": true, "schlagwörter": true, "schlagworte": true,
		"标签": true, "標籤": true, "关键词": true, "關鍵詞": true, "タグ": true, "キーワード": true,
	}
	// answerWords label a line like tagWords ("Answer: a, b"), but are no key
	// that says its JSON member holds tags.
	answerWords = map[string]bool{"answer": true, "output": true, "result": true}
	// leadIns are the words that may stand around the tag word of a label
	// ("Here are the tags", "The answer is", "Las etiquetas son", "Hier sind
	// die Schlagwörter"). Any other word makes the text before a colon part
	// of a tag ("Topic modeling: LDA").
	leadIns = map[string]bool{
		"here": true, "are": true, "is": true, "the": true, "my": true, "some": true, "these": true,
		"suggested": true, "possible": true, "relevant": true, "final": true, "top": true,
		"and": true, "for": true, "this": true, "text": true,
		"aquí": true, "están": true, "son": true, "las": true, "los": true,
		"hier": true, "sind": true, "die": true,
	}
	// noTagAnswers are the replies, lowercased and without their spaces, by
	// which a model says in words that a text has no tag, in the languages of
	// tagWords. The list is short on purpose and matched whole (see
	// saysNoTags): a sentence that only means the same is not guessed at.
	noTagAnswers = map[string]bool{
		"notags": true, "notag": true, "none": true, "n/a": true,
		"sinetiquetas": true, "ninguna": true, "ninguno": true,
		"keinetags": true, "keineschlagwörter": true, "keine": true,
	}
	// emphasis removes the Markdown emphasis and code marks inside a tag.
	emphasis = strings.NewReplacer("**", "", "__", "", "`", "")
	// arrows turns the arrows a model writes between the levels of a path
	// into the separator of one (see cleanPath).
	arrows = strings.NewReplacer("->", "/", "→", "/")
	// closes maps each closing bracket to the one that opens it.
	closes = map[rune]rune{')': '(', ']': '[', '}': '{', '>': '<'}
)

// parseTags reads the model's reply as a list of tag names, each cleaned by
// clean (cleanTag, or cleanPath for tree tags), which drops an item that is
// not a tag. The reply is first rid of what is no part of the list: a
// reasoning block and code fences (see stripScaffolding). A reply that is
// JSON gives the strings that are its tags and nothing else (see jsonTags).
// Any other reply is read by lines (see lineItems): one name per line, or
// several separated by commas or semicolons or written as hashtags; a line
// ending in a colon is a heading ("Here are the tags:") and is skipped, and a
// label before the names of a line is cut ("Tags: a, b"). Names that differ
// only in case are one tag, the first spelling wins, and at most limit are
// returned, in reply order. A reply that is one name saying that there are no
// tags ("None", "No tags."; see saysNoTags) gives none.
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
		// with a limit of one, a second name still tells a tag from a reply
		// that says there is none
		if limit > 0 && len(names) >= max(limit, 2) {
			break
		}
	}
	if len(names) == 1 && saysNoTags(names[0]) {
		return nil
	}
	if limit > 0 && len(names) > limit {
		names = names[:limit]
	}
	return names
}

// saysNoTags reports whether name, the one name of a reply, is the model
// saying that the text has no tag: one of noTagAnswers, whatever its case and
// spacing.
func saysNoTags(name string) bool {
	return noTagAnswers[strings.ToLower(strings.Join(strings.Fields(name), ""))]
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

// jsonTags returns the tags of a reply that starts with a JSON value, an
// array or an object: the strings jsonItems finds in it. Text after the value
// is the model's closing remark and is ignored. A reply that parsed as JSON is
// ok and is never read by lines as well, also when it holds no tag: its
// numbers, its keys and its other strings are no tags. Only a reply that is no
// JSON, or JSON cut short, is not ok, and is read by lines.
func jsonTags(answer string) (tags []string, ok bool) {
	answer = strings.TrimSpace(answer)
	if !strings.HasPrefix(answer, "[") && !strings.HasPrefix(answer, "{") {
		return nil, false
	}
	var value any
	if err := json.NewDecoder(strings.NewReader(answer)).Decode(&value); err != nil {
		return nil, false
	}
	return jsonItems(value, false, nil), true
}

// jsonItems adds to items the strings of a decoded JSON value that are tags,
// and returns them. In an array they are its strings, and what its arrays and
// objects hold. In an object they are in the members whose key says so, in
// any case: one of tagWords ("tags", "keywords", "labels", "topics" and their
// kin) and, for an object that is an item of a list (listed), "name". Such a
// member is a string, which may list several ("go, rust"), or an array or
// object read in turn. An object with a single member under another key is a
// wrapper ({"result": [...]}), read when it holds an array or an object.
// Numbers, booleans, nulls and the strings of every other member (a score, a
// language, a summary) are no tags. Members are read in the order of their
// keys, since a decoded object has no other.
func jsonItems(value any, listed bool, items []string) []string {
	switch v := value.(type) {
	case string:
		return append(items, v)
	case []any:
		for _, item := range v {
			items = jsonItems(item, true, items)
		}
	case map[string]any:
		held := false
		for _, key := range slices.Sorted(maps.Keys(v)) {
			word := strings.ToLower(strings.TrimSpace(key))
			if !tagWords[word] && (!listed || word != "name") {
				continue
			}
			held = true
			if list, isString := v[key].(string); isString {
				items = append(items, splitItems(list)...)
				continue
			}
			items = jsonItems(v[key], false, items)
		}
		if held || len(v) != 1 {
			return items
		}
		for _, only := range v {
			if _, isString := only.(string); !isString {
				items = jsonItems(only, false, items)
			}
		}
	}
	return items
}

// lineItems returns the items of a reply read by lines: each line that is no
// heading (it ends in a colon) is cut of its label (see cutLabel), split at
// the separators outside brackets (see splitItems), and each part split once
// more when it is a run of hashtags (see hashtags). An item written as a
// "key": value pair, the rest of JSON that jsonTags could not read, is its
// value, and is dropped when that is a number, a boolean or null.
func lineItems(answer string) []string {
	var items []string
	for _, line := range strings.Split(answer, "\n") {
		if heading := strings.TrimRight(line, wrapMarks); strings.HasSuffix(heading, ":") || strings.HasSuffix(heading, "：") {
			continue
		}
		for _, item := range splitItems(cutLabel(line)) {
			if key, value, found := strings.Cut(item, ":"); found && strings.HasSuffix(strings.TrimSpace(key), `"`) && strings.TrimSpace(value) != "" {
				if jsonScalar(value) {
					continue
				}
				item = value
			}
			items = append(items, hashtags(item)...)
		}
	}
	return items
}

// jsonScalar reports whether value, the text after the colon of a "key":
// value pair, is a JSON number, boolean or null: a score or a flag, not a
// tag. A number in quotes is a string, and may be one ("2024").
func jsonScalar(value string) bool {
	value = strings.Trim(value, " \t\r,]}")
	switch value {
	case "true", "false", "null":
		return true
	}
	_, err := strconv.ParseFloat(value, 64)
	return err == nil
}

// hashtags returns the tags of an item written as hashtags one after the
// other ("#go #rust #devops"): two words or more, each starting with a hash.
// Any other item is returned as it is, so a hash inside a tag stays ("c#
// basics") and a Markdown heading is one item ("# Tags").
func hashtags(item string) []string {
	fields := strings.Fields(item)
	if len(fields) < 2 {
		return []string{item}
	}
	for _, f := range fields {
		if !strings.HasPrefix(f, "#") || strings.Trim(f, "#") == "" {
			return []string{item}
		}
	}
	return fields
}

// cutLabel removes the label a model puts before the tags of a line: the text
// before the first colon, full-width or not, when it is a label (see isLabel:
// "Tags: sourdough", "Here are the tags: a, b", "标签：烘焙"). Any other text
// before a colon stays, so a tag with a colon of its own ("Star Trek:
// Voyager", "Topic modeling: LDA") is whole. A colon that follows a separator
// belongs to a later item, and one before "//" to a link.
func cutLabel(line string) string {
	i := strings.IndexAny(line, colons)
	if i < 0 {
		return line
	}
	_, size := utf8.DecodeRuneInString(line[i:])
	label, rest := line[:i], line[i+size:]
	if strings.ContainsAny(label, separators) || strings.HasPrefix(rest, "//") || !isLabel(label) {
		return line
	}
	return rest
}

// isLabel reports whether label, the text before a colon, says that tags
// follow rather than being the start of one: it is at most maxLabelWords
// words, one of them a tag word or an answer word ("tags", "keywords",
// "answer"; see tagWords) and every other one a lead-in ("here are the";
// see leadIns) or a count. So "Tags", "The keywords", "Top 3 tags" and "Here
// are the tags" are labels, and "Topic modeling", "Keyword research" and
// "Record label" are not: a phrase that merely holds a tag word is a tag's.
func isLabel(label string) bool {
	words := strings.Fields(strings.ToLower(strings.Trim(label, anyMarks)))
	if len(words) == 0 || len(words) > maxLabelWords {
		return false
	}
	named := false
	for _, w := range words {
		switch {
		case tagWords[w] || answerWords[w]:
			named = true
		case !leadIns[w] && !isCount(w):
			return false
		}
	}
	return named
}

// isCount reports whether w is a number, in brackets or not: how many tags a
// label says follow ("Top 3 tags", "Tags (5)").
func isCount(w string) bool {
	w = strings.Trim(w, "()[]")
	if w == "" {
		return false
	}
	for _, r := range w {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// splitItems splits a line at its separators, the comma and the semicolon of
// several scripts, except the ones inside round brackets: "baking (a
// technique, mostly)" is one item.
func splitItems(line string) []string {
	var items []string
	depth, start := 0, 0
	for i, r := range line {
		switch {
		case r == '(':
			depth++
		case r == ')':
			depth = max(depth-1, 0)
		case depth == 0 && strings.ContainsRune(separators, r):
			items = append(items, line[start:i])
			start = i + utf8.RuneLen(r)
		}
	}
	return append(items, line[start:])
}

// tidy returns s without what a model writes around a tag: list bullets,
// numbers and checkboxes, quotes and brackets, Markdown emphasis, trailing
// punctuation, and a remark in brackets after it ("sourdough (bread)"), with
// its words separated by single spaces. What is part of the tag stays: a
// bracket that closes inside it ("(re)insurance"), an inch mark ("5\"
// floppy"), a lone "<" or ">" ("x < y", "a -> b"). It returns "" for a name
// left with markup that does not close (see unbalanced) or with an HTML
// closing tag: half a link, half a quotation or an element is no tag.
func tidy(s string) string {
	s = strings.TrimLeft(strings.TrimSpace(s), leadingMarks)
	s = strings.TrimLeft(trimListMarker(s), leadingMarks)
	s = emphasis.Replace(s)
	for {
		s = strings.TrimRight(s, endMarks)
		open := openingOf(s)
		if open <= 0 {
			break
		}
		s = s[:open]
	}
	name := strings.Join(strings.Fields(unwrap(s)), " ")
	if unbalanced(name) || strings.Contains(name, "</") {
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

// unwrap returns s without the marks and brackets around it. A bracket at
// either end goes when it is around the tag: its partner is at the other end
// ("[sourdough]"), or it has none at all, which is the scaffolding of
// something cut short ("[\"sourdough\"", the start of a JSON array). A
// bracket whose partner is inside the text is part of the tag and stays
// ("(re)insurance", "state of [the art]"). The work is linear in the length
// of s, whatever s holds.
func unwrap(s string) string {
	const brackets = openMarks + closeMarks
	if !strings.ContainsAny(s, brackets) {
		return strings.TrimRight(strings.TrimLeft(s, leadingMarks), endMarks)
	}
	partner := partners(s)
	lo, hi := 0, len(s)
	for {
		lo = hi - len(strings.TrimLeft(s[lo:hi], leadingMarks))
		hi = lo + len(strings.TrimRight(s[lo:hi], endMarks))
		if lo == hi {
			return ""
		}
		first, last := strings.IndexByte(brackets, s[lo]) >= 0, strings.IndexByte(brackets, s[hi-1]) >= 0
		switch {
		case first && (partner[lo] < lo || partner[lo] >= hi): // it pairs with nothing left
			lo++
		case last && (partner[hi-1] < lo || partner[hi-1] >= hi):
			hi--
		case first && partner[lo] == hi-1: // the two wrap what is left
			lo, hi = lo+1, hi-1
		default:
			return s[lo:hi]
		}
	}
}

// partners returns, for each byte of s that is a bracket, where the bracket
// it pairs with is, and -1 for one that pairs with none and for every other
// byte.
func partners(s string) []int {
	partner := make([]int, len(s))
	var open []int
	for i := range len(s) {
		partner[i] = -1
		if strings.IndexByte(openMarks, s[i]) >= 0 {
			open = append(open, i)
			continue
		}
		k := strings.IndexByte(closeMarks, s[i])
		if n := len(open); k >= 0 && n > 0 && s[open[n-1]] == openMarks[k] {
			partner[i], partner[open[n-1]] = open[n-1], i
			open = open[:n-1]
		}
	}
	return partner
}

// unbalanced reports whether name holds a bracket that does not close or was
// never opened, or an odd number of double quotes. A double quote after a
// digit is an inch mark and does not count ("5\" floppy"), and "<" and ">"
// are brackets only when name holds both: one alone is a sign ("x < y", "a ->
// b").
func unbalanced(name string) bool {
	angles := strings.Contains(name, "<") && strings.Contains(name, ">")
	var open []rune
	var prev rune
	quotes := 0
	for _, r := range name {
		switch r {
		case '"':
			if !unicode.IsDigit(prev) {
				quotes++
			}
		case '(', '[', '{', '<':
			if r != '<' || angles {
				open = append(open, r)
			}
		case ')', ']', '}', '>':
			if r == '>' && !angles {
				break
			}
			if len(open) == 0 || open[len(open)-1] != closes[r] {
				return true
			}
			open = open[:len(open)-1]
		}
		prev = r
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

// cleanTag returns the tag s holds, or "" when it holds none. An arrow in it
// stays as the model wrote it ("a -> b"): only a tree tag reads one as the
// step to the next level (see cleanPath).
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
// when it holds none. Levels are split at "/", ">" and "›" and at the arrows
// "->" and "→", which models use for paths ("a -> b" is "a / b" here, where a
// flat tag keeps its arrow; see cleanTag), and each is tidied (see tidy)
// after the list marks of the whole line are gone; blank levels are dropped.
//
// The longest run of upper levels that is a path in existing, or the start of
// one, is the application's own: it comes back in the application's spelling
// and is not judged, so a path the model reused as asked is returned as the
// application knows it even where one of its levels is longer than a tag the
// model may make up. Each level below it has to be short enough to be a tag
// (see short), or the whole path is the model explaining itself and is
// dropped; levels beyond MaxTreeLevels in all are cut. A line with an HTML
// closing tag is markup, as for a flat tag, and no path.
func cleanPath(s string, existing pathIndex) string {
	if strings.Contains(s, "</") {
		return ""
	}
	s = strings.TrimLeft(strings.TrimSpace(s), leadingMarks)
	s = strings.TrimLeft(trimListMarker(s), leadingMarks)
	var levels []string
	for _, p := range strings.FieldsFunc(arrows.Replace(s), func(r rune) bool { return r == '/' || r == '>' || r == '›' }) {
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

// trimListMarker removes a list marker from the front of s: a number such as
// "1. ", "2) " or "(3) ", or a checkbox, "[ ] " or "[x] ". The marker has to
// be followed by a space, so "3.5 inch", "3d" and "(re)insurance" are kept.
func trimListMarker(s string) string {
	for _, box := range []string{"[ ] ", "[x] ", "[X] "} {
		if rest, ok := strings.CutPrefix(s, box); ok {
			return rest
		}
	}
	body, bracketed := strings.CutPrefix(s, "(")
	i := 0
	for i < len(body) && body[i] >= '0' && body[i] <= '9' {
		i++
	}
	if i == 0 || i+1 >= len(body) || body[i+1] != ' ' {
		return s
	}
	if body[i] != ')' && (bracketed || body[i] != '.') {
		return s
	}
	return body[i+1:]
}

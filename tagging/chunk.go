package tagging

import (
	"fmt"
	"math"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Chunk splits text into pieces of at most maxBytes bytes each, of about equal
// size, cut at whitespace where there is some in the second half of a piece
// and on a rune boundary where there is none. Each piece is trimmed, and blank
// ones are dropped. A text of at most maxBytes, or a maxBytes of zero or less,
// comes back whole, trimmed. It is how a Tagger with TaggerConfig.ChunkBytes
// splits an item's text.
func Chunk(text string, maxBytes int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	if maxBytes <= 0 || len(text) <= maxBytes {
		return []string{text}
	}
	var out []string
	for len(text) > maxBytes {
		pieces := (len(text) + maxBytes - 1) / maxBytes
		cut := cutAt(text, (len(text)+pieces-1)/pieces)
		if piece := strings.TrimSpace(text[:cut]); piece != "" {
			out = append(out, piece)
		}
		text = strings.TrimSpace(text[cut:])
	}
	if text != "" {
		out = append(out, text)
	}
	return out
}

// cutAt returns where to end a piece of at most target bytes of text, which
// is longer than target: after the last whitespace in the second half of the
// piece, or else at the last rune boundary at or before target.
func cutAt(text string, target int) int {
	for i := target; i > target/2; {
		r, size := utf8.DecodeLastRuneInString(text[:i])
		if unicode.IsSpace(r) {
			return i
		}
		i -= size
	}
	cut := target
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	if cut == 0 { // a target inside the first rune: take that rune whole
		_, cut = utf8.DecodeRuneInString(text)
	}
	return cut
}

// itemPieces returns the texts an item's text is embedded as: the text itself,
// as given, when it makes one chunk at most (or chunkBytes is zero or less),
// and else its chunks (see Chunk). Each piece is embedded and cached on its
// own, as a query, under the Tagger's model.
func itemPieces(text string, chunkBytes int) []string {
	if chunkBytes <= 0 {
		return []string{text}
	}
	pieces := Chunk(text, chunkBytes)
	if len(pieces) <= 1 {
		return []string{text}
	}
	return pieces
}

// itemScore is an item's score against a tag's vector, from the vectors of
// its pieces (see itemPieces; at least one): the cosine for an item of one,
// the best of the two pieces' cosines and their centroid's for an item of two,
// and the mean of the two best cosines for a longer one.
//
// Measured on 2026-11-07 (docs/superpowers/specs/2026-11-07-tagger-long-items-
// results.md): against the centroid of the pieces, which dilutes a topic one
// section is about with the sections that are not, the two best pieces found
// side topics 0.89 of the time against 0.75, scored by exact label, at no cost
// in precision. The best piece alone found 0.96 but let one paragraph carry a
// tag: 5.0 tags a document were labels of other documents, against 3.5. (Under
// the topic matching zarf/eval used before 2026-11-09 the same comparison read
// 0.68 against 0.58, and 0.74 with 3.5 off-topic tags against 2.0; it was not
// run again under today's.) Two pieces are both "the
// two best", so their mean would only dilute; a page of two, a header and two
// halves, is best read whole as well as by halves, and their centroid scores
// both of its topics where neither half holds one whole.
//
// A piece that is not as long as the tag's vector is an error: cosine would
// answer 0 for it, which reads as "unrelated" when it means "not comparable".
func itemScore(pieces [][]float32, tag []float32) (float64, error) {
	for _, p := range pieces {
		if len(p) != len(tag) {
			return 0, fmt.Errorf("the text has a vector of %d dimensions and the tag one of %d", len(p), len(tag))
		}
	}
	switch len(pieces) {
	case 1:
		return cosine(pieces[0], tag), nil
	case 2:
		whole, err := centroid(pieces)
		if err != nil {
			return 0, err
		}
		return max(cosine(pieces[0], tag), cosine(pieces[1], tag), cosine(whole, tag)), nil
	}
	first, second := math.Inf(-1), math.Inf(-1)
	for _, p := range pieces {
		switch c := cosine(p, tag); {
		case c > first:
			first, second = c, first
		case c > second:
			second = c
		}
	}
	return (first + second) / 2, nil
}

// queryVector is the vector an item is looked up by in a TagIndex (see
// TagIndex.Candidates): its one piece's, or the centroid of its pieces, which
// must all be of one length.
func queryVector(pieces [][]float32) ([]float32, error) {
	if len(pieces) == 1 {
		return pieces[0], nil
	}
	return centroid(pieces)
}

// memberVector is the vector a member enters its tag's blend with, from the
// vectors of its pieces (at least one): its one piece's, or the piece closest
// to the tag's name. A long member is often about the tag in one section only,
// and the centroid of its pieces sits nearer its other topics; measured over
// long members (see itemScore), the best piece tagged 0.92 F1 against 0.77,
// and its scores stay on the scale of a short member's, so one MemberMinScore
// holds for both.
func memberVector(pieces [][]float32, nameVec []float32) []float32 {
	best, bestCos := pieces[0], cosine(pieces[0], nameVec)
	for _, p := range pieces[1:] {
		if c := cosine(p, nameVec); c > bestCos {
			best, bestCos = p, c
		}
	}
	return best
}

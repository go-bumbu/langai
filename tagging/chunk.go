package tagging

import (
	"context"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/go-bumbu/langai"
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

// itemModel is the model an item's vector is cached under: the Tagger's
// model, marked with the chunk size when items are embedded in chunks, so a
// change of ChunkBytes re-embeds every item as a change of model does.
func itemModel(cfg TaggerConfig) string {
	if cfg.ChunkBytes <= 0 {
		return cfg.Model
	}
	return fmt.Sprintf("%s+chunks=%d", cfg.Model, cfg.ChunkBytes)
}

// embedItem embeds an item's text with emb as a query, the way Suggest scores
// it. With chunkBytes above zero, a longer text is embedded in pieces (see
// Chunk), in one call, and its vector is the unit-length mean of theirs, so
// all of it counts instead of the part an embedder reads before it truncates.
func embedItem(ctx context.Context, emb langai.Embedder, chunkBytes int, text string) ([]float32, error) {
	pieces := Chunk(text, chunkBytes)
	if len(pieces) <= 1 {
		return embedOne(ctx, emb, langai.RoleQuery, text)
	}
	vs, err := emb.Embed(ctx, langai.RoleQuery, pieces)
	if err != nil {
		return nil, err
	}
	if len(vs) != len(pieces) {
		return nil, fmt.Errorf("tagging: embedder returned %d vectors for %d pieces of an item", len(vs), len(pieces))
	}
	dims := len(vs[0])
	if d := emb.Dimensions(); d > 0 {
		dims = d
	}
	for _, v := range vs {
		if len(v) == 0 || len(v) != dims {
			return nil, fmt.Errorf("tagging: embedder returned a vector of %d dimensions, want %d", len(v), dims)
		}
	}
	return centroid(vs, dims), nil
}

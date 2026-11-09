package tagging

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/go-bumbu/langai"
)

// windowBytes is the most a window of a long text holds, the part the model
// reads in one call. Measured on 2026-11-07 over 24 long golden documents
// (docs/superpowers/specs/2026-11-07-generator-long-texts-results.md), 6 KB
// windows found the topics of later sections best: 2 KB ones made the model
// write near-duplicates in nearly three times the calls. A window, the system
// prompt and the reply fit the 4096-token context of provider/yzma's default
// Completer.
const windowBytes = 6144

// maxTitleBytes is the longest first line of a text that is read as its title.
// A window past the first, cut from the middle of the text, often cannot tell
// what its section is about, so the title is repeated at its head. Measured on
// the same 24 documents: side-topic recall 0.656 without, 0.708 with.
const maxTitleBytes = 200

// titleOf returns the first line of text, trimmed, when it is a title: at
// most maxTitleBytes and followed by more text. Otherwise it is empty.
func titleOf(text string) string {
	line, rest, found := strings.Cut(text, "\n")
	line = strings.TrimSpace(line)
	if !found || len(line) > maxTitleBytes || strings.TrimSpace(rest) == "" {
		return ""
	}
	return line
}

// generateLong is Generate for a text longer than ChunkBytes, already cut and
// trimmed: the model writes tags for each window of it (see windowBytes), and
// each name is scored by longScore over the text's chunks. The windows go to
// the model as Chunk returns them, the ones after the first headed by the
// text's title (see maxTitleBytes); the chunks that score are the text's own.
func (g *Generator) generateLong(ctx context.Context, text string, minScore float64) ([]Tag, error) {
	clean := g.clean
	var names []string
	seen := make(map[string]bool)
	title := titleOf(text)
	for i, w := range Chunk(text, windowBytes) {
		if i > 0 && title != "" {
			w = title + "\n" + w
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		answer, err := g.llm.Complete(ctx, g.prompt, w)
		if err != nil {
			return nil, fmt.Errorf("tagging: write tags: %w", err)
		}
		// a window with no usable tag adds nothing, and the others still count
		for _, name := range parseTags(answer, g.cfg.Candidates, clean) {
			if key := strings.ToLower(name); !seen[key] {
				seen[key] = true
				names = append(names, name)
			}
		}
	}
	if len(names) == 0 {
		return nil, nil
	}
	chunks := Chunk(text, g.cfg.ChunkBytes)
	chunkVecs, err := g.embed(ctx, langai.RoleQuery, chunks)
	if err != nil {
		return nil, err
	}
	nameVecs, err := g.embed(ctx, langai.RoleDocument, names)
	if err != nil {
		return nil, err
	}
	var tags []Tag
	cos := make([]float64, len(chunkVecs))
	for i, name := range names {
		for j, cv := range chunkVecs {
			if len(nameVecs[i]) != len(cv) {
				return nil, fmt.Errorf("tagging: the embedder returned a %d-dimensional vector for %q but a %d-dimensional one for chunk %d of the text", len(nameVecs[i]), name, len(cv), j+1)
			}
			cos[j] = cosine(cv, nameVecs[i])
		}
		score := longScore(cos)
		// "not at or above" and not "below": every comparison with a NaN is
		// false, and a NaN cutoff or score must pick nothing
		if !(score >= minScore) {
			continue
		}
		tags = append(tags, Tag{Name: name, Score: score})
	}
	slices.SortStableFunc(tags, func(a, b Tag) int { return cmp.Compare(b.Score, a.Score) })
	if len(tags) > g.cfg.Candidates {
		tags = tags[:g.cfg.Candidates]
	}
	return tags, nil
}

// longScore is a name's score on a long text from its cosines to the text's
// chunks (at least one): the best of them. It is both what is compared with
// the cutoff and what the tags are ranked by. The measurement behind
// windowBytes chose it over the cosine to the chunks' centroid and over the
// mean of the best few, which both dilute a topic that one section is about
// with the sections that are not, exactly the topics the windows find; it
// keeps the per-chunk cosine scale, so DefaultGeneratorMinScore holds on long
// texts as on short ones.
func longScore(cos []float64) float64 {
	return slices.Max(cos)
}

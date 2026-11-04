package tagging

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/go-bumbu/langai"
)

const (
	// DefaultGeneratorMinScore is a cutoff that fits the default models: Qwen3
	// 4B Instruct writing the tags and EmbeddingGemma scoring them (provider/yzma).
	// Measured on 2026-10-25 over ten golden documents in English, Spanish and
	// German, the ten tags the model wrote for each scored 0.31 to 0.68, the ones
	// on a text's main topics at 0.5 and above; 6 to 10 of them reached 0.45,
	// and what fell below it was side detail (the steam for a loaf, a side
	// salad). That is a judgment by eye on a small set, not a measured
	// precision, and another embedder scores on another scale, so choose the
	// cutoff again for it.
	DefaultGeneratorMinScore = 0.45
	// DefaultCandidates is how many tags the model is asked for when
	// GeneratorConfig.Candidates is zero.
	DefaultCandidates = 10
	// DefaultMaxBytes is how much of a text is read when GeneratorConfig.MaxBytes
	// is zero: about the 2 KB a Tagger item is judged on, and about what the
	// default embedding model reads (its 512-token window). The model then
	// writes tags about the words they are scored against.
	DefaultMaxBytes = 2048
)

// Tag is one generated tag and how well it fits the text.
type Tag struct {
	// Name is the tag as the model wrote it, without list markers, quotes or
	// trailing punctuation. Its case is the model's.
	Name string
	// Score is the cosine between the embeddings of the text and of the name:
	// 1 for identical vectors and about 0 for unrelated ones.
	Score float64
}

// GeneratorConfig tunes a Generator. The zero value is valid.
type GeneratorConfig struct {
	// Candidates is how many tags the model is asked for, and the most that
	// are scored. Zero means DefaultCandidates.
	Candidates int
	// MaxBytes is how much of a text is read, cut on a rune boundary: the model
	// and the embedder both see exactly this prefix. Zero means DefaultMaxBytes.
	MaxBytes int
}

// Generator writes and scores tags with one Completer and one Embedder.
type Generator struct {
	llm langai.Completer
	emb langai.Embedder
	cfg GeneratorConfig
}

// NewGenerator returns a Generator that has llm write the tags and emb score
// them.
func NewGenerator(llm langai.Completer, emb langai.Embedder, cfg GeneratorConfig) (*Generator, error) {
	if llm == nil {
		return nil, errors.New("tagging: nil Completer")
	}
	if emb == nil {
		return nil, errors.New("tagging: nil Embedder")
	}
	if cfg.Candidates < 0 || cfg.MaxBytes < 0 {
		return nil, fmt.Errorf("tagging: Candidates and MaxBytes must not be negative, got %d and %d", cfg.Candidates, cfg.MaxBytes)
	}
	cfg.Candidates = cmp.Or(cfg.Candidates, DefaultCandidates)
	cfg.MaxBytes = cmp.Or(cfg.MaxBytes, DefaultMaxBytes)
	return &Generator{llm: llm, emb: emb, cfg: cfg}, nil
}

// systemPrompt sets the task; %d is the number of tags asked for. The text
// itself is the user message, so a text that talks to the model is still only
// the thing being tagged.
const systemPrompt = "You write tags for texts. Reply with up to %d tags for the text you are given: " +
	"one tag per line, the most relevant first, and nothing else (no numbers, no bullets, no explanations). " +
	"A tag is one to three words naming a topic, a thing or a kind of content the text is about. " +
	"Write every tag in the language of the text."

// Generate returns the tags for text that score at least minScore, best score
// first and equal scores in the order the model wrote them, or nil when none
// does. A minScore of -1 returns every tag the model wrote, and a NaN one
// returns none; DefaultGeneratorMinScore fits the default models. Only the
// first GeneratorConfig.MaxBytes bytes of text are read, and a blank text asks
// nobody. A model that writes no usable tag is not an error.
//
// Every call has the model write its tags afresh, which takes seconds on a CPU
// and is deterministic only as far as the Completer is.
func (g *Generator) Generate(ctx context.Context, text string, minScore float64) ([]Tag, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	text = strings.TrimSpace(clip(strings.ToValidUTF8(text, "�"), g.cfg.MaxBytes))
	if text == "" {
		return nil, nil
	}
	answer, err := g.llm.Complete(ctx, fmt.Sprintf(systemPrompt, g.cfg.Candidates), text)
	if err != nil {
		return nil, fmt.Errorf("tagging: write tags: %w", err)
	}
	names := parseTags(answer, g.cfg.Candidates)
	if len(names) == 0 {
		return nil, nil
	}
	textVecs, err := g.embed(ctx, langai.RoleQuery, []string{text})
	if err != nil {
		return nil, err
	}
	nameVecs, err := g.embed(ctx, langai.RoleDocument, names)
	if err != nil {
		return nil, err
	}
	var tags []Tag
	for i, name := range names {
		if len(nameVecs[i]) != len(textVecs[0]) {
			return nil, fmt.Errorf("tagging: the embedder returned a %d-dimensional vector for %q but a %d-dimensional one for the text", len(nameVecs[i]), name, len(textVecs[0]))
		}
		score := cosine(textVecs[0], nameVecs[i])
		// "not at or above" and not "below": every comparison with a NaN is
		// false, and a NaN cutoff or score must pick nothing
		if !(score >= minScore) {
			continue
		}
		tags = append(tags, Tag{Name: name, Score: score})
	}
	slices.SortStableFunc(tags, func(a, b Tag) int { return cmp.Compare(b.Score, a.Score) })
	return tags, nil
}

// embed embeds texts as role and checks that every one was answered.
func (g *Generator) embed(ctx context.Context, role langai.Role, texts []string) ([][]float32, error) {
	vecs, err := g.emb.Embed(ctx, role, texts)
	if err != nil {
		return nil, fmt.Errorf("tagging: embed as %s: %w", role, err)
	}
	if len(vecs) != len(texts) {
		return nil, fmt.Errorf("tagging: the embedder returned %d vectors for %d texts", len(vecs), len(texts))
	}
	return vecs, nil
}

// clip returns the longest prefix of s of at most n bytes that ends on a rune
// boundary.
func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

package tagging

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/language"
	"golang.org/x/text/language/display"

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
	// is zero: 64 KB, a long article whole. Past DefaultChunkBytes a text is
	// read in windows, one model call per ~6 KB, so the limit is also what
	// bounds the cost: about eleven calls for a text this long.
	DefaultMaxBytes = 65536
	// DefaultChunkBytes is GeneratorConfig.ChunkBytes when it is zero: about
	// the 2 KB a Tagger item is judged on, and about what the default
	// embedding model reads (its 512-token window), so a tag is scored
	// against words the embedder reads whole.
	DefaultChunkBytes = 2048
)

// Tag is one generated tag and how well it fits the text.
type Tag struct {
	// Name is the tag as the model wrote it, without list markers, quotes or
	// trailing punctuation. Its case is the model's.
	Name string
	// Score is the cosine between the embeddings of the text and of the name:
	// 1 for identical vectors and about 0 for unrelated ones. For a text read
	// in chunks (see GeneratorConfig.ChunkBytes) it is the best cosine between
	// the name and one chunk, on the same scale.
	Score float64
}

// GeneratorConfig tunes a Generator. The zero value is valid.
type GeneratorConfig struct {
	// Candidates is how many tags the model is asked for, in each window of a
	// long text (see ChunkBytes), and the most that are returned. Zero means
	// DefaultCandidates.
	Candidates int
	// MaxBytes is how much of a text is read, cut on a rune boundary: nothing
	// after this prefix reaches the model or the embedder. A text longer than
	// ChunkBytes sees them read it in windows and chunks of the prefix, not
	// whole. Zero means DefaultMaxBytes. Set 2048 to read only the first 2 KB
	// in one call, as before the long-text path: it also bounds the cost to one
	// call, and it is the setting to keep with a chat context under 4096 tokens.
	MaxBytes int
	// ChunkBytes is the longest text read in one piece. A text of at most this
	// many bytes, after the MaxBytes cut, is written about in one model call
	// and scored as a whole. A longer one is read in windows of about 6 KB,
	// one model call each (a short first line is read as the title and
	// repeated at the head of every window after the first), and its tags are scored against chunks of this
	// size (see Chunk), each tag by the chunk it fits best, so a topic of one
	// section is not diluted by the rest of the text. Raising it above about
	// 6 KB sends that much to the model in one call on the short path, which
	// may not fit the default chat context. Zero means DefaultChunkBytes.
	ChunkBytes int
}

// Generator writes and scores tags with one Completer and one Embedder.
type Generator struct {
	llm    langai.Completer
	emb    langai.Embedder
	cfg    GeneratorConfig
	opts   generatorOptions
	prompt string // the system prompt, built once from cfg and opts
}

// generatorOptions are what the GeneratorOptions set.
type generatorOptions struct {
	// language is the BCP 47 tag the tags are written in; empty: the text's own
	language string
	tree     bool
	existing []string // tree paths to extend
	err      error    // a bad option, reported by NewGenerator
}

// GeneratorOption sets how a Generator writes its tags: in which language
// (WithNativeLanguageTags, the default, or WithLanguageTags) and in what shape
// (one to three words, the default, or paths of a tree, WithTreeTags). When
// two options set the same thing, the later one wins.
type GeneratorOption func(*generatorOptions)

// WithNativeLanguageTags has every tag written in the language of the text it
// tags, so a Spanish text gets Spanish tags. It is the default.
func WithNativeLanguageTags() GeneratorOption {
	return func(o *generatorOptions) { o.language = "" }
}

// WithLanguageTags has every tag written in one language, whatever the
// language of the text: lang is a BCP 47 language tag such as "en", "es" or
// "pt-BR". A multilingual embedder scores a tag in one language against a text
// in another, so the cutoff still applies, a little lower across languages.
// NewGenerator refuses a tag it cannot parse.
func WithLanguageTags(lang string) GeneratorOption {
	return func(o *generatorOptions) {
		tag, err := language.Parse(lang)
		if err != nil {
			o.err = fmt.Errorf("tagging: WithLanguageTags(%q): %w", lang, err)
			return
		}
		o.language = tag.String()
	}
}

// WithTreeTags has every tag written as a path of a tree, from a broad
// category down to the topic, its levels joined by " / " ("cooking / baking /
// sourdough"), at most MaxTreeLevels deep, and the tags of one text sharing
// their upper levels where they belong together: a recommendation is a small
// tree, not a list of words. The name a tag is scored by is its whole path.
//
// existing are the paths an application has already, the tree to extend: the
// model is asked to start from one of them, written exactly as it is, where the
// text fits it, to add a level below it only when the text is more specific,
// and to start a new top-level category only when none fits. The first ones up
// to MaxTreeBytes in all are shown to the model; with none, it writes a tree of
// its own.
func WithTreeTags(existing ...string) GeneratorOption {
	return func(o *generatorOptions) {
		o.tree = true
		o.existing = slices.Clone(existing)
	}
}

const (
	// MaxTreeLevels is how deep a tree tag may go; a deeper path the model
	// writes is cut to it.
	MaxTreeLevels = 4
	// MaxTreeBytes caps the existing paths WithTreeTags shows the model, so
	// they, the text and the reply fit the model's context.
	MaxTreeBytes = 1500
)

// NewGenerator returns a Generator that has llm write the tags and emb score
// them. Without options, it writes tags of one to three words in the language
// of the text.
func NewGenerator(llm langai.Completer, emb langai.Embedder, cfg GeneratorConfig, opts ...GeneratorOption) (*Generator, error) {
	if llm == nil {
		return nil, errors.New("tagging: nil Completer")
	}
	if emb == nil {
		return nil, errors.New("tagging: nil Embedder")
	}
	if cfg.Candidates < 0 || cfg.MaxBytes < 0 || cfg.ChunkBytes < 0 {
		return nil, fmt.Errorf("tagging: Candidates, MaxBytes and ChunkBytes must not be negative, got %d, %d and %d", cfg.Candidates, cfg.MaxBytes, cfg.ChunkBytes)
	}
	cfg.Candidates = cmp.Or(cfg.Candidates, DefaultCandidates)
	cfg.MaxBytes = cmp.Or(cfg.MaxBytes, DefaultMaxBytes)
	cfg.ChunkBytes = cmp.Or(cfg.ChunkBytes, DefaultChunkBytes)
	var o generatorOptions
	for _, opt := range opts {
		opt(&o)
		if o.err != nil {
			return nil, o.err
		}
	}
	return &Generator{llm: llm, emb: emb, cfg: cfg, opts: o, prompt: systemPrompt(cfg.Candidates, o)}, nil
}

// systemPrompt sets the task. The text itself is the user message, so a text
// that talks to the model is still only the thing being tagged.
func systemPrompt(candidates int, o generatorOptions) string {
	var b strings.Builder
	fmt.Fprintf(&b, "You write tags for texts. Reply with up to %d tags for the text you are given: "+
		"one tag per line, the most relevant first, and nothing else (no numbers, no bullets, no explanations). ", candidates)
	// first and last: a small model copies the text's own words unless the
	// language is the rule it reads before anything else, and the long tree
	// prompt, its example path and the existing tree pull it into English.
	// The flat prompt in the text's language states the rule last only:
	// stated first as well, side recall on the 24 long documents fell from
	// 0.708 to 0.640 and precision from 0.882 to 0.853.
	switch {
	case o.language != "":
		fmt.Fprintf(&b, "Every tag must be in %[1]s. The text may be in another language: translate its topics into %[1]s "+
			"and do not copy its words, except proper names such as places, people, products and dishes. ", languageName(o.language))
	case o.tree:
		b.WriteString("Every tag must be in the language of the text, even though these instructions and their examples are in English. ")
	}
	if !o.tree {
		b.WriteString("A tag is one to three words naming a topic, a thing or a kind of content the text is about. ")
	} else {
		fmt.Fprintf(&b, "Write every tag as a path from a broad category down to a topic, a thing or a kind of content the text is about, "+
			"its levels separated by \" / \", two or three levels and never more than %d, each level one to three words, "+
			"such as \"cooking / baking / sourdough\". Tags that belong together share their upper levels, so that together they form a small tree. ", MaxTreeLevels)
		if paths := treeShown(o.existing); len(paths) > 0 {
			b.WriteString("These tags exist already, one per line:\n")
			for _, p := range paths {
				b.WriteString(p)
				b.WriteByte('\n')
			}
			b.WriteString("Where the text fits one of them, start from its path, written exactly as above, and add a level below it only when the text is more specific. " +
				"Start a new top-level category only when none of them fits. ")
		}
	}
	if o.language == "" {
		if !o.tree {
			b.WriteString("Write every tag in the language of the text.")
		} else {
			b.WriteString("Existing paths keep their words as they are, and the levels you add are in the language of the text. " +
				"Remember: write every tag in the language of the text.")
		}
	} else {
		if o.tree {
			b.WriteString("Existing paths keep their words as they are. ")
		}
		fmt.Fprintf(&b, "Remember: write every tag in %s, whatever the language of the text.", languageName(o.language))
	}
	return b.String()
}

// treeShown returns the existing paths the model is shown: the non-blank ones,
// trimmed, in order, up to MaxTreeBytes in all.
func treeShown(existing []string) []string {
	var out []string
	size := 0
	for _, p := range existing {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if size += len(p) + 1; size > MaxTreeBytes {
			break
		}
		out = append(out, p)
	}
	return out
}

// languageName names a BCP 47 tag in English for the prompt ("es" is
// "Spanish (es)"), or gives the tag alone when it has no English name.
func languageName(code string) string {
	tag := language.Make(code)
	if name := display.English.Tags().Name(tag); name != "" {
		return fmt.Sprintf("%s (%s)", name, code)
	}
	return "the language " + code
}

// Generate returns the tags for text that score at least minScore, best score
// first and equal scores in the order the model wrote them, or nil when none
// does. A minScore of -1 returns every tag the model wrote, and a NaN one
// returns none; DefaultGeneratorMinScore fits the default models. Only the
// first GeneratorConfig.MaxBytes bytes of text are read, and a blank text asks
// nobody. A model that writes no usable tag is not an error.
//
// A text of at most GeneratorConfig.ChunkBytes is written about in one model
// call and every tag is scored against the whole of it. A longer one is read
// in windows of about 6 KB, one model call each, so that topics of later
// sections get tags too: the names the windows propose are merged (names that
// differ only in case are one, the first spelling wins), each is scored by its
// best cosine to a chunk of ChunkBytes, and at most GeneratorConfig.Candidates
// are returned. One call per window, not one smaller window per chunk, is what
// measured best: a model that reads 6 KB names the topics of a section, while
// one that reads 2 KB writes near-duplicates and costs nearly three times the
// calls. A first line of at most 200 bytes, with more text after it, is read
// as the title and repeated at the head of every window after the first, so
// pass a long text as "title\n" + body, as the essences are built.
//
// Every model call writes its tags afresh, which takes seconds on a CPU, so a
// long text costs that once per window; the result is deterministic only as
// far as the Completer is.
func (g *Generator) Generate(ctx context.Context, text string, minScore float64) ([]Tag, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	text = strings.TrimSpace(clip(strings.ToValidUTF8(text, "�"), g.cfg.MaxBytes))
	if text == "" {
		return nil, nil
	}
	if len(text) > g.cfg.ChunkBytes {
		return g.generateLong(ctx, text, minScore)
	}
	answer, err := g.llm.Complete(ctx, g.prompt, text)
	if err != nil {
		return nil, fmt.Errorf("tagging: write tags: %w", err)
	}
	names := parseTags(answer, g.cfg.Candidates, g.clean())
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

// clean returns how a name the model wrote is cleaned: as a tree path with
// WithTreeTags, as a tag otherwise.
func (g *Generator) clean() func(string) string {
	if g.opts.tree {
		return cleanPath
	}
	return cleanTag
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

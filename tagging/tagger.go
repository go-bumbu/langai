package tagging

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"time"

	"github.com/go-bumbu/langai"
)

// Defaults for the zero values of TaggerConfig. The two minimums are the
// default yzma model's values on the eval golden set (see TestGoldenTagging),
// measured with the one-tag-per-path rule; that model's scores sit in a
// narrow band, so another model needs them measured again.
const (
	DefaultNameMinScore   = 0.42
	DefaultMemberMinScore = 0.58
	DefaultSmoothingK     = 3
)

// DefaultRebuildGrace is TaggerConfig.RebuildGrace when it is zero.
const DefaultRebuildGrace = 30 * time.Second

// scoringRule names how tag vectors are computed, for the params
// fingerprint: tags a Tagger built with another rule are stale and rebuilt
// (from the embeddings the cache keeps; nothing is embedded again). "mean"
// blends the mean of the members' vectors (see memberMean); the rule before
// it, the cosine to their centroid, had no name in the fingerprint.
const scoringRule = "mean"

// longRule names how a member of several pieces enters its tag's vector, for
// the params fingerprint of a Tagger with ChunkBytes: "best" by its piece
// closest to the tag's name (see memberVector); the rule before it, the mean
// of its pieces, had no name in the fingerprint.
const longRule = "best"

// TaggerConfig tunes a Tagger. Model is the identity of the embedding model,
// and NewTagger requires it: every embedding is cached under the model it came
// from, so a Tagger of another model embeds the texts it needs again rather
// than mix two models' vectors. NameMinScore and MemberMinScore are the
// minimum cosine scores against a tag's name and, on average, against its
// members. The two score on different scales, so each has its own minimum,
// mixed per tag by the blend weight (see tagCutoff); 0 defaults them to
// DefaultNameMinScore and DefaultMemberMinScore, and NewTagger refuses one that is
// not a finite number. SmoothingK is the Bayesian smoothing constant in blend:
// a tag's members weigh as much as its name at SmoothingK members, and
// outweigh it beyond that (<=0 defaults to DefaultSmoothingK; NewTagger refuses NaN
// and +Inf). There is no limit on how many tags a text gets.
//
// ChunkBytes, when above zero, has a text longer than that embedded in
// pieces of at most ChunkBytes bytes (see Chunk). An embedder reads a window
// of tokens and silently drops the rest (512 tokens, some 2 KB of prose, for
// the default yzma model), so without it only the start of a long text
// counts. Set it a little under what the embedder reads: 2000 for the default
// model. A text of two pieces scores the best of its pieces and their mean, a
// longer one the mean of its two best pieces (see itemScore), and a member of
// several pieces teaches its tag with the one closest to the tag's name (see
// memberVector). Zero, the default, embeds every text whole, as the golden
// numbers were measured. Each piece is cached on its own under Model, so a
// change of ChunkBytes embeds only pieces the cache does not hold. NewTagger
// refuses a negative one.
//
// Tag names are read as paths of a tree, levels separated by "/" ("dev / go"
// is a child of "dev"). Of the tags a text reaches the cutoff of on one path,
// Suggest returns only the one that clears its cutoff by most: a path implies
// its ancestors, so "dev" next to "dev / go" says nothing more, and of a
// parent and a child, the one cleared by more is what the text is about.
// Tags on different paths, siblings among them, are judged on their own.
// AllPathLevels turns that off: every tag that reaches its cutoff is
// returned, ancestors and descendants together.
//
// Taggers that share a cache may use different models: the model is part of every key.
//
// RebuildGrace is how long a rebuild goes on once the last caller waiting for
// it has gone, its context ended, before the rebuild is cancelled. A caller
// that asks about the space within it joins the rebuild instead of starting it
// over, so a space whose rebuild takes longer than any request waits is still
// built: by the requests that gave up on it, for the ones that come after.
// Zero means DefaultRebuildGrace, 30 seconds; a negative value cancels a
// rebuild as soon as its last caller has gone, which is right only when every
// caller waits longer than a rebuild takes. A rebuild that hangs is joined for
// as long as callers keep coming within the period, each waiting until its own
// context ends, so give an embedder that can hang a timeout of its own.
// Tagger.Close never waits for the period. It is no scoring setting.
//
// Logger receives, at Warn, what the Tagger worked around instead of failing
// a call: a member text it left out of its tag because the embedder failed on
// it or answered with a vector that is no use (with the index, the partition,
// the tag's id, and the SHA-256 and length of the text, which is not logged
// itself), and a cache it could read but not write. Nil discards them; the
// tags built without a member are listed by Scope.Incomplete all the same. It
// is no scoring setting: Taggers that differ only in it share a TagIndex
// freely.
type TaggerConfig struct {
	Model          string
	NameMinScore   float64
	MemberMinScore float64
	SmoothingK     float64
	ChunkBytes     int
	AllPathLevels  bool
	RebuildGrace   time.Duration
	Logger         *slog.Logger
}

// Tagger suggests tags for texts in every space of one application, choosing
// among the tags, and learning from the hand-made links, its Catalog reports.
// It keeps no copy of them: the embeddings it makes go to its EmbeddingCache,
// and each space's tag vectors to its TagIndex, derived data it can always
// rebuild. It is safe for concurrent use when its Embedder, Catalog, cache and
// index are.
//
// A rebuild runs on a goroutine of the Tagger's, shared by the callers that
// wait for it. When the last of them has gone it goes on for
// TaggerConfig.RebuildGrace, for a caller that comes back to join, and is
// cancelled after that. A panic of the Catalog or the Embedder in it becomes
// the error of those callers (wrapping ErrNotRebuilt and holding the value
// panicked with) instead of ending the process, and Close stops it at once.
type Tagger struct {
	e *engine
}

// Close stops the Tagger: every later call on it, or on one of its Scopes,
// returns an error wrapping langai.ErrClosed. The calls and rebuilds in
// flight have their contexts cancelled, and Close returns when they have
// ended, so nothing of the Tagger runs after it; their callers get an error
// wrapping langai.ErrClosed too. An Embedder or a Catalog that does not
// return when its context ends holds Close up for as long. It does not close
// the Embedder, the cache or the index, which the application made. Closing
// twice is fine, and the error is always nil.
func (t *Tagger) Close() error {
	t.e.close()
	return nil
}

// taggerOptions are what TaggerOptions set.
type taggerOptions struct {
	ix TagIndex
}

// TaggerOption changes a default of NewTagger.
type TaggerOption func(*taggerOptions)

// WithTagIndex has the Tagger keep each space's tag vectors in ix instead of
// a NewMemIndex(DefaultMemIndexBytes) of its own: one with another budget, or
// one shared by several Taggers of one configuration. A nil ix keeps the
// default.
func WithTagIndex(ix TagIndex) TaggerOption {
	return func(o *taggerOptions) { o.ix = ix }
}

// NewTagger returns a Tagger that embeds text with emb, reads the tags of a
// space and their members from cat when the space's revision changes, and
// caches every embedding it makes in cache. It fails without an embedder, a
// catalog, a cache or a Model; the zero values of cfg's other fields take the
// defaults TaggerConfig documents. It also fails, once those defaults are
// applied, on a scoring setting that is not a finite number: a minimum score
// of -Inf would let the tags it weighs in match every text, and one of NaN or
// +Inf would let them match none; a SmoothingK of +Inf would leave every
// member out of its tag's vector, and one of NaN would let no tag with members
// match.
func NewTagger(emb langai.Embedder, cat Catalog, cache EmbeddingCache, cfg TaggerConfig, opts ...TaggerOption) (*Tagger, error) {
	var o taggerOptions
	for _, opt := range opts {
		opt(&o)
	}
	if o.ix == nil {
		o.ix = NewMemIndex(DefaultMemIndexBytes)
	}
	e, err := newEngine(emb, cat, cache, o.ix, cfg)
	if err != nil {
		return nil, err
	}
	return &Tagger{e: e}, nil
}

// checkConfig returns cfg with the defaults TaggerConfig documents applied,
// and the fingerprint of its scoring settings: tags built with other ones are
// stale. It fails without a Model, on a negative ChunkBytes, and on a scoring
// setting that is not a finite number once the defaults are applied (see
// NewTagger).
func checkConfig(cfg TaggerConfig) (TaggerConfig, string, error) {
	switch {
	case cfg.Model == "":
		return cfg, "", errors.New("tagging: no TaggerConfig.Model")
	case cfg.ChunkBytes < 0:
		return cfg, "", fmt.Errorf("tagging: TaggerConfig.ChunkBytes is %d, want 0 or more", cfg.ChunkBytes)
	}
	if cfg.NameMinScore == 0 {
		cfg.NameMinScore = DefaultNameMinScore
	}
	if cfg.MemberMinScore == 0 {
		cfg.MemberMinScore = DefaultMemberMinScore
	}
	if cfg.SmoothingK <= 0 {
		cfg.SmoothingK = DefaultSmoothingK
	}
	if cfg.RebuildGrace == 0 {
		cfg.RebuildGrace = DefaultRebuildGrace
	}
	for _, s := range []struct {
		name  string
		value float64
	}{{"NameMinScore", cfg.NameMinScore}, {"MemberMinScore", cfg.MemberMinScore}, {"SmoothingK", cfg.SmoothingK}} {
		if math.IsNaN(s.value) || math.IsInf(s.value, 0) {
			return cfg, "", fmt.Errorf("tagging: TaggerConfig.%s is %v, want a finite number", s.name, s.value)
		}
	}
	params := fmt.Sprintf("%s k=%g name=%g member=%g", scoringRule, cfg.SmoothingK, cfg.NameMinScore, cfg.MemberMinScore)
	if cfg.ChunkBytes > 0 {
		// a tag's vectors blend its members', which chunking changes, and so
		// does how a member of several pieces enters (see longRule); without
		// it the fingerprint stays what it was before ChunkBytes existed
		params += fmt.Sprintf(" chunk=%d long=%s", cfg.ChunkBytes, longRule)
	}
	return cfg, params, nil
}

// ErrNotRebuilt is wrapped, together with its cause, by the error of a
// Suggest, Scores or Warm that could not bring its space up to the revision it
// was given: the Catalog failed during the rebuild, the embedder failed on a
// tag's name, which the error then names by the tag's id, or the embedder is
// down. The call returns nothing. The index is left as it was and the
// revision unrecorded, so the next call diffs again; texts are embedded and
// cached in batches, so what was embedded before the failure stays cached and
// the next call embeds only what is missing. Test for it with errors.Is.
//
// The embedder failing on one member's text while it takes others is not such
// a failure: the member is left out of its tag, which is built from its name
// and its other members, and reported to TaggerConfig.Logger; the tag is
// listed by Scope.Incomplete and built again at the space's next revision,
// when the member gets another try. A member is only left out on evidence
// that the embedder itself works: it answered with a vector that is no use,
// or an Embed after the one that failed worked, of a later text or, when
// there is none, of the tag's name embedded once more as a canary. An
// embedder that fails on that too is down, and so is one whose error wraps
// langai.ErrClosed, context.Canceled or context.DeadlineExceeded, which no
// text is to blame for: the rebuild fails with the embedder's error, after a
// handful of Embed calls rather than one per text, and records nothing.
var ErrNotRebuilt = errors.New("tagging: tag not rebuilt")

// Scope is a handle on one space of tags: its index is the hard isolation
// boundary (a tenant, say) and its partition a further coordinate within it
// (a space of that tenant). Scopes never see each other's tags, and no
// embedding is read on behalf of another index.
type Scope struct {
	tg               *Tagger
	index, partition string
}

// Scope returns the handle on the space at index and partition. It holds no
// data, so it is cheap to make. Every call through a scope with an empty index
// fails.
func (t *Tagger) Scope(index, partition string) *Scope {
	return &Scope{tg: t, index: index, partition: partition}
}

// Suggest returns the ids of the tags whose score for text reaches that tag's
// own cutoff (see tagCutoff), best score first and equal scores by id, with no
// limit on how many, or nil when none does: the picked ids of Scores, in the
// same order. A suggestion is never recorded: only the application's
// hand-made links teach a tag.
//
// revision is the space's revision, which the application changes whenever a
// tag, a link or a tagged item of the space does (see Catalog). When the
// index was not built at it, Suggest first reads the catalog's tags and
// rebuilds the ones that changed, once for all the callers that ask at that
// revision together; a caller at another revision waits for that rebuild and
// then looks again, so the rebuilds of one space never overlap. text is the
// document's own text: it is embedded as a
// query, unless the cache holds it already, and cached. An empty index or
// revision, or a blank text, is an error. Any error returns no ids; one of
// the rebuild wraps ErrNotRebuilt.
func (s *Scope) Suggest(ctx context.Context, revision, text string) ([]string, error) {
	ss, err := s.Scores(ctx, revision, text)
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, sc := range ss {
		if sc.Picked() {
			ids = append(ids, sc.ID)
		}
	}
	return ids, nil
}

// TagScore is how a text scores against one tag: the cosine between the
// text's vector and the tag's blended one (for a text in several pieces, see
// TaggerConfig.ChunkBytes, the score of its pieces), and the cutoff the tag
// picks at.
// ShadowedBy, when not empty, is the id of a tag on the same path that the
// text clears the cutoff of by more: the score reaches this tag's cutoff, but
// Suggest returns that tag instead (see TaggerConfig).
type TagScore struct {
	ID         string
	Score      float64
	Cutoff     float64
	ShadowedBy string
}

// Picked reports whether Suggest returns the tag: the score reaches the cutoff
// and no tag on its path shadows it. A NaN score or cutoff picks nothing.
func (t TagScore) Picked() bool { return t.Score >= t.Cutoff && t.ShadowedBy == "" }

// Scores returns text's score against every tag of the space, the ones
// Suggest leaves out included, best score first and equal scores by id: what
// Suggest decides from, for a page that shows why a tag was or was not
// picked. It brings the space up to revision and handles errors as Suggest
// does.
func (s *Scope) Scores(ctx context.Context, revision, text string) ([]TagScore, error) {
	var scores []TagScore
	err := s.tg.e.call(ctx, func(ctx context.Context) (err error) {
		scores, err = s.tg.e.scores(ctx, s.index, s.partition, revision, text)
		return err
	})
	return scores, err
}

// Warm brings the space up to revision, as the next Suggest at it would, and
// embeds and caches each of texts the cache does not hold yet, the blank ones
// skipped, so no Suggest pays for them: an item's text when it is saved, say,
// or the whole corpus after a switch of models. An empty index or revision is
// an error, and one of the rebuild wraps ErrNotRebuilt.
//
// The texts are embedded and cached a batch at a time, so a Warm that fails or
// is cancelled keeps what it had done and the next one embeds the rest. A text
// the embedder fails on while it takes others does not stop them: Warm embeds
// them all the same and then returns an error that names each text it could
// not embed by its place among texts ("text 3 of 500") and wraps the cause.
// An embedder that is down (it is closed, or it fails on every text it is
// given, one it took a moment ago included) ends the Warm with its error
// after a handful of calls, not one per text.
func (s *Scope) Warm(ctx context.Context, revision string, texts ...string) error {
	return s.tg.e.call(ctx, func(ctx context.Context) error {
		return s.tg.e.warm(ctx, s.index, s.partition, revision, texts)
	})
}

// Incomplete returns the ids, sorted, of the space's tags that were last built
// without one or more of their members, because the embedder could not take
// the member's text (see ErrNotRebuilt): each is built from its name and its
// other members, and suggests on that until the space's next revision, when
// the members left out get another try. It is how an application without a
// TaggerConfig.Logger sees that a member was skipped: check it after a Warm,
// say, and change the space's revision to try again sooner. It reads the
// TagIndex only, so it is nil for a space that was never built, or that the
// index evicted. An empty index is an error.
func (s *Scope) Incomplete(ctx context.Context) ([]string, error) {
	var ids []string
	err := s.tg.e.call(ctx, func(ctx context.Context) (err error) {
		ids, err = s.tg.e.incomplete(ctx, s.index, s.partition)
		return err
	})
	return ids, err
}

// Touch marks every embedding the space's tags are built from as used now:
// the name of each tag its Catalog lists and the texts of their members, under
// the Tagger's Model. It embeds nothing. Tagger.Prune does this for the spaces
// it is given; Touch on its own is for the spaces of another Tagger that
// shares the cache (one of another Model, say), which that Tagger's Prune
// would otherwise take for unused. An empty index is an error.
func (s *Scope) Touch(ctx context.Context) error {
	return s.tg.e.call(ctx, func(ctx context.Context) error {
		return s.tg.e.touch(ctx, s.index, s.partition)
	})
}

// Prune removes from the cache, across every index, the embeddings last used
// before unusedSince, and returns how many: the texts of items edited or
// deleted since, and the rows of a model no longer used. It is the one call
// for that, on a schedule, instead of the cache's own Prune.
//
// A tag that does not change is built once and never read from the cache
// again, so the embeddings of its name and members look unused however often
// the tag is suggested. Prune therefore first reads the Catalog of each of
// spaces and marks what its tags are built from as used now (see Scope.Touch),
// and only then removes the rest: pass every space still in use. A space left
// out loses the embeddings not read since unusedSince, which costs embedding
// them again at its next rebuild, never correctness. When a space cannot be
// read, nothing is removed.
//
// unusedSince must be at least CacheRefreshAfter (24 hours) in the past: a
// read rewrites an entry's last-used time only once it is that stale, so a
// more recent cutoff would remove entries read minutes ago, and is refused.
func (t *Tagger) Prune(ctx context.Context, unusedSince time.Time, spaces ...Space) (int64, error) {
	var removed int64
	err := t.e.call(ctx, func(ctx context.Context) (err error) {
		removed, err = t.e.prune(ctx, unusedSince, spaces)
		return err
	})
	return removed, err
}

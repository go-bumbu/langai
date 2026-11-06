package tagging

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

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

// scoringRule names how tag vectors are computed, for the params
// fingerprint: tags a Tagger built with another rule are stale and rebuilt
// (from the vectors the store keeps; nothing is embedded again). "mean" blends
// the mean of the members' vectors (see memberMean); the rule before it, the
// cosine to their centroid, had no name in the fingerprint.
const scoringRule = "mean"

// TaggerConfig tunes a Tagger. Model is the identity of the embedding model,
// and NewTagger requires it: every vector a Store holds is kept with the model it came
// from, so a Tagger of another model re-embeds the stored texts it needs
// rather than mix two models' vectors. NameMinScore and MemberMinScore are the
// minimum cosine scores against a tag's name and, on average, against its
// members. The two score on different scales, so each has its own minimum,
// mixed per tag by the blend weight (see tagCutoff); 0 defaults them to
// DefaultNameMinScore and DefaultMemberMinScore, and NewTagger refuses one that is
// not a finite number. SmoothingK is the Bayesian smoothing constant in blend:
// a tag's members weigh as much as its name at SmoothingK members, and
// outweigh it beyond that (<=0 defaults to DefaultSmoothingK; NewTagger refuses NaN
// and +Inf). There is no limit on how many tags an item gets.
//
// ChunkBytes, when above zero, has an item longer than that embedded in
// pieces of at most ChunkBytes bytes (see Chunk), its vector the mean of
// theirs. An embedder reads a window of tokens and silently drops the rest
// (512 tokens, some 2 KB of prose, for the default yzma model), so without it
// only the start of a long item counts. Set it a little under what the
// embedder reads. Zero, the default, embeds every item whole, as the golden
// numbers were measured. Items are stored under the model marked with the
// chunk size, so a change of ChunkBytes re-embeds them as a change of Model
// does. NewTagger refuses a negative one.
//
// Tag names are read as paths of a tree, levels separated by "/" ("dev / go"
// is a child of "dev"). Of the tags an item reaches the cutoff of on one path,
// Suggest returns only the one that clears its cutoff by most: a path implies
// its ancestors, so "dev" next to "dev / go" says nothing more, and of a
// parent and a child, the one cleared by more is what the item is about.
// Tags on different paths, siblings among them, are judged on their own.
// AllPathLevels turns that off: every tag that reaches its cutoff is
// returned, ancestors and descendants together.
//
// Every Tagger that shares a Store must use one Model and one set of scoring
// settings. A Tagger takes a tag built under another model or with other
// settings for stale, and rebuilds it, re-embedding its members under its own
// model when the models differ, so two Taggers that differ rebuild each
// other's tags at every Suggest and never settle. To switch models in place,
// stop every instance together, start them with the new Model, and call
// Rebuild on every scope. To switch without stopping, open a new generation
// of the store (WithGeneration, in sqlitestore and pgstore): its tables start
// empty, so replay every tag, item and hand-made link into it, and write to
// both stores until the cutover.
type TaggerConfig struct {
	Model          string
	NameMinScore   float64
	MemberMinScore float64
	SmoothingK     float64
	ChunkBytes     int
	AllPathLevels  bool
}

// Tagger suggests tags for the items of every scope of one Store. It keeps no
// state beyond its embedder, its store and its configuration, so it is safe
// for concurrent use when its Store and its Embedder are.
type Tagger struct {
	emb   langai.Embedder
	store Store
	cfg   TaggerConfig
	// params fingerprints cfg's scoring settings, for TagVectors.Params: a tag
	// built with others is stale (see stale)
	params string
	// itemModel is what item vectors are stored under (see itemModel)
	itemModel string
}

// NewTagger returns a Tagger that embeds text with emb and keeps every scope's
// tags, items and hand-made links in store. It fails without an embedder, a
// store or a Model; the zero values of cfg's other fields take the defaults
// TaggerConfig documents. It also fails, once those defaults are applied, on a
// scoring setting that is not a finite number: a minimum score of -Inf would
// let the tags it weighs in match every item, and one of NaN or +Inf would let
// them match none; a SmoothingK of +Inf would leave every member out of its
// tag's vector, and one of NaN would let no tag with members match.
func NewTagger(emb langai.Embedder, store Store, cfg TaggerConfig) (*Tagger, error) {
	switch {
	case emb == nil:
		return nil, errors.New("tagging: no embedder")
	case store == nil:
		return nil, errors.New("tagging: no store")
	}
	cfg, params, err := checkConfig(cfg)
	if err != nil {
		return nil, err
	}
	return &Tagger{emb: emb, store: store, cfg: cfg, params: params, itemModel: itemModel(cfg)}, nil
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
		// a tag's vectors blend its members', which chunking changes; without
		// it the fingerprint stays what stores built before ChunkBytes hold
		params += fmt.Sprintf(" chunk=%d", cfg.ChunkBytes)
	}
	return cfg, params, nil
}

// ErrNotRebuilt is wrapped, together with its cause, by the error of a write
// that stored its change but could not then rebuild a tag the change affects:
// the change is stored; the tag is rebuilt by a later Suggest or Rebuild;
// retrying the call is a safe no-op. Suggest and Rebuild wrap it too, for the
// tags they could not rebuild, and name each one. Test for it with errors.Is.
var ErrNotRebuilt = errors.New("tagging: tag not rebuilt")

// Scope is a handle on one set of tags and items: its index is the store's
// hard isolation boundary (a tenant, say) and its partition a further
// coordinate within it (a space of that tenant). Scopes never see each
// other's tags, items or links.
//
// A write rebuilds the tags it changes before it returns. If a rebuild fails,
// the write returns an error that wraps ErrNotRebuilt and the cause, naming
// the tag: the change is stored; the tag is rebuilt by a later Suggest or
// Rebuild; retrying the call is a safe no-op.
type Scope struct {
	tg               *Tagger
	index, partition string
}

// Scope returns the handle on the scope at index and partition. It holds no
// data, so it is cheap to make. Every call through a scope with an empty index
// fails, as the Store refuses one.
func (t *Tagger) Scope(index, partition string) *Scope {
	return &Scope{tg: t, index: index, partition: partition}
}

// SetTag creates the tag, or renames it, and rebuilds it. The name is
// embedded as the tag's prior: a full path such as "colors / warm / red"
// carries more than the leaf alone. Setting the name the tag has changes
// nothing. An empty id or a blank name is an error. When the error wraps
// ErrNotRebuilt, the change is stored; the tag is rebuilt by a later Suggest
// or Rebuild; retrying the call is a safe no-op.
func (s *Scope) SetTag(ctx context.Context, tagID, name string) error {
	if tagID == "" {
		return errors.New("tagging: empty tag id")
	}
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("tagging: blank name for tag %q", tagID)
	}
	changed, err := s.tg.store.PutTag(ctx, s.index, s.partition, tagID, name)
	if err != nil {
		return err
	}
	if !changed {
		return nil
	}
	return s.rebuild(ctx, tagID)
}

// DeleteTag deletes the tag and its hand-made links. A tag created later
// under the same id starts without them. An unknown tag is no error.
func (s *Scope) DeleteTag(ctx context.Context, tagID string) error {
	return s.tg.store.DeleteTag(ctx, s.index, s.partition, tagID)
}

// SetItem creates the item, or replaces its text, and rebuilds the tags it is
// tagged with by hand, in id order, stopping at the first that fails. The text
// is embedded unless it is the item's current text, embedded under the
// Tagger's model already. An empty id or blank text is an error. When the
// error wraps ErrNotRebuilt, the change is stored; the tag is rebuilt by a
// later Suggest or Rebuild, and so are the ones after it; retrying the call is
// a safe no-op.
func (s *Scope) SetItem(ctx context.Context, itemID, text string) error {
	if itemID == "" {
		return errors.New("tagging: empty item id")
	}
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("tagging: blank text for item %q", itemID)
	}
	item, ok, err := s.tg.store.Item(ctx, s.index, s.partition, itemID)
	if err != nil {
		return err
	}
	if ok && item.Text == text && item.Model == s.tg.itemModel {
		return nil
	}
	linked, err := s.putItem(ctx, itemID, text)
	if err != nil {
		return err
	}
	return s.rebuildAll(ctx, linked)
}

// DeleteItem deletes the item and its hand-made links, and rebuilds the tags
// it was tagged with, in id order, stopping at the first that fails. An
// unknown item is no error. When the error wraps ErrNotRebuilt, the change is
// stored; the tag is rebuilt by a later Suggest or Rebuild, and so are the
// ones after it; retrying the call is a safe no-op.
func (s *Scope) DeleteItem(ctx context.Context, itemID string) error {
	linked, err := s.tg.store.DeleteItem(ctx, s.index, s.partition, itemID)
	if err != nil {
		return err
	}
	return s.rebuildAll(ctx, linked)
}

// TagByHand records that a person tagged the item with the tag, which makes
// the item a member of the tag, one of the examples it learns from, and
// rebuilds the tag at once, from all of its members: a bulk import of n hand
// tags of one tag reads its members n times, about n²/2 rows in all. A link
// that exists already changes nothing. An unknown item is ErrUnknownItem, and
// an unknown tag ErrUnknownTag; an error of the store comes back naming both
// ids. When the error wraps ErrNotRebuilt, the change is stored; the tag is
// rebuilt by a later Suggest or Rebuild; retrying the call is a safe no-op.
func (s *Scope) TagByHand(ctx context.Context, itemID, tagID string) error {
	changed, err := s.tg.store.Link(ctx, s.index, s.partition, itemID, tagID)
	if err != nil {
		return fmt.Errorf("tagging: tag item %q with %q: %w", itemID, tagID, err)
	}
	if !changed {
		return nil
	}
	return s.rebuild(ctx, tagID)
}

// UntagByHand removes the hand-made link between the item and the tag, and
// rebuilds the tag. Without such a link it does nothing. An error of the store
// comes back naming both ids. When the error wraps ErrNotRebuilt, the change
// is stored; the tag is rebuilt by a later Suggest or Rebuild; retrying the
// call is a safe no-op.
func (s *Scope) UntagByHand(ctx context.Context, itemID, tagID string) error {
	changed, err := s.tg.store.Unlink(ctx, s.index, s.partition, itemID, tagID)
	if err != nil {
		return fmt.Errorf("tagging: untag item %q from %q: %w", itemID, tagID, err)
	}
	if !changed {
		return nil
	}
	return s.rebuild(ctx, tagID)
}

// Suggest returns the ids of the tags whose score for the item reaches that
// tag's own cutoff (see tagCutoff), best score first and equal scores by id,
// with no limit on how many, or nil when none does. A suggestion is never
// stored as a link: only TagByHand teaches a tag. Suggest first rebuilds every
// stale tag of the scope, and re-embeds the item's stored text when it was
// embedded under another model. An unknown item is ErrUnknownItem.
//
// A tag that cannot be rebuilt does not stop Suggest: it rebuilds the others,
// ranks every tag it can, and returns the ids with an error that wraps
// ErrNotRebuilt and names each tag that failed. When errors.Is(err,
// ErrNotRebuilt), ids is still valid: it leaves out, or uses the previous
// vectors of, the tags that could not be rebuilt. Any other error returns no
// ids.
func (s *Scope) Suggest(ctx context.Context, itemID string) ([]string, error) {
	vec, tags, err := s.scoring(ctx, itemID)
	if err != nil && !errors.Is(err, ErrNotRebuilt) {
		return nil, err
	}
	return rank(vec, tags, s.tg.cfg.Model, !s.tg.cfg.AllPathLevels), err
}

// TagScore is how an item scores against one tag: the cosine between the
// item's vector and the tag's blended one, and the cutoff the tag picks at.
// ShadowedBy, when not empty, is the id of a tag on the same path that the
// item clears the cutoff of by more: the score reaches this tag's cutoff, but
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

// Scores returns the item's score against every tag of the scope, the ones
// Suggest leaves out included, best score first and equal scores by id: what
// Suggest decides from, for a page that shows why a tag was or was not
// picked. It rebuilds stale tags and handles errors as Suggest does.
func (s *Scope) Scores(ctx context.Context, itemID string) ([]TagScore, error) {
	vec, tags, err := s.scoring(ctx, itemID)
	if err != nil && !errors.Is(err, ErrNotRebuilt) {
		return nil, err
	}
	return scores(vec, tags, s.tg.cfg.Model, !s.tg.cfg.AllPathLevels), err
}

// scoring returns what an item is scored with: its vector (see itemVector)
// and the scope's tags, stale ones rebuilt first (see rebuildStale). An error
// that wraps ErrNotRebuilt comes with both; any other comes with neither.
func (s *Scope) scoring(ctx context.Context, itemID string) ([]float32, []TagRecord, error) {
	item, ok, err := s.tg.store.Item(ctx, s.index, s.partition, itemID)
	if err != nil {
		return nil, nil, err
	}
	if !ok {
		return nil, nil, unknownItem(itemID)
	}
	vec, err := s.itemVector(ctx, item)
	if err != nil {
		return nil, nil, err
	}
	tags, err := s.rebuildStale(ctx)
	if err != nil && !errors.Is(err, ErrNotRebuilt) {
		return nil, nil, err
	}
	return vec, tags, err
}

// Rebuild rebuilds every stale tag of the scope now, rather than at the next
// Suggest: a tag whose rebuild failed or was overtaken, or one built under
// another model or with other scoring settings. It warms a scope up, after a
// switch of models say; an item that no tag is linked to is still re-embedded
// only at its next Suggest. A tag that cannot be rebuilt does not stop it: once
// it has rebuilt the others, it returns the error Suggest would, which wraps
// ErrNotRebuilt and names each tag that failed.
func (s *Scope) Rebuild(ctx context.Context) error {
	_, err := s.rebuildStale(ctx)
	return err
}

// embedOne embeds a single text, refusing an embedder that breaks its contract
// by answering with no vector, or several, rather than panicking on the
// missing one, and refusing a vector that is empty, or whose length is not the
// embedder's Dimensions when it says them, rather than storing it for a
// rebuild to blend with vectors of another length.
func embedOne(ctx context.Context, emb langai.Embedder, role langai.Role, text string) ([]float32, error) {
	vs, err := emb.Embed(ctx, role, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vs) != 1 {
		return nil, fmt.Errorf("tagging: embedder returned %d vectors for one %s", len(vs), role)
	}
	switch dims := emb.Dimensions(); {
	case dims > 0 && len(vs[0]) != dims:
		return nil, fmt.Errorf("tagging: embedder returned a vector of %d dimensions, want %d", len(vs[0]), dims)
	case len(vs[0]) == 0:
		return nil, errors.New("tagging: embedder returned an empty vector")
	}
	return vs[0], nil
}

// putItem embeds the item's text as a query (see embedItem), the way Suggest
// scores an item against the tag vectors, and stores the item under the
// Tagger's item model. It returns the ids of the tags the item is linked to,
// whose Versions the store bumped.
func (s *Scope) putItem(ctx context.Context, itemID, text string) (linked []string, err error) {
	vec, err := embedItem(ctx, s.tg.emb, s.tg.cfg.ChunkBytes, text)
	if err != nil {
		return nil, err
	}
	return s.tg.store.PutItem(ctx, s.index, s.partition,
		ItemRecord{ID: itemID, Text: text, Vec: vec, Model: s.tg.itemModel})
}

// reembedItem embeds the item's stored text again under the Tagger's model,
// and stores the vector only while the item still has that text (see
// Store.SetItemVector), so it never overwrites an edit that landed after the
// item was read. stored is false when the item is gone or its text changed;
// the store then bumped no tag, and its writer bumped them instead.
func (s *Scope) reembedItem(ctx context.Context, item ItemRecord) (vec []float32, stored bool, err error) {
	vec, err = embedItem(ctx, s.tg.emb, s.tg.cfg.ChunkBytes, item.Text)
	if err != nil {
		return nil, false, err
	}
	_, stored, err = s.tg.store.SetItemVector(ctx, s.index, s.partition, item.ID, item.Text, vec, s.tg.itemModel)
	if err != nil {
		return nil, false, err
	}
	return vec, stored, nil
}

// itemVector returns the vector Suggest scores the item with: the stored one
// when it is of the Tagger's model. Otherwise the item is re-embedded (see
// reembedItem), which leaves the tags it is linked to stale, for rebuildStale.
// If an edit landed meanwhile, the item is read again: its writer's vector
// serves when it is of the Tagger's model, and otherwise the new text is
// embedded for this call alone and not stored, so the edit stays as its writer
// stored it.
func (s *Scope) itemVector(ctx context.Context, item ItemRecord) ([]float32, error) {
	if item.Model == s.tg.itemModel {
		return item.Vec, nil
	}
	vec, stored, err := s.reembedItem(ctx, item)
	if err != nil {
		return nil, err
	}
	if stored {
		return vec, nil
	}
	edited, ok, err := s.tg.store.Item(ctx, s.index, s.partition, item.ID)
	switch {
	case err != nil:
		return nil, err
	case !ok:
		return nil, unknownItem(item.ID)
	case edited.Model == s.tg.itemModel:
		return edited.Vec, nil
	}
	return embedItem(ctx, s.tg.emb, s.tg.cfg.ChunkBytes, edited.Text)
}

// unknownItem is the error for an item the scope does not hold.
func unknownItem(itemID string) error {
	return fmt.Errorf("%w %q", ErrUnknownItem, itemID)
}

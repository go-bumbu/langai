// Package autotag suggests tags for items, such as bookmarks or notes, from
// their embeddings, and learns from the items a person tags by hand. It owns
// what it learns from: an app reports its tags and its items, by id and text,
// and the tags a person put on items by hand, and asks for suggestions.
//
// Each tag's vector blends an embedding of its name (a prior) with the
// centroid of the items tagged with it by hand, its members (see blend), so a
// brand-new tag with no members yet still tags by its name, and a tag that
// accumulates members gradually tags by example instead. An item gets every
// tag whose score reaches that tag's own cutoff (see tagCutoff), with no limit
// on how many.
//
// A Store keeps the tags, the items with their embeddings and the hand-made
// links of every scope. Every change rebuilds the vectors of the tags it
// affects right away. A tag's Version goes up with every change to what its
// vectors depend on, and a rebuild stores its result only while the tag is at
// the Version it read, so a rebuild never overwrites a change that landed
// meanwhile. A tag left stale that way, by a rebuild that failed, or by a
// switch to another embedding model is rebuilt by the next Suggest, or by
// Rebuild.
package autotag

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/go-bumbu/langai/semsearch"
)

// Defaults for the zero values of Config. The two minimums are the default
// yzma model's values on the semsearch golden set (see TestGoldenTagging);
// that model's scores sit in a narrow band, so another model needs them
// measured again.
const (
	DefaultNameMinScore   = 0.46
	DefaultMemberMinScore = 0.70
	DefaultSmoothingK     = 3
)

// Config tunes a Tagger. Model is the identity of the embedding model, and
// New requires it: every vector a Store holds is kept with the model it came
// from, so a Tagger of another model re-embeds the stored texts it needs
// rather than mix two models' vectors. NameMinScore and MemberMinScore are the
// minimum cosine scores against a tag's name and against the centroid of its
// members. The two score on different scales, so each has its own minimum,
// mixed per tag by the blend weight (see tagCutoff); 0 defaults them to
// DefaultNameMinScore and DefaultMemberMinScore. SmoothingK is the Bayesian
// smoothing constant in blend: a tag's members weigh as much as its name at
// SmoothingK members, and outweigh it beyond that (<=0 defaults to
// DefaultSmoothingK). A tag's vector and cutoff are computed when it is
// rebuilt, so other minimums or another SmoothingK reach a stored tag at its
// next change, while another Model reaches every tag at the next Suggest or
// Rebuild. There is no limit on how many tags an item gets.
type Config struct {
	Model          string
	NameMinScore   float64
	MemberMinScore float64
	SmoothingK     float64
}

// Tagger suggests tags for the items of every scope of one Store. It keeps no
// state beyond its embedder, its store and its configuration, so it is safe
// for concurrent use when its Store and its Embedder are.
type Tagger struct {
	emb   semsearch.Embedder
	store Store
	cfg   Config
}

// New returns a Tagger that embeds text with emb and keeps every scope's
// tags, items and hand-made links in store. It fails without an embedder, a
// store or a Model; the zero values of cfg's other fields take the defaults
// Config documents.
func New(emb semsearch.Embedder, store Store, cfg Config) (*Tagger, error) {
	switch {
	case emb == nil:
		return nil, errors.New("autotag: no embedder")
	case store == nil:
		return nil, errors.New("autotag: no store")
	case cfg.Model == "":
		return nil, errors.New("autotag: no Config.Model")
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
	return &Tagger{emb: emb, store: store, cfg: cfg}, nil
}

// Scope is a handle on one set of tags and items: its index is the store's
// hard isolation boundary (a tenant, say) and its partition a further
// coordinate within it (a space of that tenant). Scopes never see each
// other's tags, items or links.
//
// A write rebuilds the tags it changes before it returns. If that rebuild
// fails, the write stays and the error is returned, naming the tag; the next
// Suggest rebuilds it.
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
// nothing. An empty id or a blank name is an error.
func (s *Scope) SetTag(ctx context.Context, tagID, name string) error {
	if tagID == "" {
		return errors.New("autotag: empty tag id")
	}
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("autotag: blank name for tag %q", tagID)
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
// tagged with by hand. The text is embedded unless it is the item's current
// text, embedded under the Tagger's model already. An empty id or blank text
// is an error.
func (s *Scope) SetItem(ctx context.Context, itemID, text string) error {
	if itemID == "" {
		return errors.New("autotag: empty item id")
	}
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("autotag: blank text for item %q", itemID)
	}
	item, ok, err := s.tg.store.Item(ctx, s.index, s.partition, itemID)
	if err != nil {
		return err
	}
	if ok && item.Text == text && item.Model == s.tg.cfg.Model {
		return nil
	}
	_, linked, err := s.putItem(ctx, itemID, text)
	if err != nil {
		return err
	}
	return s.rebuildAll(ctx, linked)
}

// DeleteItem deletes the item and its hand-made links, and rebuilds the tags
// it was tagged with. An unknown item is no error.
func (s *Scope) DeleteItem(ctx context.Context, itemID string) error {
	linked, err := s.tg.store.DeleteItem(ctx, s.index, s.partition, itemID)
	if err != nil {
		return err
	}
	return s.rebuildAll(ctx, linked)
}

// TagByHand records that a person tagged the item with the tag, which makes
// the item a member of the tag, one of the examples it learns from, and
// rebuilds the tag. A link that exists already changes nothing. An unknown
// item is ErrUnknownItem, and an unknown tag ErrUnknownTag.
func (s *Scope) TagByHand(ctx context.Context, itemID, tagID string) error {
	changed, err := s.tg.store.Link(ctx, s.index, s.partition, itemID, tagID)
	if err != nil {
		return err
	}
	if !changed {
		return nil
	}
	return s.rebuild(ctx, tagID)
}

// UntagByHand removes the hand-made link between the item and the tag, and
// rebuilds the tag. Without such a link it does nothing.
func (s *Scope) UntagByHand(ctx context.Context, itemID, tagID string) error {
	changed, err := s.tg.store.Unlink(ctx, s.index, s.partition, itemID, tagID)
	if err != nil {
		return err
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
func (s *Scope) Suggest(ctx context.Context, itemID string) ([]string, error) {
	item, ok, err := s.tg.store.Item(ctx, s.index, s.partition, itemID)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrUnknownItem, itemID)
	}
	vec := item.Vec
	if item.Model != s.tg.cfg.Model {
		// the tags it is linked to go stale, and rebuildStale rebuilds them
		vec, _, err = s.putItem(ctx, itemID, item.Text)
		if err != nil {
			return nil, err
		}
	}
	tags, err := s.rebuildStale(ctx)
	if err != nil {
		return nil, err
	}
	return rank(vec, tags, s.tg.cfg.Model), nil
}

// Rebuild rebuilds every stale tag of the scope now, rather than at the next
// Suggest: a tag whose rebuild failed or was overtaken, or one built under
// another model. It warms a scope up, after a switch of models say; an item
// that no tag is linked to is still re-embedded only at its next Suggest.
func (s *Scope) Rebuild(ctx context.Context) error {
	_, err := s.rebuildStale(ctx)
	return err
}

// embedOne embeds a single text, refusing an embedder that breaks its contract
// by answering with no vector, or several, rather than panicking on the
// missing one.
func (t *Tagger) embedOne(ctx context.Context, role semsearch.Role, text string) ([]float32, error) {
	vs, err := t.emb.Embed(ctx, role, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vs) != 1 {
		return nil, fmt.Errorf("autotag: embedder returned %d vectors for one %s", len(vs), role)
	}
	return vs[0], nil
}

// putItem embeds the item's text as a query, the way Suggest scores an item
// against the tag vectors, and stores the item under the Tagger's model. It
// returns the vector and the ids of the tags the item is linked to, whose
// Versions the store bumped.
func (s *Scope) putItem(ctx context.Context, itemID, text string) (vec []float32, linked []string, err error) {
	vec, err = s.tg.embedOne(ctx, semsearch.RoleQuery, text)
	if err != nil {
		return nil, nil, err
	}
	linked, err = s.tg.store.PutItem(ctx, s.index, s.partition,
		ItemRecord{ID: itemID, Text: text, Vec: vec, Model: s.tg.cfg.Model})
	if err != nil {
		return nil, nil, err
	}
	return vec, linked, nil
}

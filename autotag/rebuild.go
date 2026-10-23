package autotag

import (
	"context"
	"fmt"
	"slices"

	"github.com/go-bumbu/langai/semsearch"
)

// rebuildPasses bounds the passes rebuildStale makes over a scope's tags.
const rebuildPasses = 3

// stale reports whether the tag needs a rebuild: something its vectors depend
// on changed since they were built (a tag never built has Built 0), or they
// were built under another model or with other scoring settings than the
// Tagger's.
func (t *Tagger) stale(tag TagRecord) bool {
	return tag.Built != tag.Version || tag.Model != t.cfg.Model || tag.Params != t.params
}

// rebuild computes the tag's vectors from its name and its members, with the
// Tagger's scoring settings, and stores them only while the tag is at the
// Version they were computed from. Members embedded under another model are
// re-embedded first (see refreshMembers), and the name is embedded, as a
// document, when the tag has no name vector of the Tagger's model; other
// scoring settings alone embed nothing. A tag deleted meanwhile is no error,
// and neither is one changed meanwhile: the change left it stale, for its
// writer or the next Suggest to rebuild. An error comes back wrapped, naming
// the tag.
func (s *Scope) rebuild(ctx context.Context, id string) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("autotag: rebuild tag %q: %w", id, err)
		}
	}()
	if err = s.refreshMembers(ctx, id); err != nil {
		return err
	}
	store, cfg := s.tg.store, s.tg.cfg
	rec, ok, err := store.Tag(ctx, s.index, s.partition, id)
	if err != nil {
		return err
	}
	if !ok {
		return nil // deleted meanwhile
	}
	// read after rec: a change in between bumps the tag's Version, and
	// SetTagVectors refuses vectors computed for an older one
	members, err := store.Members(ctx, s.index, s.partition, id)
	if err != nil {
		return err
	}
	nameVec := rec.NameVec
	if nameVec == nil || rec.Model != cfg.Model {
		if nameVec, err = s.tg.embedOne(ctx, semsearch.RoleDocument, rec.Name); err != nil {
			return err
		}
	}
	var vecs [][]float32
	for _, m := range members {
		if m.Model == cfg.Model {
			vecs = append(vecs, m.Vec)
		}
	}
	k := cfg.SmoothingK
	// false: the tag changed meanwhile, which is no error (see above)
	_, err = store.SetTagVectors(ctx, s.index, s.partition, id, rec.Version, TagVectors{
		NameVec: nameVec,
		Vec:     blend(nameVec, vecs, k),
		Cutoff:  storedCutoff(tagCutoff(nameVec, vecs, k, cfg.NameMinScore, cfg.MemberMinScore)),
		Model:   cfg.Model,
		Params:  s.tg.params,
	})
	return err
}

// refreshMembers re-embeds, from the text the store keeps, the tag's members
// whose vectors come from another model than the Tagger's, so that rebuild can
// blend them in (see reembedItem). Each item it re-embeds bumps the Version of
// every tag linked to it: rebuild reads this tag after that, and the others are
// left stale, for their own rebuilds.
func (s *Scope) refreshMembers(ctx context.Context, id string) error {
	members, err := s.tg.store.Members(ctx, s.index, s.partition, id)
	if err != nil {
		return err
	}
	for _, m := range members {
		if m.Model == s.tg.cfg.Model {
			continue
		}
		item, ok, err := s.tg.store.Item(ctx, s.index, s.partition, m.ItemID)
		if err != nil {
			return err
		}
		if !ok || item.Model == s.tg.cfg.Model {
			continue // deleted or re-embedded meanwhile
		}
		// not stored: the item was edited or deleted meanwhile by a writer that
		// bumped this tag, and rebuild reads what that writer left
		if _, _, err := s.reembedItem(ctx, item); err != nil {
			return err
		}
	}
	return nil
}

// rebuildAll rebuilds the tags in order and stops at the first error; the
// tags after it stay stale until the next Suggest.
func (s *Scope) rebuildAll(ctx context.Context, ids []string) error {
	for _, id := range ids {
		if err := s.rebuild(ctx, id); err != nil {
			return err
		}
	}
	return nil
}

// rebuildStale rebuilds the scope's stale tags and returns all its tags. A
// change that lands during a pass can leave a tag stale again, so it makes up
// to rebuildPasses passes and then returns the tags as they are: one still
// stale then is scored with the vectors it has if they are of the Tagger's
// model, and skipped if not (see rank).
func (s *Scope) rebuildStale(ctx context.Context) ([]TagRecord, error) {
	for range rebuildPasses {
		tags, err := s.tg.store.Tags(ctx, s.index, s.partition)
		if err != nil {
			return nil, err
		}
		if !slices.ContainsFunc(tags, s.tg.stale) {
			return tags, nil
		}
		for _, tag := range tags {
			if !s.tg.stale(tag) {
				continue
			}
			if err := s.rebuild(ctx, tag.ID); err != nil {
				return nil, err
			}
		}
	}
	return s.tg.store.Tags(ctx, s.index, s.partition)
}

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
// Version they were computed from. It reads the tag and its members once when
// they are all of the Tagger's model; members embedded under another model are
// re-embedded first, and the tag and its members read again (see
// currentMembers). The name is embedded, as a document, when the tag has no
// name vector of the Tagger's model; other scoring settings alone embed
// nothing. A tag deleted meanwhile is no error, and neither is one changed
// meanwhile: the change left it stale, for its writer or the next Suggest to
// rebuild. Nor is a member that is still of another model after the re-embed,
// because it was edited under that model meanwhile: rather than build the tag
// without it, rebuild stores nothing and leaves the tag stale, for the next
// pass of rebuildStale to re-embed the member. An error comes back wrapped,
// naming the tag.
func (s *Scope) rebuild(ctx context.Context, id string) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("autotag: rebuild tag %q: %w", id, err)
		}
	}()
	rec, members, ok, err := s.currentMembers(ctx, id)
	if err != nil || !ok {
		return err // !ok: deleted, or a member left of another model (see above)
	}
	store, cfg := s.tg.store, s.tg.cfg
	nameVec := rec.NameVec
	if nameVec == nil || rec.Model != cfg.Model {
		if nameVec, err = s.tg.embedOne(ctx, semsearch.RoleDocument, rec.Name); err != nil {
			return err
		}
	}
	// blend and tagCutoff index every member by the name vector's length
	var vecs [][]float32
	for _, m := range members {
		if len(m.Vec) != len(nameVec) {
			return fmt.Errorf("autotag: item %q has a vector of %d dimensions and the tag's name one of %d", m.ItemID, len(m.Vec), len(nameVec))
		}
		vecs = append(vecs, m.Vec)
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

// readTag reads the tag and then its members, in that order: a change that
// lands in between bumps the tag's Version past the one read, and
// SetTagVectors refuses vectors computed for an older one. ok is false when
// the tag is gone, and then its members are not read.
func (s *Scope) readTag(ctx context.Context, id string) (rec TagRecord, members []Member, ok bool, err error) {
	rec, ok, err = s.tg.store.Tag(ctx, s.index, s.partition, id)
	if err != nil || !ok {
		return TagRecord{}, nil, false, err
	}
	if members, err = s.tg.store.Members(ctx, s.index, s.partition, id); err != nil {
		return TagRecord{}, nil, false, err
	}
	return rec, members, true, nil
}

// currentMembers reads the tag and its members (see readTag), all a rebuild
// needs when every member is of the Tagger's model. Otherwise it re-embeds
// those of another model (see refreshMembers) and reads the tag and its
// members again. ok is false when there is nothing to build: the tag is gone,
// or a member is still of another model (see rebuild).
func (s *Scope) currentMembers(ctx context.Context, id string) (rec TagRecord, members []Member, ok bool, err error) {
	rec, members, ok, err = s.readTag(ctx, id)
	if err != nil || !ok || !slices.ContainsFunc(members, s.tg.otherModel) {
		return rec, members, ok, err
	}
	if err = s.refreshMembers(ctx, members); err != nil {
		return TagRecord{}, nil, false, err
	}
	if rec, members, ok, err = s.readTag(ctx, id); err != nil || !ok {
		return rec, members, ok, err
	}
	return rec, members, !slices.ContainsFunc(members, s.tg.otherModel), nil
}

// otherModel reports whether the member's vector comes from another model than
// the Tagger's.
func (t *Tagger) otherModel(m Member) bool { return m.Model != t.cfg.Model }

// refreshMembers re-embeds, from the text the store keeps, the members whose
// vectors come from another model than the Tagger's, so that rebuild can blend
// them in (see reembedItem). Each item it re-embeds bumps the Version of every
// tag linked to it: currentMembers reads this tag again after that, and the
// others are left stale, for their own rebuilds.
func (s *Scope) refreshMembers(ctx context.Context, members []Member) error {
	for _, m := range members {
		if !s.tg.otherModel(m) {
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
		// bumped this tag, and currentMembers reads what that writer left
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

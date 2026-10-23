package autotag

import (
	"context"
	"errors"
	"maps"
	"slices"
	"sync"
)

var errEmptyIndex = errors.New("autotag: empty index")

// scopeKey names one scope of a MemStore.
type scopeKey struct{ index, partition string }

// memScope is what one scope holds. links maps the id of a tag to the ids of
// the items linked to it. A link exists only between a tag and an item that
// both exist, and goes when either does.
type memScope struct {
	tags  map[string]TagRecord
	items map[string]ItemRecord
	links map[string]map[string]bool
}

// MemStore is a Store that keeps everything in memory. It suits tests and tiny
// corpora. It is safe for concurrent use, and it copies every vector it is given
// and every vector it returns, so no caller shares a slice with it. Create it
// with NewMemStore: the zero value has no maps and panics on its first write.
type MemStore struct {
	mu     sync.Mutex
	scopes map[scopeKey]*memScope
}

var _ Store = (*MemStore)(nil)

// NewMemStore returns an empty MemStore.
func NewMemStore() *MemStore { return &MemStore{scopes: make(map[scopeKey]*memScope)} }

func (r TagRecord) clone() TagRecord {
	r.NameVec = slices.Clone(r.NameVec)
	r.Vec = slices.Clone(r.Vec)
	return r
}

func (r ItemRecord) clone() ItemRecord {
	r.Vec = slices.Clone(r.Vec)
	return r
}

// existing returns the scope's data. A scope that nothing was ever put in has
// none, and existing then returns an empty scope that the store does not keep,
// so a read, or a delete that finds nothing, creates nothing. That scope has no
// maps, and adding to one panics: only ensure makes room.
func (s *MemStore) existing(index, partition string) *memScope {
	if sc, ok := s.scopes[scopeKey{index, partition}]; ok {
		return sc
	}
	return &memScope{}
}

// ensure returns the scope's data, creating the scope when it is new.
func (s *MemStore) ensure(index, partition string) *memScope {
	key := scopeKey{index, partition}
	sc, ok := s.scopes[key]
	if !ok {
		sc = &memScope{
			tags:  make(map[string]TagRecord),
			items: make(map[string]ItemRecord),
			links: make(map[string]map[string]bool),
		}
		s.scopes[key] = sc
	}
	return sc
}

// linkedTags returns the ids of the tags linked to the item, ordered.
func (sc *memScope) linkedTags(itemID string) []string {
	var ids []string
	for tagID, items := range sc.links {
		if items[itemID] {
			ids = append(ids, tagID)
		}
	}
	slices.Sort(ids)
	return ids
}

// bump adds one to the Version of each of the tags, which must exist.
func (sc *memScope) bump(ids []string) {
	for _, id := range ids {
		rec := sc.tags[id]
		rec.Version++
		sc.tags[id] = rec
	}
}

func (s *MemStore) Tag(_ context.Context, index, partition, tagID string) (TagRecord, bool, error) {
	if index == "" {
		return TagRecord{}, false, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.existing(index, partition).tags[tagID]
	return rec.clone(), ok, nil
}

func (s *MemStore) Tags(_ context.Context, index, partition string) ([]TagRecord, error) {
	if index == "" {
		return nil, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.existing(index, partition)
	var out []TagRecord
	for _, id := range slices.Sorted(maps.Keys(sc.tags)) {
		out = append(out, sc.tags[id].clone())
	}
	return out, nil
}

func (s *MemStore) PutTag(_ context.Context, index, partition, tagID, name string) (bool, error) {
	if index == "" {
		return false, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.ensure(index, partition)
	rec, ok := sc.tags[tagID]
	switch {
	case !ok:
		sc.tags[tagID] = TagRecord{ID: tagID, Name: name, Version: 1}
	case rec.Name == name:
		return false, nil
	default:
		rec.Name, rec.NameVec = name, nil
		rec.Version++
		sc.tags[tagID] = rec
	}
	return true, nil
}

func (s *MemStore) SetTagVectors(_ context.Context, index, partition, tagID string, version int64, v TagVectors) (bool, error) {
	if index == "" {
		return false, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.existing(index, partition)
	rec, ok := sc.tags[tagID]
	if !ok || rec.Version != version {
		return false, nil
	}
	rec.NameVec, rec.Vec = slices.Clone(v.NameVec), slices.Clone(v.Vec)
	rec.Cutoff, rec.Model, rec.Params, rec.Built = v.Cutoff, v.Model, v.Params, version
	sc.tags[tagID] = rec
	return true, nil
}

func (s *MemStore) DeleteTag(_ context.Context, index, partition, tagID string) error {
	if index == "" {
		return errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.existing(index, partition)
	delete(sc.tags, tagID)
	delete(sc.links, tagID)
	return nil
}

func (s *MemStore) Item(_ context.Context, index, partition, itemID string) (ItemRecord, bool, error) {
	if index == "" {
		return ItemRecord{}, false, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.existing(index, partition).items[itemID]
	return rec.clone(), ok, nil
}

func (s *MemStore) PutItem(_ context.Context, index, partition string, item ItemRecord) ([]string, error) {
	if index == "" {
		return nil, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.ensure(index, partition)
	sc.items[item.ID] = item.clone()
	linked := sc.linkedTags(item.ID)
	sc.bump(linked)
	return linked, nil
}

func (s *MemStore) SetItemVector(_ context.Context, index, partition, itemID, text string, vec []float32, model string) ([]string, bool, error) {
	if index == "" {
		return nil, false, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.existing(index, partition)
	item, ok := sc.items[itemID]
	if !ok || item.Text != text {
		return nil, false, nil
	}
	item.Vec, item.Model = slices.Clone(vec), model
	sc.items[itemID] = item
	linked := sc.linkedTags(itemID)
	sc.bump(linked)
	return linked, true, nil
}

func (s *MemStore) DeleteItem(_ context.Context, index, partition, itemID string) ([]string, error) {
	if index == "" {
		return nil, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.existing(index, partition)
	linked := sc.linkedTags(itemID)
	for _, tagID := range linked {
		delete(sc.links[tagID], itemID)
	}
	delete(sc.items, itemID)
	sc.bump(linked)
	return linked, nil
}

func (s *MemStore) Link(_ context.Context, index, partition, itemID, tagID string) (bool, error) {
	if index == "" {
		return false, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.existing(index, partition)
	if _, ok := sc.items[itemID]; !ok {
		return false, ErrUnknownItem
	}
	if _, ok := sc.tags[tagID]; !ok {
		return false, ErrUnknownTag
	}
	if sc.links[tagID][itemID] {
		return false, nil
	}
	if sc.links[tagID] == nil {
		sc.links[tagID] = make(map[string]bool)
	}
	sc.links[tagID][itemID] = true
	sc.bump([]string{tagID})
	return true, nil
}

func (s *MemStore) Unlink(_ context.Context, index, partition, itemID, tagID string) (bool, error) {
	if index == "" {
		return false, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.existing(index, partition)
	if !sc.links[tagID][itemID] {
		return false, nil
	}
	delete(sc.links[tagID], itemID)
	sc.bump([]string{tagID})
	return true, nil
}

func (s *MemStore) Members(_ context.Context, index, partition, tagID string) ([]Member, error) {
	if index == "" {
		return nil, errEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.existing(index, partition)
	var out []Member
	for _, itemID := range slices.Sorted(maps.Keys(sc.links[tagID])) {
		item := sc.items[itemID]
		out = append(out, Member{ItemID: itemID, Vec: slices.Clone(item.Vec), Model: item.Model})
	}
	return out, nil
}

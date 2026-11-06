package tagging

import (
	"container/list"
	"context"
	"maps"
	"slices"
	"strings"
	"sync"
)

// DefaultMemIndexBytes is the MemIndex budget when none is given: enough for
// tens of thousands of tags of a typical embedding size.
const DefaultMemIndexBytes = 256 << 20

// spaceKey identifies one space of a MemIndex.
type spaceKey struct{ index, partition string }

// memSpace is the tags and recorded state of one space.
type memSpace struct {
	key      spaceKey
	state    string
	hasState bool
	tags     map[string]TagVector
	size     int64
}

// tagBytes is the memory a tag is accounted for: its vector plus its strings
// plus a fixed allowance for the struct and map overhead.
func tagBytes(t TagVector) int64 {
	return int64(4*len(t.Vec) + len(t.ID) + len(t.Name) + len(t.Fingerprint) + 64)
}

// MemIndex is the in-memory TagIndex. Candidates returns every tag of the
// space, so it suits spaces small enough to score exhaustively. Spaces are
// evicted least-recently-used while the total is over budget; an evicted space
// reads as unknown, which makes the Tagger rebuild it. It is safe for
// concurrent use.
type MemIndex struct {
	mu     sync.Mutex
	budget int64
	total  int64
	spaces map[spaceKey]*list.Element // values are *memSpace
	lru    *list.List                 // front is the most recently used
}

// NewMemIndex returns an empty MemIndex holding at most about budgetBytes; a
// budget of zero or less means DefaultMemIndexBytes. A single space larger
// than the budget is still kept, alone.
func NewMemIndex(budgetBytes int64) *MemIndex {
	if budgetBytes <= 0 {
		budgetBytes = DefaultMemIndexBytes
	}
	return &MemIndex{budget: budgetBytes, spaces: map[spaceKey]*list.Element{}, lru: list.New()}
}

// State implements TagIndex. The returned map is a fresh copy.
func (m *MemIndex) State(_ context.Context, index, partition string) (string, map[string]string, bool, error) {
	if index == "" {
		return "", nil, false, errEmptyIndex
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	sp := m.touch(spaceKey{index, partition})
	if sp == nil {
		return "", nil, false, nil
	}
	fps := make(map[string]string, len(sp.tags))
	for id, t := range sp.tags {
		fps[id] = t.Fingerprint
	}
	return sp.state, fps, true, nil
}

// Apply implements TagIndex. The drops are applied before the puts, so an id
// in both is held.
func (m *MemIndex) Apply(_ context.Context, index, partition, state string, put []TagVector, drop []string) error {
	if index == "" {
		return errEmptyIndex
	}
	key := spaceKey{index, partition}
	m.mu.Lock()
	defer m.mu.Unlock()
	sp := m.touch(key)
	if sp == nil {
		sp = &memSpace{key: key, tags: make(map[string]TagVector, len(put))}
	}
	for _, id := range drop {
		delete(sp.tags, id)
	}
	for _, t := range put {
		sp.tags[t.ID] = t
	}
	if state != "" {
		sp.state, sp.hasState = state, true
	}
	if !sp.hasState && len(sp.tags) == 0 {
		m.remove(key)
		return nil
	}
	var size int64
	for _, t := range sp.tags {
		size += tagBytes(t)
	}
	if el, ok := m.spaces[key]; ok {
		m.total += size - sp.size
		m.lru.MoveToFront(el)
	} else {
		m.spaces[key] = m.lru.PushFront(sp)
		m.total += size
	}
	sp.size = size
	m.evict(key)
	return nil
}

// Candidates implements TagIndex: every tag of the space, ordered by id. The
// slice is fresh but the vectors in it are shared.
func (m *MemIndex) Candidates(_ context.Context, index, partition string, _ []float32) ([]TagVector, error) {
	if index == "" {
		return nil, errEmptyIndex
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	sp := m.touch(spaceKey{index, partition})
	if sp == nil {
		return nil, nil
	}
	out := slices.SortedFunc(maps.Values(sp.tags), func(a, b TagVector) int { return strings.Compare(a.ID, b.ID) })
	if out == nil {
		out = []TagVector{}
	}
	return out, nil
}

// touch returns the space and marks it most recently used, or nil when it is
// unknown. The caller holds the mutex.
func (m *MemIndex) touch(key spaceKey) *memSpace {
	el, ok := m.spaces[key]
	if !ok {
		return nil
	}
	m.lru.MoveToFront(el)
	return el.Value.(*memSpace)
}

// remove forgets a space. The caller holds the mutex.
func (m *MemIndex) remove(key spaceKey) {
	el, ok := m.spaces[key]
	if !ok {
		return
	}
	m.total -= el.Value.(*memSpace).size
	m.lru.Remove(el)
	delete(m.spaces, key)
}

// evict drops least-recently-used spaces while over budget, never the space
// just used. The caller holds the mutex.
func (m *MemIndex) evict(keep spaceKey) {
	for m.total > m.budget {
		el := m.lru.Back()
		if el == nil {
			return
		}
		sp := el.Value.(*memSpace)
		if sp.key == keep {
			return
		}
		m.remove(sp.key)
	}
}

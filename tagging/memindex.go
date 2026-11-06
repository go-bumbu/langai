package tagging

import (
	"container/list"
	"context"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// DefaultMemIndexBytes is the MemIndex budget when none is given: enough for
// tens of thousands of tags of a typical embedding size.
const DefaultMemIndexBytes = 256 << 20

// unrecordedPrefix starts the state of a space no Apply could record one for.
// The engine's states are hex digests, so none starts with a NUL.
const unrecordedPrefix = "\x00unrecorded "

// spaceKey identifies one space of a MemIndex.
type spaceKey struct{ index, partition string }

// memSpace is the tags and state of one space. tags and fingerprints are built
// anew at every Apply and never changed after, so State and Candidates hand
// them out without a sort or a copy: a Suggest at an unchanged revision holds
// the index's one mutex for no longer than a map lookup.
type memSpace struct {
	key          spaceKey
	state        string            // never empty: see TagIndex.State
	tags         []TagVector       // ordered by id
	fingerprints map[string]string // by id
	size         int64
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
	mu         sync.Mutex
	budget     int64
	total      int64
	spaces     map[spaceKey]*list.Element // values are *memSpace
	lru        *list.List                 // front is the most recently used
	unrecorded int64                      // how many unrecorded states were handed out
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

// State implements TagIndex. The returned map is the space's own: Apply replaces
// it rather than changing it, and callers must not modify it.
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
	return sp.state, sp.fingerprints, true, nil
}

// Apply implements TagIndex. The state it leaves when it cannot record its own
// is unique to that Apply, so a caller that read the space before it never
// finds its from still standing, however many stale Applies overlap.
func (m *MemIndex) Apply(_ context.Context, index, partition, from, state string, put []TagVector, drop []string) error {
	if index == "" {
		return errEmptyIndex
	}
	key := spaceKey{index, partition}
	m.mu.Lock()
	defer m.mu.Unlock()
	sp := m.touch(key)
	cur := ""
	var held []TagVector
	if sp != nil {
		cur, held = sp.state, sp.tags
	}
	tags := merged(held, put, drop)
	next := cur // a state of "" over the state read leaves it as it is
	if from == cur && state != "" {
		next = state
	}
	if from != cur || next == "" {
		if len(tags) == 0 {
			m.remove(key)
			return nil
		}
		m.unrecorded++
		next = unrecordedPrefix + strconv.FormatInt(m.unrecorded, 10)
	}
	fingerprints := make(map[string]string, len(tags))
	var size int64
	for _, t := range tags {
		fingerprints[t.ID] = t.Fingerprint
		size += tagBytes(t)
	}
	if sp == nil {
		sp = &memSpace{key: key}
		m.spaces[key] = m.lru.PushFront(sp)
	}
	m.total += size - sp.size
	sp.state, sp.tags, sp.fingerprints, sp.size = next, tags, fingerprints, size
	m.evict(key)
	return nil
}

// merged returns the tags of held without the ids in drop, with put upserted,
// in a new slice ordered by id. A later put of one id replaces an earlier one.
func merged(held, put []TagVector, drop []string) []TagVector {
	upserts := make(map[string]TagVector, len(put))
	for _, t := range put {
		upserts[t.ID] = t
	}
	dropped := make(map[string]bool, len(drop))
	for _, id := range drop {
		dropped[id] = true
	}
	tags := make([]TagVector, 0, len(held)+len(upserts))
	for _, t := range held {
		if _, upserted := upserts[t.ID]; !upserted && !dropped[t.ID] {
			tags = append(tags, t)
		}
	}
	tags = slices.AppendSeq(tags, maps.Values(upserts))
	slices.SortFunc(tags, func(a, b TagVector) int { return strings.Compare(a.ID, b.ID) })
	return tags
}

// Candidates implements TagIndex: every tag of the space, ordered by id. The
// slice is the space's own: Apply replaces it rather than changing it, and
// callers must not modify it.
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
	return sp.tags, nil
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

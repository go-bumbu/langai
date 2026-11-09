package vector

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
)

// MemStore is a Store that keeps everything in memory and answers searches by
// brute force, scoring every chunk in scope with exact cosine similarity. It
// suits tests, tiny corpora and evaluation, where no approximate index may
// colour the results. An index's dimensions are those of the first vector
// written to it, as a SQL store's tables have the dimensions they were built
// with, until DeleteIndex forgets it. It is safe for concurrent use.
type MemStore struct {
	mu      sync.RWMutex
	indexes map[string]*memIndex
}

// memIndex is one index of a MemStore: its dimensions, the model of each
// collection, and the documents of each collection.
type memIndex struct {
	dims   int
	models map[string]string             // collection -> model
	colls  map[memColl]map[string]memDoc // collection -> doc id -> doc
}

type memColl struct{ partition, collection string }

type memDoc struct {
	hash   string
	chunks []Chunk
}

// NewMemStore returns an empty MemStore.
func NewMemStore() *MemStore { return &MemStore{indexes: make(map[string]*memIndex)} }

var _ Store = (*MemStore)(nil)

// Upsert implements Store. It keeps copies of the chunks, so the caller may
// reuse its slices.
func (s *MemStore) Upsert(_ context.Context, c Coord, model string, docs ...Doc) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ix := s.indexes[c.Index]
	dims := 0
	if ix != nil {
		dims = ix.dims
	} else if first := firstVector(docs); first != nil {
		dims = len(first)
	}
	if err := CheckWrite(c, model, dims, docs); err != nil {
		return err
	}
	if ix == nil {
		ix = &memIndex{models: map[string]string{}, colls: map[memColl]map[string]memDoc{}}
		s.indexes[c.Index] = ix
	}
	if err := ix.checkModel(c.Collection, model, false); err != nil {
		return err
	}
	if model != "" {
		ix.models[c.Collection] = model // the same, or the collection held no documents
	}
	if ix.dims == 0 {
		ix.dims = dims
	}
	key := memColl{c.Partition, c.Collection}
	coll := ix.colls[key]
	if coll == nil {
		coll = map[string]memDoc{}
		ix.colls[key] = coll
	}
	for _, d := range docs {
		chunks := make([]Chunk, len(d.Chunks))
		for i, ch := range d.Chunks {
			chunks[i] = Chunk{Seq: ch.Seq, Text: ch.Text, Vector: slices.Clone(ch.Vector)}
		}
		slices.SortStableFunc(chunks, func(a, b Chunk) int { return cmp.Compare(a.Seq, b.Seq) })
		coll[d.ID] = memDoc{hash: d.Hash, chunks: chunks}
	}
	return nil
}

// firstVector returns the vector of the first chunk among docs, nil when
// there is none.
func firstVector(docs []Doc) []float32 {
	for _, d := range docs {
		if len(d.Chunks) > 0 {
			return d.Chunks[0].Vector
		}
	}
	return nil
}

// checkModel fails with ErrModelMismatch when model is not empty and the
// collection, or with all any collection of the index, recorded another
// and still holds documents.
func (ix *memIndex) checkModel(collection, model string, all bool) error {
	if model == "" {
		return nil
	}
	for coll, m := range ix.models {
		if (all || coll == collection) && m != model && ix.holds(coll) {
			return fmt.Errorf("%w: collection %q holds %q, not %q", ErrModelMismatch, coll, m, model)
		}
	}
	return nil
}

// holds reports whether any partition of collection has documents.
func (ix *memIndex) holds(collection string) bool {
	for key, docs := range ix.colls {
		if key.collection == collection && len(docs) > 0 {
			return true
		}
	}
	return false
}

// Model implements Store.
func (s *MemStore) Model(_ context.Context, c Coord) (string, error) {
	if err := CheckCoord(c); err != nil {
		return "", err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	ix := s.indexes[c.Index]
	if ix == nil || !ix.holds(c.Collection) {
		return "", nil
	}
	return ix.models[c.Collection], nil
}

// DeleteCollection implements Store.
func (s *MemStore) DeleteCollection(_ context.Context, c Coord) error {
	if err := CheckCoord(c); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if ix := s.indexes[c.Index]; ix != nil {
		delete(ix.colls, memColl{c.Partition, c.Collection})
		if !ix.holds(c.Collection) {
			delete(ix.models, c.Collection)
		}
	}
	return nil
}

// Delete implements Store.
func (s *MemStore) Delete(_ context.Context, c Coord, ids ...string) error {
	if err := CheckRead(c, ids...); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if ix := s.indexes[c.Index]; ix != nil {
		coll := ix.colls[memColl{c.Partition, c.Collection}]
		for _, id := range ids {
			delete(coll, id)
		}
	}
	return nil
}

// DeleteIndex implements Store.
func (s *MemStore) DeleteIndex(_ context.Context, index string) error {
	if index == "" {
		return ErrEmptyIndex
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.indexes, index)
	return nil
}

// coll returns the documents of c, nil when there are none.
func (s *MemStore) coll(c Coord) map[string]memDoc {
	if ix := s.indexes[c.Index]; ix != nil {
		return ix.colls[memColl{c.Partition, c.Collection}]
	}
	return nil
}

// Hashes implements Store.
func (s *MemStore) Hashes(_ context.Context, c Coord, ids ...string) (map[string]string, error) {
	if err := CheckRead(c, ids...); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	coll := s.coll(c)
	out := make(map[string]string, len(ids))
	for _, id := range ids {
		if d, ok := coll[id]; ok {
			out[id] = d.hash
		}
	}
	return out, nil
}

// Docs implements Store.
func (s *MemStore) Docs(_ context.Context, c Coord, after string, limit int) ([]DocInfo, error) {
	if err := CheckDocs(c, after); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []DocInfo
	for id, d := range s.coll(c) {
		if id > after {
			out = append(out, DocInfo{ID: id, Hash: d.hash})
		}
	}
	slices.SortFunc(out, func(a, b DocInfo) int { return strings.Compare(a.ID, b.ID) })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// Chunks implements Store; the vectors are copies.
func (s *MemStore) Chunks(_ context.Context, c Coord, id string) ([]Chunk, error) {
	if err := CheckRead(c, id); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	doc, ok := s.coll(c)[id]
	if !ok || len(doc.chunks) == 0 {
		return nil, nil
	}
	out := make([]Chunk, len(doc.chunks))
	for i, ch := range doc.chunks {
		out[i] = Chunk{Seq: ch.Seq, Text: ch.Text, Vector: slices.Clone(ch.Vector)}
	}
	return out, nil
}

// Search implements Store, scoring every chunk in scope.
func (s *MemStore) Search(_ context.Context, c Coord, q Query) ([]Match, error) {
	if err := CheckCoord(c); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	ix := s.indexes[c.Index]
	dims := len(q.Vector)
	if ix != nil && ix.dims > 0 {
		dims = ix.dims
	}
	if err := CheckSearch(c, dims, q); err != nil {
		return nil, err
	}
	if ix == nil || q.Limit <= 0 {
		return nil, nil
	}
	if err := ix.checkModel(c.Collection, q.Model, q.AllCollections); err != nil {
		return nil, err
	}
	keep := idFilter(q)
	var out []Match
	for key, coll := range ix.colls {
		if (!q.AllPartitions && key.partition != c.Partition) || (!q.AllCollections && key.collection != c.Collection) {
			continue
		}
		at := Coord{Index: c.Index, Partition: key.partition, Collection: key.collection}
		for id, doc := range coll {
			if !keep(id) {
				continue
			}
			if m, ok := bestChunk(q.Vector, doc.chunks); ok {
				m.Coord, m.DocID = at, id
				out = append(out, m)
			}
		}
	}
	slices.SortFunc(out, compareMatches)
	if len(out) > q.Limit {
		out = out[:q.Limit]
	}
	return out, nil
}

// bestChunk returns the chunk of chunks (ordered by Seq) closest to query,
// the first one on a tie; false when there are none.
func bestChunk(query []float32, chunks []Chunk) (Match, bool) {
	var best Match
	for i, ch := range chunks {
		score := cosine(query, ch.Vector)
		if i == 0 || score > best.Score {
			best = Match{Seq: ch.Seq, Text: ch.Text, Score: score}
		}
	}
	return best, len(chunks) > 0
}

// idFilter returns whether q's Only and Exclude lists keep a document id.
func idFilter(q Query) func(id string) bool {
	only := setOf(q.Only)
	exclude := setOf(q.Exclude)
	return func(id string) bool {
		return (len(only) == 0 || only[id]) && !exclude[id]
	}
}

func setOf(ids []string) map[string]bool {
	set := make(map[string]bool, len(ids))
	for _, id := range ids {
		set[id] = true
	}
	return set
}

// compareMatches orders matches as Search returns them: by descending score,
// then by document id, then by Seq, and last by partition and collection, so
// that the order is the same on every run and in every Store.
func compareMatches(a, b Match) int {
	return cmp.Or(
		cmp.Compare(b.Score, a.Score),
		strings.Compare(a.DocID, b.DocID),
		cmp.Compare(a.Seq, b.Seq),
		strings.Compare(a.Coord.Partition, b.Coord.Partition),
		strings.Compare(a.Coord.Collection, b.Coord.Collection),
	)
}

// cosine returns the cosine similarity of two vectors of one length, neither
// of them zero (Validate holds for both).
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

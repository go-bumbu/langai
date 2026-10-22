package semsearch

import (
	"context"
	"math"
	"sort"
	"sync"
)

type memKey struct{ index, partition, collection, docID string }

// MemStore is a Store that keeps everything in memory and answers searches by
// brute force, scoring every chunk in scope with exact cosine similarity. It
// suits tests, tiny corpora and evaluation, where no approximate index may
// colour the results. It is safe for concurrent use.
type MemStore struct {
	mu   sync.RWMutex
	docs map[memKey]memDoc
}

type memDoc struct {
	hash   string
	chunks []Chunk
}

// NewMemStore returns an empty MemStore.
func NewMemStore() *MemStore { return &MemStore{docs: make(map[memKey]memDoc)} }

func (s *MemStore) Upsert(_ context.Context, index, partition, collection, docID, contentHash string, chunks []Chunk) error {
	cp := make([]Chunk, len(chunks))
	copy(cp, chunks)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docs[memKey{index, partition, collection, docID}] = memDoc{hash: contentHash, chunks: cp}
	return nil
}

func (s *MemStore) Delete(_ context.Context, index, partition, collection, docID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.docs, memKey{index, partition, collection, docID})
	return nil
}

func (s *MemStore) ContentHash(_ context.Context, index, partition, collection, docID string) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.docs[memKey{index, partition, collection, docID}].hash, nil
}

// Search implements Store.
func (s *MemStore) Search(ctx context.Context, index, partition, collection string, vector []float32, limit int) ([]Match, error) {
	return s.SearchScope(ctx, index, ScopeFilter{Partition: partition, Collection: collection}, vector, limit)
}

// SearchScope implements Store.
func (s *MemStore) SearchScope(_ context.Context, index string, f ScopeFilter, vector []float32, limit int) ([]Match, error) {
	if limit <= 0 {
		return nil, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []Match
	for k, doc := range s.docs {
		if k.index != index {
			continue
		}
		if f.Partition != "" && k.partition != f.Partition {
			continue
		}
		if f.Collection != "" && k.collection != f.Collection {
			continue
		}
		for _, c := range doc.chunks {
			out = append(out, Match{DocID: k.docID, Partition: k.partition, Collection: k.collection, Seq: c.Seq, Text: c.Text, Score: cosine(vector, c.Vector)})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// cosine returns the cosine similarity of two vectors, 0 for mismatched or
// zero-length input.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

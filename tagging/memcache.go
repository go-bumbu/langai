package tagging

import (
	"context"
	"errors"
	"slices"
	"sync"
	"time"
)

var errEmptyVector = errors.New("tagging: empty vector")

// cacheEntry is one cached vector and when it was last used.
type cacheEntry struct {
	vec  []float32
	used time.Time
}

// MemCache is the in-memory EmbeddingCache: the reference implementation the
// conformance suite runs against, and a cache for callers that do not need to
// outlive the process. It is safe for concurrent use.
type MemCache struct {
	mu      sync.Mutex
	indexes map[string]map[CacheKey]cacheEntry
	now     func() time.Time // time.Now, replaced by tests
}

// NewMemCache returns an empty MemCache.
func NewMemCache() *MemCache {
	return NewMemCacheWithClock(time.Now)
}

// NewMemCacheWithClock returns an empty MemCache that takes its last-used
// times from now instead of time.Now, for a test that decides what time it is
// (the conformance suite, cachetest, does). A nil now means time.Now.
func NewMemCacheWithClock(now func() time.Time) *MemCache {
	if now == nil {
		now = time.Now
	}
	return &MemCache{indexes: map[string]map[CacheKey]cacheEntry{}, now: now}
}

// Get implements EmbeddingCache. The returned slices are copies.
func (c *MemCache) Get(_ context.Context, index string, keys []CacheKey) (map[CacheKey][]float32, error) {
	if index == "" {
		return nil, errEmptyIndex
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	out := make(map[CacheKey][]float32, len(keys))
	entries := c.indexes[index]
	for _, k := range keys {
		e, ok := entries[k]
		if !ok {
			continue
		}
		if now.Sub(e.used) >= CacheRefreshAfter {
			e.used = now
			entries[k] = e
		}
		out[k] = slices.Clone(e.vec)
	}
	return out, nil
}

// Put implements EmbeddingCache. It keeps copies of the vectors.
func (c *MemCache) Put(_ context.Context, index string, vecs map[CacheKey][]float32) error {
	if index == "" {
		return errEmptyIndex
	}
	for _, v := range vecs {
		if len(v) == 0 {
			return errEmptyVector
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	entries := c.indexes[index]
	if entries == nil {
		entries = make(map[CacheKey]cacheEntry, len(vecs))
		c.indexes[index] = entries
	}
	now := c.now()
	for k, v := range vecs {
		entries[k] = cacheEntry{vec: slices.Clone(v), used: now}
	}
	return nil
}

// DeleteIndex implements EmbeddingCache.
func (c *MemCache) DeleteIndex(_ context.Context, index string) error {
	if index == "" {
		return errEmptyIndex
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.indexes, index)
	return nil
}

// Prune implements EmbeddingCache.
func (c *MemCache) Prune(_ context.Context, unusedSince time.Time) (int64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	var n int64
	for name, entries := range c.indexes {
		for k, e := range entries {
			if e.used.Before(unusedSince) {
				delete(entries, k)
				n++
			}
		}
		if len(entries) == 0 {
			delete(c.indexes, name)
		}
	}
	return n, nil
}

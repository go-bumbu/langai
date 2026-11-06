// Package cachetest is the conformance suite for tagging.EmbeddingCache. Every
// backend runs it from its own tests, so that all of them keep the contract the
// same way:
//
//	func TestConformance(t *testing.T) {
//		cachetest.Run(t, func(t *testing.T) tagging.EmbeddingCache { return open(t) })
//	}
package cachetest

import (
	"context"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/tagging"
)

const idx = "tenant"

// v1 and v2 are vectors at the edges of float32: the largest value, the
// smallest normal one, a negative and a zero. No test changes them; one that
// needs a vector to change clones it first.
var (
	v1 = []float32{1.5, -2.25, 3.4028235e38}
	v2 = []float32{1.1754944e-38, 0, -1}
)

// k1 and k2 are two keys that differ only in their text.
var (
	k1 = tagging.KeyFor("m", langai.RoleQuery, "one")
	k2 = tagging.KeyFor("m", langai.RoleQuery, "two")
)

// Run runs the suite against the cache newCache returns. Each subtest calls
// newCache with its own t, so a backend can tie what it opens to t.Cleanup, and
// the cache it returns must hold nothing yet.
func Run(t *testing.T, newCache func(t *testing.T) tagging.EmbeddingCache) {
	cases := []struct {
		name string
		run  func(t *testing.T, c tagging.EmbeddingCache)
	}{
		{"VectorsRoundTrip", vectorsRoundTrip},
		{"MissingKeysAreAbsent", missingKeysAreAbsent},
		{"EmptyKeysReturnEmpty", emptyKeysReturnEmpty},
		{"PutReplaces", putReplaces},
		{"KeysDifferByModelAndRole", keysDifferByModelAndRole},
		{"IndexesAreIsolated", indexesAreIsolated},
		{"EmptyIndexIsAnError", emptyIndexIsAnError},
		{"EmptyVectorIsAnError", emptyVectorIsAnError},
		{"SlicesAreCopies", slicesAreCopies},
		{"DeleteIndexDropsOnlyThatIndex", deleteIndexDropsOnlyThatIndex},
		{"PruneDeletesUnused", pruneDeletesUnused},
		{"LargeBatchesRoundTrip", largeBatchesRoundTrip},
		{"ConcurrentPutsAllLand", concurrentPutsAllLand},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) { c.run(t, newCache(t)) })
	}
}

// put stores vecs in index and fails the test on an error.
func put(t *testing.T, c tagging.EmbeddingCache, index string, vecs map[tagging.CacheKey][]float32) {
	t.Helper()
	if err := c.Put(context.Background(), index, vecs); err != nil {
		t.Fatalf("Put: %v", err)
	}
}

// get reads keys from index and fails the test on an error.
func get(t *testing.T, c tagging.EmbeddingCache, index string, keys ...tagging.CacheKey) map[tagging.CacheKey][]float32 {
	t.Helper()
	got, err := c.Get(context.Background(), index, keys)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	return got
}

// vectorsRoundTrip: what comes back is what went in, bit for bit.
func vectorsRoundTrip(t *testing.T, c tagging.EmbeddingCache) {
	put(t, c, idx, map[tagging.CacheKey][]float32{k1: v1, k2: v2})
	got := get(t, c, idx, k1, k2)
	if !reflect.DeepEqual(got[k1], v1) {
		t.Errorf("k1 = %v, want %v", got[k1], v1)
	}
	if !reflect.DeepEqual(got[k2], v2) {
		t.Errorf("k2 = %v, want %v", got[k2], v2)
	}
}

// missingKeysAreAbsent: a key never stored is not an error, it is just not there.
func missingKeysAreAbsent(t *testing.T, c tagging.EmbeddingCache) {
	if got := get(t, c, idx, k1); len(got) != 0 {
		t.Errorf("Get of a missing key = %v, want empty", got)
	}
}

// putReplaces: a second Put of a key overwrites the first.
func putReplaces(t *testing.T, c tagging.EmbeddingCache) {
	put(t, c, idx, map[tagging.CacheKey][]float32{k1: v1})
	put(t, c, idx, map[tagging.CacheKey][]float32{k1: v2})
	if got := get(t, c, idx, k1); !reflect.DeepEqual(got[k1], v2) {
		t.Errorf("k1 = %v, want %v", got[k1], v2)
	}
}

// keysDifferByModelAndRole: the same text under another model or role is
// another entry.
func keysDifferByModelAndRole(t *testing.T, c tagging.EmbeddingCache) {
	a := tagging.KeyFor("m1", langai.RoleQuery, "x")
	b := tagging.KeyFor("m2", langai.RoleQuery, "x")
	d := tagging.KeyFor("m1", langai.RoleDocument, "x")
	put(t, c, idx, map[tagging.CacheKey][]float32{a: {1}, b: {2}, d: {3}})
	got := get(t, c, idx, a, b, d)
	if len(got) != 3 {
		t.Fatalf("got %d entries, want 3", len(got))
	}
	if got[a][0] != 1 || got[b][0] != 2 || got[d][0] != 3 {
		t.Errorf("entries mixed up: %v", got)
	}
}

// indexesAreIsolated: an entry of one index is not seen from another.
func indexesAreIsolated(t *testing.T, c tagging.EmbeddingCache) {
	put(t, c, "a", map[tagging.CacheKey][]float32{k1: v1})
	if got := get(t, c, "b", k1); len(got) != 0 {
		t.Errorf("Get in another index = %v, want empty", got)
	}
}

// emptyIndexIsAnError: no operation on an index accepts the empty name.
func emptyIndexIsAnError(t *testing.T, c tagging.EmbeddingCache) {
	ctx := context.Background()
	if _, err := c.Get(ctx, "", []tagging.CacheKey{k1}); err == nil {
		t.Error("Get with an empty index: no error")
	}
	if err := c.Put(ctx, "", map[tagging.CacheKey][]float32{k1: v1}); err == nil {
		t.Error("Put with an empty index: no error")
	}
	if err := c.DeleteIndex(ctx, ""); err == nil {
		t.Error("DeleteIndex with an empty index: no error")
	}
}

// emptyVectorIsAnError: a nil or zero-length vector is refused.
func emptyVectorIsAnError(t *testing.T, c tagging.EmbeddingCache) {
	ctx := context.Background()
	if err := c.Put(ctx, idx, map[tagging.CacheKey][]float32{k1: nil}); err == nil {
		t.Error("Put of a nil vector: no error")
	}
	if err := c.Put(ctx, idx, map[tagging.CacheKey][]float32{k1: {}}); err == nil {
		t.Error("Put of an empty vector: no error")
	}
}

// slicesAreCopies: the cache shares no memory with its callers, in either
// direction.
func slicesAreCopies(t *testing.T, c tagging.EmbeddingCache) {
	in := slices.Clone(v1)
	put(t, c, idx, map[tagging.CacheKey][]float32{k1: in})
	in[0] = 99
	got := get(t, c, idx, k1)
	if !reflect.DeepEqual(got[k1], v1) {
		t.Fatalf("after mutating the Put slice, Get = %v, want %v", got[k1], v1)
	}
	got[k1][0] = 99
	if again := get(t, c, idx, k1); !reflect.DeepEqual(again[k1], v1) {
		t.Errorf("after mutating the returned slice, Get = %v, want %v", again[k1], v1)
	}
}

// deleteIndexDropsOnlyThatIndex: other indexes keep their entries.
func deleteIndexDropsOnlyThatIndex(t *testing.T, c tagging.EmbeddingCache) {
	put(t, c, "a", map[tagging.CacheKey][]float32{k1: v1})
	put(t, c, "b", map[tagging.CacheKey][]float32{k1: v2})
	if err := c.DeleteIndex(context.Background(), "a"); err != nil {
		t.Fatalf("DeleteIndex: %v", err)
	}
	if got := get(t, c, "a", k1); len(got) != 0 {
		t.Errorf("index a after DeleteIndex = %v, want empty", got)
	}
	if got := get(t, c, "b", k1); !reflect.DeepEqual(got[k1], v2) {
		t.Errorf("index b = %v, want %v", got[k1], v2)
	}
}

// pruneDeletesUnused: Prune removes what was last used before the cutoff, of
// every index, and counts it.
func pruneDeletesUnused(t *testing.T, c tagging.EmbeddingCache) {
	ctx := context.Background()
	k3 := tagging.KeyFor("m", langai.RoleQuery, "three")
	put(t, c, "a", map[tagging.CacheKey][]float32{k1: v1, k2: v2})
	put(t, c, "b", map[tagging.CacheKey][]float32{k3: v1})

	n, err := c.Prune(ctx, time.Now().Add(-time.Hour))
	if err != nil || n != 0 {
		t.Fatalf("Prune of nothing unused = %d, %v; want 0, nil", n, err)
	}
	n, err = c.Prune(ctx, time.Now().Add(time.Hour))
	if err != nil || n != 3 {
		t.Fatalf("Prune of everything = %d, %v; want 3, nil", n, err)
	}
	if got := get(t, c, "a", k1, k2); len(got) != 0 {
		t.Errorf("index a after Prune = %v, want empty", got)
	}
	if got := get(t, c, "b", k3); len(got) != 0 {
		t.Errorf("index b after Prune = %v, want empty", got)
	}
}

// largeBatchesRoundTrip: a batch bigger than any per-statement limit a SQL
// backend has still lands whole.
func largeBatchesRoundTrip(t *testing.T, c tagging.EmbeddingCache) {
	const n = 3000
	vecs := make(map[tagging.CacheKey][]float32, n)
	keys := make([]tagging.CacheKey, 0, n)
	for i := range n {
		k := tagging.KeyFor("m", langai.RoleQuery, strconv.Itoa(i))
		vecs[k] = []float32{float32(i), 1}
		keys = append(keys, k)
	}
	put(t, c, idx, vecs)
	got := get(t, c, idx, keys...)
	if len(got) != n {
		t.Fatalf("got %d entries, want %d", len(got), n)
	}
	for k, want := range vecs {
		if !reflect.DeepEqual(got[k], want) {
			t.Fatalf("entry %x = %v, want %v", k.Hash[:4], got[k], want)
		}
	}
}

// concurrentPutsAllLand: writers on one index lose nothing to each other.
func concurrentPutsAllLand(t *testing.T, c tagging.EmbeddingCache) {
	const writers, each = 20, 10
	keys := make([]tagging.CacheKey, 0, writers*each)
	var wg sync.WaitGroup
	errs := make(chan error, writers)
	for w := range writers {
		vecs := make(map[tagging.CacheKey][]float32, each)
		for i := range each {
			k := tagging.KeyFor("m", langai.RoleQuery, strconv.Itoa(w*each+i))
			vecs[k] = []float32{float32(w), float32(i)}
			keys = append(keys, k)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- c.Put(context.Background(), idx, vecs)
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	if got := get(t, c, idx, keys...); len(got) != writers*each {
		t.Errorf("got %d entries, want %d", len(got), writers*each)
	}
}

// emptyKeysReturnEmpty: a Get of no keys is an empty answer, not an error, for
// a nil slice as for an empty one.
func emptyKeysReturnEmpty(t *testing.T, c tagging.EmbeddingCache) {
	for name, keys := range map[string][]tagging.CacheKey{"nil": nil, "empty": {}} {
		got, err := c.Get(context.Background(), idx, keys)
		if err != nil || len(got) != 0 {
			t.Errorf("Get with %s keys = %v, %v; want empty, nil", name, got, err)
		}
	}
}

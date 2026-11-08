// Package cachetest is the conformance suite for tagging.EmbeddingCache. Every
// backend runs it from its own tests, so that all of them keep the contract the
// same way:
//
//	func TestConformance(t *testing.T) {
//		cachetest.Run(t, func(t *testing.T, now func() time.Time) tagging.EmbeddingCache { return open(t, now) })
//	}
//
// The suite runs on a clock of its own, which it moves by hand: a cache
// records when an entry was last used, and what a Get or a Prune does a day
// later can only be checked by a suite that decides what time it is. So a
// backend must be able to take its last-used times from the clock it is
// handed.
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

// clock is the time the cache under test runs on. It starts at a whole second,
// since a SQL backend keeps its times in seconds, and moves only when a test
// advances it.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

// newClock returns a clock at a fixed time.
func newClock() *clock {
	return &clock{now: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
}

// Now is the clock's time: what the cache under test is handed as its now.
func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

// advance moves the clock forward by d.
func (c *clock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// timeless adapts a test that never looks at the time.
func timeless(run func(*testing.T, tagging.EmbeddingCache)) func(*testing.T, tagging.EmbeddingCache, *clock) {
	return func(t *testing.T, c tagging.EmbeddingCache, _ *clock) { run(t, c) }
}

// Run runs the suite against the cache newCache returns. Each subtest calls
// newCache with its own t, so a backend can tie what it opens to t.Cleanup, and
// with the clock of that subtest: the cache it returns must hold nothing yet,
// and must take every last-used time from now, never from time.Now.
func Run(t *testing.T, newCache func(t *testing.T, now func() time.Time) tagging.EmbeddingCache) {
	cases := []struct {
		name string
		run  func(t *testing.T, c tagging.EmbeddingCache, clk *clock)
	}{
		{"VectorsRoundTrip", timeless(vectorsRoundTrip)},
		{"MissingKeysAreAbsent", timeless(missingKeysAreAbsent)},
		{"EmptyKeysReturnEmpty", timeless(emptyKeysReturnEmpty)},
		{"DuplicateKeysCountOnce", timeless(duplicateKeysCountOnce)},
		{"PutReplaces", timeless(putReplaces)},
		{"KeysDifferByModelAndRole", timeless(keysDifferByModelAndRole)},
		{"IndexesAreIsolated", timeless(indexesAreIsolated)},
		{"EmptyIndexIsAnError", timeless(emptyIndexIsAnError)},
		{"NamesNoDatabaseStoresAreErrors", timeless(namesNoDatabaseStoresAreErrors)},
		{"EmptyVectorIsAnError", timeless(emptyVectorIsAnError)},
		{"SlicesAreCopies", timeless(slicesAreCopies)},
		{"DeleteIndexDropsOnlyThatIndex", timeless(deleteIndexDropsOnlyThatIndex)},
		{"PruneDeletesUnused", pruneDeletesUnused},
		{"PruneKeepsWhatWasUsedAtTheCutoff", pruneKeepsWhatWasUsedAtTheCutoff},
		{"GetRefreshesStaleEntries", getRefreshesStaleEntries},
		{"TouchKeepsWhatPruneWouldDrop", touchKeepsWhatPruneWouldDrop},
		{"TouchIgnoresMissingKeys", touchIgnoresMissingKeys},
		{"TouchIsPerIndex", touchIsPerIndex},
		{"LargeBatchesRoundTrip", largeBatchesRoundTrip},
		{"ConcurrentPutsAllLand", timeless(concurrentPutsAllLand)},
		{"OverlappingConcurrentPutsAllLand", timeless(overlappingConcurrentPutsAllLand)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			clk := newClock()
			c.run(t, newCache(t, clk.Now), clk)
		})
	}
}

// prune removes what was last used before cutoff and fails the test unless
// that was want entries.
func prune(t *testing.T, c tagging.EmbeddingCache, cutoff time.Time, want int64) {
	t.Helper()
	n, err := c.Prune(context.Background(), cutoff)
	if err != nil || n != want {
		t.Fatalf("Prune = %d, %v; want %d, nil", n, err, want)
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
	if err := c.Touch(ctx, "", []tagging.CacheKey{k1}); err == nil {
		t.Error("Touch with an empty index: no error")
	}
	if err := c.DeleteIndex(ctx, ""); err == nil {
		t.Error("DeleteIndex with an empty index: no error")
	}
}

// namesNoDatabaseStoresAreErrors: an index or a model with a NUL byte, or one
// that is not valid UTF-8, is refused by every backend, not only by the
// database that cannot store it as text, and nothing is stored under it.
func namesNoDatabaseStoresAreErrors(t *testing.T, c tagging.EmbeddingCache) {
	ctx := context.Background()
	for name, bad := range map[string]string{"a NUL byte": "ten\x00ant", "invalid UTF-8": "ten\xffant"} {
		if _, err := c.Get(ctx, bad, []tagging.CacheKey{k1}); err == nil {
			t.Errorf("Get with an index holding %s: no error", name)
		}
		if err := c.Put(ctx, bad, map[tagging.CacheKey][]float32{k1: v1}); err == nil {
			t.Errorf("Put with an index holding %s: no error", name)
		}
		if err := c.DeleteIndex(ctx, bad); err == nil {
			t.Errorf("DeleteIndex with an index holding %s: no error", name)
		}
		if err := c.Touch(ctx, bad, []tagging.CacheKey{k1}); err == nil {
			t.Errorf("Touch with an index holding %s: no error", name)
		}
		k := tagging.KeyFor(bad, langai.RoleQuery, "one")
		if err := c.Touch(ctx, idx, []tagging.CacheKey{k1, k}); err == nil {
			t.Errorf("Touch with a model holding %s: no error", name)
		}
		if _, err := c.Get(ctx, idx, []tagging.CacheKey{k1, k}); err == nil {
			t.Errorf("Get with a model holding %s: no error", name)
		}
		if err := c.Put(ctx, idx, map[tagging.CacheKey][]float32{k1: v1, k: v2}); err == nil {
			t.Errorf("Put with a model holding %s: no error", name)
		}
	}
	if got := get(t, c, idx, k1); len(got) != 0 {
		t.Errorf("a refused Put stored %v, want nothing", got)
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
func pruneDeletesUnused(t *testing.T, c tagging.EmbeddingCache, clk *clock) {
	k3 := tagging.KeyFor("m", langai.RoleQuery, "three")
	put(t, c, "a", map[tagging.CacheKey][]float32{k1: v1, k2: v2})
	put(t, c, "b", map[tagging.CacheKey][]float32{k3: v1})

	prune(t, c, clk.Now().Add(-time.Hour), 0)
	prune(t, c, clk.Now().Add(time.Hour), 3)
	if got := get(t, c, "a", k1, k2); len(got) != 0 {
		t.Errorf("index a after Prune = %v, want empty", got)
	}
	if got := get(t, c, "b", k3); len(got) != 0 {
		t.Errorf("index b after Prune = %v, want empty", got)
	}
}

// pruneKeepsWhatWasUsedAtTheCutoff: the cutoff is "last used before", so an
// entry last used at that very time stays, and goes a second later.
func pruneKeepsWhatWasUsedAtTheCutoff(t *testing.T, c tagging.EmbeddingCache, clk *clock) {
	put(t, c, idx, map[tagging.CacheKey][]float32{k1: v1})
	prune(t, c, clk.Now(), 0)
	if got := get(t, c, idx, k1); len(got) != 1 {
		t.Fatal("an entry last used at the cutoff was pruned")
	}
	prune(t, c, clk.Now().Add(time.Second), 1)
}

// getRefreshesStaleEntries: a Get of an entry last used CacheRefreshAfter ago
// or longer rewrites its last-used time, so a Prune that follows keeps what is
// still read, while an entry nobody read keeps its old time and goes. A
// backend that skips the refresh passes every other test and then loses the
// vectors in use.
func getRefreshesStaleEntries(t *testing.T, c tagging.EmbeddingCache, clk *clock) {
	put(t, c, idx, map[tagging.CacheKey][]float32{k1: v1, k2: v2})
	clk.advance(tagging.CacheRefreshAfter + time.Hour)
	if got := get(t, c, idx, k1); !reflect.DeepEqual(got[k1], v1) {
		t.Fatalf("k1 = %v, want %v", got[k1], v1)
	}
	// k1 was read a moment ago; k2 not since it was stored, a day and an hour ago
	prune(t, c, clk.Now().Add(-time.Hour), 1)
	if got := get(t, c, idx, k1, k2); len(got) != 1 || !reflect.DeepEqual(got[k1], v1) {
		t.Errorf("after Prune = %v, want only k1: the Get refreshed its last-used time", got)
	}
}

// touchKeepsWhatPruneWouldDrop: Touch sets the last-used time of the keys it
// is given to now, whatever it was, also where a Get would not: an hour after
// the Put the entries are not CacheRefreshAfter old. The others keep theirs.
func touchKeepsWhatPruneWouldDrop(t *testing.T, c tagging.EmbeddingCache, clk *clock) {
	put(t, c, idx, map[tagging.CacheKey][]float32{k1: v1, k2: v2})
	clk.advance(time.Hour)
	if got := get(t, c, idx, k1, k2); len(got) != 2 {
		t.Fatalf("Get = %v, want both entries", got)
	}
	if err := c.Touch(context.Background(), idx, []tagging.CacheKey{k1, k1}); err != nil {
		t.Fatalf("Touch: %v", err)
	}
	prune(t, c, clk.Now().Add(-time.Minute), 1)
	if got := get(t, c, idx, k1, k2); len(got) != 1 || !reflect.DeepEqual(got[k1], v1) {
		t.Errorf("after Touch and Prune = %v, want only k1, unchanged", got)
	}
}

// touchIgnoresMissingKeys: touching a key the cache does not hold, or no key
// at all, is not an error and stores nothing.
func touchIgnoresMissingKeys(t *testing.T, c tagging.EmbeddingCache, clk *clock) {
	ctx := context.Background()
	for name, keys := range map[string][]tagging.CacheKey{"nil": nil, "empty": {}, "missing": {k1, k2}} {
		if err := c.Touch(ctx, idx, keys); err != nil {
			t.Errorf("Touch with %s keys: %v", name, err)
		}
	}
	if got := get(t, c, idx, k1, k2); len(got) != 0 {
		t.Errorf("Touch stored %v, want nothing", got)
	}
	prune(t, c, clk.Now().Add(time.Hour), 0)
}

// touchIsPerIndex: a Touch in one index leaves the same key of another alone.
func touchIsPerIndex(t *testing.T, c tagging.EmbeddingCache, clk *clock) {
	put(t, c, "a", map[tagging.CacheKey][]float32{k1: v1})
	put(t, c, "b", map[tagging.CacheKey][]float32{k1: v2})
	clk.advance(time.Hour)
	if err := c.Touch(context.Background(), "a", []tagging.CacheKey{k1}); err != nil {
		t.Fatalf("Touch: %v", err)
	}
	prune(t, c, clk.Now().Add(-time.Minute), 1)
	if got := get(t, c, "a", k1); !reflect.DeepEqual(got[k1], v1) {
		t.Errorf("index a = %v, want its touched entry", got)
	}
	if got := get(t, c, "b", k1); len(got) != 0 {
		t.Errorf("index b = %v, want empty: nobody touched it", got)
	}
}

// duplicateKeysCountOnce: a key listed several times in one Get is one entry,
// and does not disturb the others.
func duplicateKeysCountOnce(t *testing.T, c tagging.EmbeddingCache) {
	put(t, c, idx, map[tagging.CacheKey][]float32{k1: v1, k2: v2})
	got := get(t, c, idx, k1, k2, k1, k1)
	if len(got) != 2 || !reflect.DeepEqual(got[k1], v1) || !reflect.DeepEqual(got[k2], v2) {
		t.Errorf("Get with a key listed three times = %v, want k1 and k2", got)
	}
	missing := tagging.KeyFor("m", langai.RoleQuery, "missing")
	if got := get(t, c, idx, missing, missing); len(got) != 0 {
		t.Errorf("Get with a missing key listed twice = %v, want empty", got)
	}
}

// largeBatchesRoundTrip: a batch bigger than any per-statement limit a SQL
// backend has still lands whole, and is touched whole.
func largeBatchesRoundTrip(t *testing.T, c tagging.EmbeddingCache, clk *clock) {
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
	clk.advance(time.Hour)
	if err := c.Touch(context.Background(), idx, keys); err != nil {
		t.Fatalf("Touch: %v", err)
	}
	prune(t, c, clk.Now().Add(-time.Minute), 0)
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

// overlappingConcurrentPutsAllLand: writers that put the same large set of keys
// at once, as two processes rebuilding one space do, all succeed, round after
// round: a SQL backend that locks the rows of a Put in no fixed order
// deadlocks here.
func overlappingConcurrentPutsAllLand(t *testing.T, c tagging.EmbeddingCache) {
	const rounds, writers, n = 5, 8, 200
	vecs := make(map[tagging.CacheKey][]float32, n)
	keys := make([]tagging.CacheKey, 0, n)
	for i := range n {
		k := tagging.KeyFor("m", langai.RoleQuery, strconv.Itoa(i))
		vecs[k] = []float32{float32(i), 1}
		keys = append(keys, k)
	}
	for round := range rounds {
		index := idx + strconv.Itoa(round)
		var wg sync.WaitGroup
		errs := make(chan error, writers)
		for range writers {
			wg.Go(func() { errs <- c.Put(context.Background(), index, vecs) })
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Fatalf("round %d: Put: %v", round, err)
			}
		}
		got := get(t, c, index, keys...)
		if len(got) != n {
			t.Fatalf("round %d: got %d entries, want %d", round, len(got), n)
		}
		for k, want := range vecs {
			if !reflect.DeepEqual(got[k], want) {
				t.Fatalf("round %d: entry %x = %v, want %v", round, k.Hash[:4], got[k], want)
			}
		}
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

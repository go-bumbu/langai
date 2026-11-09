package sqlcache

import (
	"database/sql"
	"errors"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/tagging"
)

// TestNew_RejectsANilClock: a Cache without a clock would panic on its first
// Put or Get, so New refuses it.
func TestNew_RejectsANilClock(t *testing.T) {
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "c.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if _, err := New(t.Context(), db, SQLite, nil); err == nil {
		t.Fatal("New accepted a nil clock")
	}
}

// TestKeyOrder_IsThePrimaryKeyOrder: Put upserts by model, then role, then
// hash, whatever order the map gives, so overlapping Puts lock rows in one
// order (cachetest's OverlappingConcurrentPutsAllLand is the deadlock itself).
func TestKeyOrder_IsThePrimaryKeyOrder(t *testing.T) {
	vecs := map[tagging.CacheKey][]float32{}
	for _, model := range []string{"m2", "m1"} {
		for _, role := range []sense.Role{sense.RoleDocument, sense.RoleQuery} {
			for _, text := range []string{"a", "b", "c", "d"} {
				vecs[tagging.KeyFor(model, role, text)] = []float32{1}
			}
		}
	}
	got := keyOrder(vecs)
	if len(got) != len(vecs) {
		t.Fatalf("keyOrder returned %d keys, want %d", len(got), len(vecs))
	}
	for i := 1; i < len(got); i++ {
		a, b := got[i-1], got[i]
		switch {
		case a.Model != b.Model:
			if a.Model > b.Model {
				t.Fatalf("key %d: model %q after %q", i, b.Model, a.Model)
			}
		case a.Role != b.Role:
			if a.Role > b.Role {
				t.Fatalf("key %d: role %v after %v", i, b.Role, a.Role)
			}
		case string(a.Hash[:]) >= string(b.Hash[:]):
			t.Fatalf("key %d: hash %x after %x", i, b.Hash[:4], a.Hash[:4])
		}
	}
}

// TestCache_GetOnAReadOnlyDatabaseStillReturnsTheVectors: on a read-only
// replica, or a database another process holds locked, the vectors can be
// read while the last-used time of a stale one cannot be rewritten. That
// write is housekeeping: Get returns what it read together with an error
// wrapping tagging.ErrCacheWrite, so its caller can use the vectors and report
// the rest.
func TestCache_GetOnAReadOnlyDatabaseStillReturnsTheVectors(t *testing.T) {
	ctx := t.Context()
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "c.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	db.SetMaxOpenConns(1) // the pragma below is per connection
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	c, err := New(ctx, db, SQLite, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	stale := tagging.KeyFor("m", sense.RoleQuery, "stale")
	other := tagging.KeyFor("m2", sense.RoleDocument, "stale too")
	want := map[tagging.CacheKey][]float32{stale: {1, 2}, other: {3}}
	if err := c.Put(ctx, "i", want); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `PRAGMA query_only = ON`); err != nil {
		t.Fatal(err)
	}

	got, err := c.Get(ctx, "i", []tagging.CacheKey{stale, other})
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("Get of fresh entries = %v, %v; want %v, nil: nothing needs writing", got, err, want)
	}
	now = now.Add(tagging.CacheRefreshAfter + time.Hour)
	got, err = c.Get(ctx, "i", []tagging.CacheKey{stale, other})
	if !errors.Is(err, tagging.ErrCacheWrite) {
		t.Fatalf("Get of stale entries: err = %v, want one wrapping ErrCacheWrite", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Get of stale entries = %v, want %v next to the error", got, want)
	}
	if err := c.Touch(ctx, "i", []tagging.CacheKey{stale}); err == nil || errors.Is(err, tagging.ErrCacheWrite) {
		t.Errorf("Touch on a read-only database: err = %v, want a plain failure", err)
	}
}

// TestCache_GetRefreshesOnlyStaleEntries: a Get rewrites the last-used time
// only once it is CacheRefreshAfter old, so reads do not turn into writes.
func TestCache_GetRefreshesOnlyStaleEntries(t *testing.T) {
	ctx := t.Context()
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	k := tagging.KeyFor("m", sense.RoleQuery, "x")

	for _, tc := range []struct {
		name     string
		getAt    time.Duration
		pruneAt  time.Duration
		wantGone int64
	}{
		{"fresh entry is not refreshed", time.Hour, 30 * time.Minute, 1},
		{"stale entry is refreshed", 25 * time.Hour, 24 * time.Hour, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "c.db")+"?_busy_timeout=5000&_txlock=immediate")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = db.Close() })
			now := t0
			c, err := New(ctx, db, SQLite, func() time.Time { return now })
			if err != nil {
				t.Fatal(err)
			}
			if err := c.Put(ctx, "i", map[tagging.CacheKey][]float32{k: {1}}); err != nil {
				t.Fatal(err)
			}
			now = t0.Add(tc.getAt)
			if got, err := c.Get(ctx, "i", []tagging.CacheKey{k}); err != nil || len(got) != 1 {
				t.Fatalf("Get = %v, %v", got, err)
			}
			n, err := c.Prune(ctx, t0.Add(tc.pruneAt))
			if err != nil || n != tc.wantGone {
				t.Errorf("Prune = %d, %v; want %d, nil", n, err, tc.wantGone)
			}
		})
	}
}

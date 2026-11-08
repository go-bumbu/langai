package sqlcache

import (
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/tagging"
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
		for _, role := range []langai.Role{langai.RoleDocument, langai.RoleQuery} {
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

// TestCache_GetRefreshesOnlyStaleEntries: a Get rewrites the last-used time
// only once it is CacheRefreshAfter old, so reads do not turn into writes.
func TestCache_GetRefreshesOnlyStaleEntries(t *testing.T) {
	ctx := t.Context()
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	k := tagging.KeyFor("m", langai.RoleQuery, "x")

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

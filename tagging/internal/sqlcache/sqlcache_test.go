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

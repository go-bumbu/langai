package tagging

import (
	"context"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
)

// TestMemCache_GetRefreshesOnlyStaleEntries: a Get rewrites the last-used time
// only once it is CacheRefreshAfter old, so reads do not turn into writes.
func TestMemCache_GetRefreshesOnlyStaleEntries(t *testing.T) {
	ctx := context.Background()
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	k := KeyFor("m", langai.RoleQuery, "x")

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
			c := NewMemCache()
			now := t0
			c.now = func() time.Time { return now }
			if err := c.Put(ctx, "i", map[CacheKey][]float32{k: {1}}); err != nil {
				t.Fatal(err)
			}
			now = t0.Add(tc.getAt)
			if got, err := c.Get(ctx, "i", []CacheKey{k}); err != nil || len(got) != 1 {
				t.Fatalf("Get = %v, %v", got, err)
			}
			n, err := c.Prune(ctx, t0.Add(tc.pruneAt))
			if err != nil || n != tc.wantGone {
				t.Errorf("Prune = %d, %v; want %d, nil", n, err, tc.wantGone)
			}
		})
	}
}

package autotag

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/tagging"
)

// lockedBuffer is a log sink the test reads while KeepCachePruned writes.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// agedNotebook is a notebook whose cache holds, all last used 40 days ago,
// the embeddings of a tag, of a note that carries it by hand and of a note
// deleted since. Its log goes to the returned buffer.
func agedNotebook(t *testing.T) (*Notebook, *lockedBuffer) {
	t.Helper()
	ctx := context.Background()
	var today atomic.Bool
	cache := tagging.NewMemCacheWithClock(func() time.Time {
		if today.Load() {
			return time.Now()
		}
		return time.Now().Add(-40 * 24 * time.Hour)
	})
	logs := &lockedBuffer{}
	nb, err := New(ctx, Config{
		Cache:    cache,
		Tagging:  tagging.TaggerConfig{Model: "fake", NameMinScore: 0.2, MemberMinScore: 0.2},
		Chat:     &langaitest.FakeCompleter{},
		Embedder: langaitest.FakeEmbedder{},
		Logger:   slog.New(slog.NewTextHandler(logs, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = nb.Close() })
	bread, err := nb.AddTag(ctx, "cooking / bread", false)
	if err != nil {
		t.Fatal(err)
	}
	kept, err := nb.AddNote(ctx, "Sourdough", "bread dough with a starter, baked in a hot oven")
	if err != nil {
		t.Fatal(err)
	}
	if err := nb.TagNote(ctx, kept.ID, bread.ID); err != nil {
		t.Fatal(err)
	}
	gone, err := nb.AddNote(ctx, "Lisbon", "a city guide to trams and viewpoints")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nb.View(ctx); err != nil { // embeds the tag and both notes
		t.Fatal(err)
	}
	if err := nb.DeleteNote(ctx, gone.ID); err != nil {
		t.Fatal(err)
	}
	today.Store(true)
	return nb, logs
}

// TestNotebook_PruneCacheDropsOnlyWhatIsGone: a prune removes the embedding
// of the deleted note and keeps the tag's name and its member, although
// nothing read them for 40 days; a cutoff under tagging.MinPruneAge is
// refused.
func TestNotebook_PruneCacheDropsOnlyWhatIsGone(t *testing.T) {
	ctx := context.Background()
	nb, _ := agedNotebook(t)
	if _, err := nb.PruneCache(ctx, time.Now().Add(-tagging.MinPruneAge/2)); err == nil {
		t.Error("a cutoff younger than tagging.MinPruneAge was accepted")
	}
	cutoff := time.Now().Add(-30 * 24 * time.Hour)
	if removed, err := nb.PruneCache(ctx, cutoff); err != nil || removed != 1 {
		t.Fatalf("PruneCache = %d, %v; want 1, the deleted note's embedding", removed, err)
	}
	if removed, err := nb.PruneCache(ctx, cutoff); err != nil || removed != 0 {
		t.Errorf("a second PruneCache = %d, %v; want 0", removed, err)
	}
	if v, err := nb.View(ctx); err != nil || len(v.Notes) != 1 || v.Notes[0].Error != "" {
		t.Errorf("View after the prune = %+v, %v; want the one note left, scored", v.Notes, err)
	}
}

// TestNotebook_KeepCachePrunedRunsUntilItsContextEnds: it prunes at once and
// again at every interval, logs how many embeddings went, and returns when
// its context is cancelled.
func TestNotebook_KeepCachePrunedRunsUntilItsContextEnds(t *testing.T) {
	nb, logs := agedNotebook(t)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		nb.KeepCachePruned(ctx, 30*24*time.Hour, time.Millisecond)
	}()
	deadline := time.Now().Add(10 * time.Second)
	for strings.Count(logs.String(), "tag cache pruned") < 2 {
		if time.Now().After(deadline) {
			cancel()
			t.Fatalf("no second prune within 10 s; log:\n%s", logs.String())
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("KeepCachePruned did not return when its context was cancelled")
	}
	if out := logs.String(); !strings.Contains(out, "removed=1") || strings.Contains(out, "not pruned") {
		t.Errorf("want the first run to log removed=1 and none to fail; log:\n%s", out)
	}
}

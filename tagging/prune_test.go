package tagging

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
)

const day = 24 * time.Hour

// pruneRig is a rig whose engine and MemCache run on one clock the test moves.
type pruneRig struct {
	*rig
	now *time.Time
}

// newPruneRig returns a rig over cat with a cache and an engine on a clock
// that starts at a fixed time.
func newPruneRig(t *testing.T, cfg TaggerConfig, cat *fakeCatalog) pruneRig {
	t.Helper()
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	clock := func() time.Time { return now }
	r := newRig(t, cfg, cat, NewMemCacheWithClock(clock), nil)
	r.e.now = clock
	return pruneRig{rig: r, now: &now}
}

// restarted returns a rig over the same catalog and cache with a new, empty
// index, as the process is after a restart.
func (p pruneRig) restarted(t *testing.T, cfg TaggerConfig) *rig {
	t.Helper()
	return newRig(t, cfg, p.cat, p.cache, nil)
}

// TestEngine_PruneKeepsWhatTheCatalogStillUses: forty days at one revision,
// nothing rebuilds a tag, so no Get reads a name or a member again and their
// last-used times stay forty days old. A prune of what was unused for thirty
// days must still keep them, and remove the text nobody asked about again:
// after it, a restart embeds nothing.
func TestEngine_PruneKeepsWhatTheCatalogStillUses(t *testing.T) {
	for _, chunkBytes := range []int{0, 20} {
		cfg := engineCfg
		cfg.ChunkBytes = chunkBytes
		p := newPruneRig(t, cfg, breadWithRecipes("1"))
		p.scores(t, "1", croissantText)
		p.scores(t, "1", "an item nobody asks about again")
		for range 40 {
			*p.now = p.now.Add(day)
			p.scores(t, "1", croissantText)
		}
		gone := int64(len(itemPieces("an item nobody asks about again", chunkBytes)))
		n, err := p.e.prune(context.Background(), p.now.Add(-30*day), []Space{{"1", "10"}})
		if err != nil || n != gone {
			t.Fatalf("ChunkBytes %d: prune = %d, %v; want %d, nil: only the item nobody asked about", chunkBytes, n, err, gone)
		}
		b := p.restarted(t, cfg)
		b.scores(t, "1", croissantText)
		if n := b.emb.total(); n != 0 {
			t.Errorf("ChunkBytes %d: after the prune a restart embedded %d texts, want 0", chunkBytes, n)
		}
	}
}

// TestEngine_PruneDropsWhatNoListedSpaceUses: the embeddings of a space that is
// not passed are unused as far as the prune knows, and go once they are old.
func TestEngine_PruneDropsWhatNoListedSpaceUses(t *testing.T) {
	p := newPruneRig(t, engineCfg, breadWithRecipes("1"))
	p.scores(t, "1", croissantText)
	*p.now = p.now.Add(40 * day)
	want := int64(len(croissantRecipes) + 2 + 1) // the members, the two names and the item
	if n, err := p.e.prune(context.Background(), p.now.Add(-30*day), nil); err != nil || n != want {
		t.Fatalf("prune = %d, %v; want %d, nil", n, err, want)
	}
}

// TestEngine_PruneRefusesARecentCutoff: a Get rewrites a last-used time only
// once it is CacheRefreshAfter old, so an entry read a minute ago may still
// say "yesterday": a cutoff of age A only keeps what was read in the last A -
// CacheRefreshAfter. One less than twice CacheRefreshAfter old would protect
// less than a day of reads, at exactly CacheRefreshAfter none at all, and is
// refused.
func TestEngine_PruneRefusesARecentCutoff(t *testing.T) {
	p := newPruneRig(t, engineCfg, breadWithRecipes("1"))
	p.scores(t, "1", croissantText)
	*p.now = p.now.Add(23 * time.Hour)
	p.scores(t, "1", croissantText) // read now; its last-used time stays 23 hours old
	tags, members := p.cat.counts()
	for _, cutoff := range []time.Time{
		p.now.Add(-time.Hour), *p.now, p.now.Add(time.Hour),
		p.now.Add(-CacheRefreshAfter + time.Second), p.now.Add(-CacheRefreshAfter),
		p.now.Add(-2*CacheRefreshAfter + time.Second),
	} {
		n, err := p.e.prune(context.Background(), cutoff, []Space{{"1", "10"}})
		if err == nil || !strings.Contains(err.Error(), "CacheRefreshAfter") || n != 0 {
			t.Errorf("prune at now%+v = %d, %v; want 0 and an error naming CacheRefreshAfter", cutoff.Sub(*p.now), n, err)
		}
	}
	if t2, m2 := p.cat.counts(); t2 != tags || m2 != members {
		t.Errorf("a refused prune made %d Tags and %d Members calls, want none", t2-tags, m2-members)
	}
	b := p.restarted(t, engineCfg)
	b.scores(t, "1", croissantText)
	if n := b.emb.total(); n != 0 {
		t.Errorf("a refused prune removed embeddings: a restart embedded %d texts", n)
	}
	if n, err := p.e.prune(context.Background(), p.now.Add(-2*CacheRefreshAfter), []Space{{"1", "10"}}); err != nil || n != 0 {
		t.Errorf("prune at exactly twice CacheRefreshAfter ago = %d, %v; want 0, nil", n, err)
	}
}

// TestEngine_PruneKeepsWhatWasReadWithinItsGuarantee: an item's text is
// stored, and read again 23 hours and 59 minutes later, which does not rewrite
// its last-used time. A prune two minutes after that with a cutoff a day old
// used to be accepted, and removed the entry just read. A cutoff of age A
// keeps what was read in the last A - CacheRefreshAfter, whatever the
// last-used time says, and removes only what was not.
func TestEngine_PruneKeepsWhatWasReadWithinItsGuarantee(t *testing.T) {
	const item = "an item the application asks about now and then"
	const readAt = 23*time.Hour + 59*time.Minute
	for _, tc := range []struct {
		name    string
		pruneAt time.Duration // when the prune runs, after the item was stored
		age     time.Duration // how old its cutoff is
		refused bool
		kept    bool
	}{
		{"a cutoff a day old, two minutes after the read", readAt + 2*time.Minute, CacheRefreshAfter, true, true},
		{"two days old, two minutes after the read", readAt + 2*time.Minute, 2 * CacheRefreshAfter, false, true},
		{"two days old, a day after the read", readAt + CacheRefreshAfter, 2 * CacheRefreshAfter, false, true},
		{"two days old, a day and two minutes after the read", readAt + CacheRefreshAfter + 2*time.Minute, 2 * CacheRefreshAfter, false, false},
		{"thirty days old, twenty-nine days after the read", readAt + 29*day, 30 * day, false, true},
		{"thirty days old, twenty-nine days and two minutes after the read", readAt + 29*day + 2*time.Minute, 30 * day, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newPruneRig(t, engineCfg, breadWithRecipes("1"))
			stored := *p.now
			p.scores(t, "1", item)
			*p.now = stored.Add(readAt)
			p.scores(t, "1", item)
			*p.now = stored.Add(tc.pruneAt)
			n, err := p.e.prune(context.Background(), p.now.Add(-tc.age), []Space{{"1", "10"}})
			if (err != nil) != tc.refused {
				t.Fatalf("prune = %d, %v; want refused: %v", n, err, tc.refused)
			}
			// the guarantee: read within the cutoff's age less CacheRefreshAfter
			if guaranteed := tc.pruneAt-readAt <= tc.age-CacheRefreshAfter; guaranteed && !tc.kept {
				t.Fatalf("the case expects an entry read %v ago gone at a cutoff %v old", tc.pruneAt-readAt, tc.age)
			}
			if got := cached(t, p.cache, item)[item]; got != tc.kept {
				t.Errorf("after the prune the item's embedding is kept: %v, want %v (removed %d)", got, tc.kept, n)
			}
		})
	}
}

// TestEngine_PruneFailsWhenASpaceCannotBeRead: with a space's catalog down the
// prune cannot know what the space uses, so it removes nothing.
func TestEngine_PruneFailsWhenASpaceCannotBeRead(t *testing.T) {
	p := newPruneRig(t, engineCfg, breadWithRecipes("1"))
	p.scores(t, "1", croissantText)
	*p.now = p.now.Add(40 * day)
	p.cat.setErr(errBoom)
	if n, err := p.e.prune(context.Background(), p.now.Add(-30*day), []Space{{"1", "10"}}); err == nil || n != 0 {
		t.Fatalf("prune = %d, %v; want 0 and the catalog's error", n, err)
	}
	p.cat.setErr(nil)
	b := p.restarted(t, engineCfg)
	b.scores(t, "1", croissantText)
	if n := b.emb.total(); n != 0 {
		t.Errorf("a failed prune removed embeddings: a restart embedded %d texts", n)
	}
}

// TestTagger_PruneAndTouch: Tagger.Prune and Scope.Touch are the engine's, and
// a Touch alone keeps a space's embeddings through a prune that does not list
// the space.
func TestTagger_PruneAndTouch(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	clock := func() time.Time { return now }
	cache := NewMemCacheWithClock(clock)
	tg, err := NewTagger(&tallyEmbedder{}, breadWithRecipes("1"), cache, engineCfg)
	if err != nil {
		t.Fatal(err)
	}
	tg.e.now = clock
	ctx := context.Background()
	space := tg.Scope("1", "10")
	if _, err := space.Suggest(ctx, "1", croissantText); err != nil {
		t.Fatal(err)
	}
	now = now.Add(40 * day)
	if err := space.Touch(ctx); err != nil {
		t.Fatal(err)
	}
	if n, err := tg.Prune(ctx, now.Add(-30*day)); err != nil || n != 1 {
		t.Fatalf("Prune after Touch = %d, %v; want 1, nil: the item's text only", n, err)
	}
	now = now.Add(40 * day)
	if n, err := tg.Prune(ctx, now.Add(-30*day), Space{Index: "1", Partition: "10"}); err != nil || n != 0 {
		t.Fatalf("Prune of the listed space = %d, %v; want 0, nil", n, err)
	}
	if err := tg.Scope("", "10").Touch(ctx); err == nil {
		t.Error("Touch with an empty index succeeded")
	}
	got, err := cache.Get(ctx, "1", []CacheKey{KeyFor("fake", langai.RoleDocument, "cooking / bread")})
	if err != nil || len(got) != 1 {
		t.Errorf("the tag's name is gone from the cache: %v, %v", got, err)
	}
}

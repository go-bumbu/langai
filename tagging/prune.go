package tagging

import (
	"context"
	"fmt"
	"time"

	"github.com/go-bumbu/langai"
)

// membersGroupSize is how many tags one Catalog.Members call asks about, so
// the member texts of a space with thousands of tags are never all in memory
// at once.
const membersGroupSize = 64

// Space names one space of tags: an index and a partition within it, as
// Tagger.Scope takes them.
type Space struct{ Index, Partition string }

// touch sets the last-used time of every embedding the space's tags are built
// from to now: the name of each tag the catalog lists and the pieces of each
// of its members, under the engine's model. Nothing is embedded, and what the
// cache does not hold is not missed. The members are read membersGroupSize
// tags at a time.
func (e *engine) touch(ctx context.Context, index, partition string) error {
	if index == "" {
		return errEmptyIndex
	}
	tags, err := e.cat.Tags(ctx, index, partition)
	if err != nil {
		return fmt.Errorf("tagging: read the tags of space (%q, %q): %w", index, partition, err)
	}
	ids := make([]string, len(tags))
	keys := make([]CacheKey, len(tags))
	for i, t := range tags {
		ids[i] = t.ID
		keys[i] = KeyFor(e.cfg.Model, langai.RoleDocument, t.Name)
	}
	if err := e.cache.Touch(ctx, index, keys); err != nil {
		return err
	}
	for start := 0; start < len(ids); start += membersGroupSize {
		group := ids[start:min(start+membersGroupSize, len(ids))]
		got, err := e.cat.Members(ctx, index, partition, group)
		if err != nil {
			return fmt.Errorf("tagging: read the members of space (%q, %q): %w", index, partition, err)
		}
		keys = keys[:0]
		for _, id := range group {
			for _, text := range nonBlank(got[id]) {
				for _, piece := range itemPieces(text, e.cfg.ChunkBytes) {
					keys = append(keys, KeyFor(e.cfg.Model, langai.RoleQuery, piece))
				}
			}
		}
		if err := e.cache.Touch(ctx, index, keys); err != nil {
			return err
		}
	}
	return nil
}

// prune is Tagger.Prune: it refuses a cutoff less than CacheRefreshAfter old
// before it reads anything, touches each of spaces, and only when all of them
// were touched removes the cache's entries last used before unusedSince.
func (e *engine) prune(ctx context.Context, unusedSince time.Time, spaces []Space) (int64, error) {
	if age := e.now().Sub(unusedSince); age < CacheRefreshAfter {
		return 0, fmt.Errorf("tagging: prune cutoff is %s old, want at least CacheRefreshAfter (%s): "+
			"a read rewrites a last-used time only once it is that stale, so the entries read since would be removed",
			age.Round(time.Second), CacheRefreshAfter)
	}
	for _, s := range spaces {
		if err := e.touch(ctx, s.Index, s.Partition); err != nil {
			return 0, err
		}
	}
	return e.cache.Prune(ctx, unusedSince)
}

package tagging

import "context"

// Catalog is what the application provides: its own tags and hand-made links,
// read when a space's revision changes.
type Catalog interface {
	Tags(ctx context.Context, index, partition string) ([]CatalogTag, error)
	// Members returns, for each of tagIDs, the texts of the items tagged with
	// it by hand.
	Members(ctx context.Context, index, partition string, tagIDs []string) (map[string][]string, error)
}

// CatalogTag is a tag as the application reports it. Revision must change
// whenever the tag's member set does: a link added or removed, a linked
// item's text edited, a linked item deleted. Renames need no new revision.
type CatalogTag struct{ ID, Name, Revision string }

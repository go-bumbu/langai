package search

import (
	"context"
	"fmt"
	"iter"

	"github.com/go-bumbu/langai/vector"
)

// docsPage is how many documents Docs reads from the store per call.
const docsPage = 500

// Docs lists the indexed documents ordered by id (byte by byte), each with
// the hash it was stored with, an opaque fingerprint of what was indexed.
// It reads the store a page at a time while the loop runs, so it suits
// reconciling with the application's own records, e.g. deleting the
// documents removed while it was offline, or counting what is indexed. A
// store error ends the listing as its last pair.
func (ix *Index) Docs(ctx context.Context) iter.Seq2[vector.DocInfo, error] {
	return listDocs(ctx, ix.store, ix.coord)
}

// Delete removes the documents of the given ids from the index in one store
// call. Unknown ids are no error; an empty one is ErrEmptyID, one no store
// can keep is vector.ErrInvalidID, and then nothing is deleted.
func (ix *Index) Delete(ctx context.Context, ids ...string) error {
	return deleteDocs(ctx, ix.store, ix.coord, ids)
}

// Docs lists the indexed items as Index.Docs lists documents.
func (ix *MediaIndex) Docs(ctx context.Context) iter.Seq2[vector.DocInfo, error] {
	return listDocs(ctx, ix.store, ix.coord)
}

// Delete removes the items of the given ids and their frames, as
// Index.Delete does.
func (ix *MediaIndex) Delete(ctx context.Context, ids ...string) error {
	return deleteDocs(ctx, ix.store, ix.coord, ids)
}

// listDocs pages through the documents of c, docsPage per store call.
func listDocs(ctx context.Context, store vector.Store, c vector.Coord) iter.Seq2[vector.DocInfo, error] {
	return func(yield func(vector.DocInfo, error) bool) {
		after := ""
		for {
			page, err := store.Docs(ctx, c, after, docsPage)
			if err != nil {
				yield(vector.DocInfo{}, fmt.Errorf("search: list documents: %w", err))
				return
			}
			for _, d := range page {
				if !yield(d, nil) {
					return
				}
			}
			if len(page) < docsPage {
				return
			}
			after = page[len(page)-1].ID
		}
	}
}

// deleteDocs checks ids and deletes them from c in one call.
func deleteDocs(ctx context.Context, store vector.Store, c vector.Coord, ids []string) error {
	for _, id := range ids {
		if err := checkID(id); err != nil {
			return err
		}
	}
	if len(ids) == 0 {
		return nil
	}
	if err := store.Delete(ctx, c, ids...); err != nil {
		if len(ids) == 1 {
			return fmt.Errorf("search: delete %q: %w", ids[0], err)
		}
		return fmt.Errorf("search: delete %d documents: %w", len(ids), err)
	}
	return nil
}

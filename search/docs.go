package search

import (
	"context"
	"fmt"
	"iter"

	"github.com/go-bumbu/sense/vector"
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

// Reset removes every document of the index's collection (in its
// partition) in one store call. It is the way to a new embedding model: the
// store refuses another model's vectors while the collection holds
// documents (vector.ErrModelMismatch, which IndexBatch returns before the
// new model embeds anything), so after a model change Reset empties it and
// the next indexing run rebuilds it with the new model. A collection whose
// other partitions still hold documents keeps its model until those are
// emptied too.
func (ix *Index) Reset(ctx context.Context) error {
	return resetDocs(ctx, ix.store, ix.coord)
}

// Reset removes every item of the index's collection, as Index.Reset does.
func (ix *MediaIndex) Reset(ctx context.Context) error {
	return resetDocs(ctx, ix.store, ix.coord)
}

// resetDocs empties c.
func resetDocs(ctx context.Context, store vector.Store, c vector.Coord) error {
	if err := store.DeleteCollection(ctx, c); err != nil {
		return fmt.Errorf("search: reset: %w", err)
	}
	return nil
}

// checkModel fails with vector.ErrModelMismatch, before any model time is
// spent, when model is not empty and c's collection holds documents of
// another model.
func checkModel(ctx context.Context, store vector.Store, c vector.Coord, model string) error {
	if model == "" {
		return nil
	}
	have, err := store.Model(ctx, c)
	if err != nil {
		return fmt.Errorf("search: read the collection's model: %w", err)
	}
	if have != "" && have != model {
		return fmt.Errorf("search: %w: collection %q holds %q, not %q; Reset the index to rebuild it with the new model", vector.ErrModelMismatch, c.Collection, have, model)
	}
	return nil
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

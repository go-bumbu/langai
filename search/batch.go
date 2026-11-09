package search

import (
	"context"
	"fmt"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/vector"
)

// embedBatch is the most chunk texts IndexBatch hands the embedder in one
// call, and about the most it writes in one Upsert: a local model embeds a
// batch at a time and a remote one caps the inputs per request, so neither
// gets a whole corpus at once.
const embedBatch = 32

// Doc is one document handed to IndexBatch: its id and its text.
type Doc struct {
	ID   string
	Text string
}

// BatchResult says what IndexBatch did with the documents it was handed,
// by id. A document in none of the three was not reached: the call stopped
// before it.
type BatchResult struct {
	// Indexed are the documents chunked, embedded and stored.
	Indexed []string
	// Unchanged are the documents already stored with the same text and
	// settings: nothing was done for them.
	Unchanged []string
	// Failed are the documents refused for a reason of their own, such as
	// an id no store can keep (vector.ErrInvalidID) or a chunk that embeds
	// to the zero vector, with the error naming each; whatever was stored
	// for them before is kept.
	Failed map[string]error
}

// IndexBatch indexes docs as Index indexes one, in bounded batches: one
// Hashes read for all of them, the chunks of the changed ones embedded at
// most embedBatch per call, and one Upsert per group of documents whose
// chunks fill about one such call (a document of more chunks is a group of
// its own, embedded in several calls). Unchanged documents cost only their
// share of the hash read, so a full re-index at startup is cheap when little
// changed.
//
// Documents are processed in order. A document refused for a reason of its
// own goes to Failed and the others go on. An error from the embedder or
// the store stops the call and is returned with the result so far: the
// documents in Indexed were stored before it, the group it happened in and
// those after it were not touched, and calling again with the same docs
// picks up where it stopped. An empty id (ErrEmptyID) or one repeated fails
// the call before anything is read or written; an id that is not valid
// UTF-8 or holds a NUL byte (a Latin-1 file path, say) fails only its
// document, with vector.ErrInvalidID, before any model time is spent on it.
// When the collection holds another model's vectors the call fails with
// vector.ErrModelMismatch before the first changed document is embedded
// (see Reset).
func (ix *Index) IndexBatch(ctx context.Context, docs []Doc) (BatchResult, error) {
	var res BatchResult
	ids := make([]string, 0, len(docs))
	seen := make(map[string]bool, len(docs))
	for _, d := range docs {
		if d.ID == "" {
			return res, ErrEmptyID
		}
		if seen[d.ID] {
			return res, fmt.Errorf("search: document %q appears twice in the batch", d.ID)
		}
		seen[d.ID] = true
		if err := vector.CheckID(d.ID); err != nil {
			res.fail(d.ID, fmt.Errorf("search: %w", err))
			continue
		}
		ids = append(ids, d.ID)
	}
	if len(ids) == 0 {
		return res, nil
	}
	stored, err := ix.store.Hashes(ctx, ix.coord, ids...)
	if err != nil {
		return res, fmt.Errorf("search: read content hashes: %w", err)
	}
	var group []pendingDoc
	size := 0
	checked := false // the collection's model, before the first embedding
	for _, d := range docs {
		if res.Failed[d.ID] != nil {
			continue
		}
		text := sanitize(d.Text)
		hash := ix.hash(text)
		if h, ok := stored[d.ID]; ok && h == hash {
			res.Unchanged = append(res.Unchanged, d.ID)
			continue
		}
		if !checked {
			if err := checkModel(ctx, ix.store, ix.coord, ix.model); err != nil {
				return res, err
			}
			checked = true
		}
		parts := ix.chunker.Chunk(text)
		group = append(group, pendingDoc{id: d.ID, hash: hash, parts: parts})
		size += max(1, len(parts))
		if size >= embedBatch {
			if err := ix.writeGroup(ctx, group, &res); err != nil {
				return res, err
			}
			group, size = group[:0], 0
		}
	}
	if err := ix.writeGroup(ctx, group, &res); err != nil {
		return res, err
	}
	return res, nil
}

// fail records a document refused for a reason of its own.
func (r *BatchResult) fail(id string, err error) {
	if r.Failed == nil {
		r.Failed = make(map[string]error)
	}
	r.Failed[id] = err
}

// pendingDoc is a changed document on its way to the store: its id, the
// hash to store and its chunk texts.
type pendingDoc struct {
	id    string
	hash  string
	parts []string
}

// writeGroup embeds the chunks of group, at most embedBatch per call, and
// stores the documents that embedded cleanly in one Upsert, recording each
// in res.
func (ix *Index) writeGroup(ctx context.Context, group []pendingDoc, res *BatchResult) error {
	if len(group) == 0 {
		return nil
	}
	var texts []string
	for _, p := range group {
		texts = append(texts, p.parts...)
	}
	vectors := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += embedBatch {
		batch := texts[start:min(start+embedBatch, len(texts))]
		got, err := ix.embedder.Embed(ctx, langai.RoleDocument, batch)
		if err != nil {
			return fmt.Errorf("search: embed documents: %w", err)
		}
		if len(got) != len(batch) {
			return fmt.Errorf("search: embedder returned %d vectors for %d texts", len(got), len(batch))
		}
		vectors = append(vectors, got...)
	}
	docs := make([]vector.Doc, 0, len(group))
	ids := make([]string, 0, len(group))
	for _, p := range group {
		doc, err := ix.vectorDoc(p, vectors[:len(p.parts)])
		vectors = vectors[len(p.parts):]
		if err != nil {
			res.fail(p.id, err)
			continue
		}
		docs = append(docs, doc)
		ids = append(ids, p.id)
	}
	if len(docs) == 0 {
		return nil
	}
	if err := ix.store.Upsert(ctx, ix.coord, ix.model, docs...); err != nil {
		return fmt.Errorf("search: store documents: %w", err)
	}
	res.Indexed = append(res.Indexed, ids...)
	return nil
}

// vectorDoc is p with its chunk vectors as the store takes it, or the error
// naming the document and the chunk whose vector the store would refuse.
func (ix *Index) vectorDoc(p pendingDoc, vectors [][]float32) (vector.Doc, error) {
	chunks := make([]vector.Chunk, len(p.parts))
	for i, part := range p.parts {
		if err := vector.Validate(vectors[i], ix.embedder.Dimensions()); err != nil {
			return vector.Doc{}, fmt.Errorf("search: embed document %q chunk %d: %w", p.id, i, err)
		}
		chunks[i] = vector.Chunk{Seq: i, Text: part, Vector: vectors[i]}
	}
	return vector.Doc{ID: p.id, Hash: p.hash, Chunks: chunks}, nil
}

package main

import (
	"context"
	"time"

	"github.com/go-bumbu/langai/provider/yzma"
	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/zarf/examples/autotag"
	"github.com/go-bumbu/langai/zarf/examples/duplicates"
	"github.com/go-bumbu/langai/zarf/examples/mediasearch"
	"github.com/go-bumbu/langai/zarf/examples/status"
	"github.com/go-bumbu/langai/zarf/examples/textsearch"
)

// indexed is where a page's index keeps its vectors: the store, the
// collection's place in it, and a description of the store.
type indexed struct {
	store vector.Store
	coord vector.Coord
	where string
}

// service is everything the status page reports on.
type service struct {
	text  *textModel
	chat  *chatModel
	media *mediaModel

	lib      *mediasearch.Library
	libAt    indexed
	docs     *textsearch.Library
	docsAt   indexed
	dups     *duplicates.Finder
	notebook *autotag.Notebook
	stores   *stores
}

// openStatus builds the status page over the models, the pages' indexes and
// the tagging cache.
func openStatus(o options, s service) (*status.Service, error) {
	llama := "llama.cpp " + yzma.LibTag + " (provider/yzma)"
	media := status.Model{
		Key: "media", Name: "image model", UsedBy: "media search", ID: s.media.id, Dimensions: s.media.embedder.Dimensions(),
		Runtime: s.media.runtime, RunsOn: s.media.runsOn.String,
		Why: "behind langai.LazyImageEmbedder: the scan of an unchanged library and every \"looks like\" search use stored vectors, so the model loads only for a new file, an upload or a re-embed",
	}
	if s.media.lazy != nil {
		media.Lazy = s.media.lazy
	} else {
		media.Name, media.Fixed = "vision model", "loaded at startup"
		media.Why = "not lazy: it is the index's image, video and text embedder in one object, the library has no lazy loader for a VideoEmbedder, and the fingerprint the index hashes with every video exists on the loaded model only"
	}
	return status.New(status.Config{
		Pages: []status.Page{
			{Path: "mediasearch/", Name: "Media search", Shows: "Images and videos that look like an image, a video or, with the vision model, a text: search.MediaIndex, with the score, the matching frame and the timing of every hit."},
			{Path: "textsearch/", Name: "Text search", Shows: "Documents that answer a question, in any language: search.Index, with the chunk that matched, the chunks that were embedded, and what the last batch did."},
			{Path: "duplicates/", Name: "Duplicates", Shows: "Exact and near-exact copies of pictures by perceptual hash, no model: provider/phash, as raw hashes with their Hamming distances and as a MediaIndex."},
			{Path: "autotag/", Name: "Auto tagging", Shows: "Tags for notes, picked among your own (tagging.Tagger) or written new by a chat model (tagging.Generator), with every score and cutoff."},
		},
		Setup: []status.Fact{
			{Name: "stores", Value: s.stores.backend},
			{Name: "data", Value: o.dataDir},
			{Name: "media", Value: o.mediaDir},
			{Name: "documents", Value: o.docsDir},
		},
		Models: []status.Model{
			{
				Key: "text", Name: "text embedding model", UsedBy: "text search; auto tagging (the Tagger, and the scores of the Generator's tags)",
				ID: s.text.Model(), Dimensions: s.text.Dimensions(), Runtime: llama, RunsOn: s.text.runsOn.String, Lazy: s.text.LazyEmbedder,
				Why: "behind langai.LazyEmbedder: one instance for both pages; unchanged documents and cached notes never reach it, so it loads on the first new text or query",
			},
			{
				Key: "chat", Name: "chat model", UsedBy: "auto tagging (the Generator writes tags with it)",
				ID: s.chat.name, Runtime: llama, RunsOn: func() string { return s.chat.accel + " (as asked; a yzma.Completer does not report its device)" }, Lazy: s.chat.LazyCompleter,
				Why: "behind langai.LazyCompleter: 2.5 GB that only the recommend button needs",
			},
			media,
			{
				Key: "phash", Name: "perceptual hashes", UsedBy: "duplicates", ID: "phash-64, dhash-64, phash-256, dhash-256",
				Runtime: "provider/phash, pure Go", RunsOn: func() string { return "cpu" }, Fixed: "no model",
				Why: "nothing to load: a hash is a few milliseconds of integer arithmetic",
			},
		},
		Collections: s.collections,
		Cache: &status.Cache{
			Name: "embedding cache of the Tagger", Backend: s.stores.cacheAt, Size: s.stores.cacheBytes,
			LastPrune: func() status.Prune { return pruneOf(s.notebook.LastPrune()) },
			PruneNow: func(ctx context.Context) (status.Prune, error) {
				_, err := s.notebook.PruneCache(ctx, time.Now().Add(-pruneUnusedFor))
				return pruneOf(s.notebook.LastPrune()), err
			},
		},
	})
}

// collections lists the pages' collections as they are now: the duplicates
// page opens one per hash, when a search first asks for it.
func (s service) collections() []status.Collection {
	cols := []status.Collection{
		{
			Key: "media", Name: "media", Page: "mediasearch/", Store: s.libAt.store, Coord: s.libAt.coord, Backend: s.libAt.where,
			Model: s.media.id, Reset: s.lib.Reset,
		},
		{
			Key: "docs", Name: "documents", Page: "textsearch/", Store: s.docsAt.store, Coord: s.docsAt.coord, Backend: s.docsAt.where,
			Model: s.text.Model(), Reset: s.docs.Reset,
		},
	}
	for _, c := range s.dups.Collections() {
		cols = append(cols, status.Collection{
			Key: "duplicates-" + c.Kind, Name: "duplicates · " + c.Kind, Page: "duplicates/", Store: c.Store, Coord: c.Coord,
			Backend: "vector.MemStore · in memory, rebuilt by the first search after a start", Model: c.Kind,
			Reset: func(ctx context.Context) error { return s.dups.Reset(ctx, c.Kind) },
		})
	}
	return cols
}

func pruneOf(p autotag.PruneResult) status.Prune {
	out := status.Prune{At: p.At, Removed: p.Removed, Err: p.Err}
	if !p.At.IsZero() {
		out.UnusedFor = p.At.Sub(p.UnusedSince).Round(time.Hour)
	}
	return out
}

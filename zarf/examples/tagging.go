package main

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/go-bumbu/sense/tagging"
	"github.com/go-bumbu/sense/zarf/examples/autotag"
)

// chunkBytes is the most of a note the Tagger embeds at once. EmbeddingGemma
// reads 512 tokens and drops the rest; measured on a 4.9 KB English note, a
// 2,500-byte prefix already embeds as the whole note, so 2,000 bytes stay
// inside the window with room for denser text.
const chunkBytes = 2000

// The embedding cache is pruned every pruneEvery of what was not used for
// pruneUnusedFor: well over tagging.MinPruneAge (two days), under which
// Tagger.Prune refuses a cutoff. What the notebook still holds is read at
// startup and by every page view, so only the embeddings of texts that are
// gone, or of another model, get that old.
const (
	pruneUnusedFor = 30 * 24 * time.Hour
	pruneEvery     = 24 * time.Hour
)

// openTagging builds the auto tagging page over the two lazy models: the
// text embedder loads on the first text the cache does not hold, the chat
// model on the first recommendation. The notebook (notes, tags, hand tags)
// is notebook.json, and the Tagger reads them from it; all the Tagger keeps
// is its embedding cache, one for every model (the model is part of every
// key), which refills itself when deleted. The Tagger's Model is the lazy
// embedder's, the identity the loaded model will report, so what one run
// cached the next finds without loading anything.
func openTagging(ctx context.Context, o options, text *textModel, chat *chatModel, cache tagging.EmbeddingCache, logger *slog.Logger) (*autotag.Notebook, func(), error) {
	samples := o.samples
	if _, err := os.Stat(samples); samples != "" && err != nil {
		logger.Info("no samples for the auto tagging page", "file", samples, "err", err)
		samples = ""
	}
	nb, err := autotag.New(ctx, autotag.Config{
		Cache:         cache,
		Tagging:       tagging.TaggerConfig{Model: text.Model(), ChunkBytes: chunkBytes, Logger: logger},
		Chat:          chat,
		Embedder:      text,
		File:          filepath.Join(o.dataDir, "notebook.json"),
		SamplesFile:   samples,
		ChatState:     func() string { return state(chat) },
		EmbedderState: func() string { return state(text) },
		RunsOn:        text.runsOn.String,
		// the warm-up runs behind the server, with the first scans
		WarmLater: true,
		Info: autotag.Info{
			Embedder:       text.name,
			Completer:      chat.name,
			NameMinScore:   tagging.DefaultNameMinScore,
			MemberMinScore: tagging.DefaultMemberMinScore,
			MinScore:       o.tagScore,
			ChunkBytes:     chunkBytes,
		},
		Logger: logger,
	})
	if err != nil {
		return nil, nil, err
	}
	// the cache only grows on its own: every edited note leaves the embedding
	// of its old text behind. Prune it at startup and once a day after that.
	pruneCtx, stopPruning := context.WithCancel(ctx)
	pruned := make(chan struct{})
	go func() {
		defer close(pruned)
		nb.KeepCachePruned(pruneCtx, pruneUnusedFor, pruneEvery)
	}()
	// the pruning first, then the Tagger, which ends its rebuilds before the
	// cache and the models go
	return nb, func() { stopPruning(); <-pruned; _ = nb.Close() }, nil
}

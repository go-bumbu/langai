package main

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-bumbu/langai/provider/yzma"
	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/tagging/sqlitecache"
	"github.com/go-bumbu/langai/zarf/examples/autotag"
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

// openTagging builds the auto tagging page: the text embedder now, the chat
// model on the first recommendation. The notebook (notes, tags, hand tags) is
// notebook.json, and the Tagger reads them from it; all the Tagger keeps is
// its embedding cache, autotag-cache.sqlite, one for every model (the model is
// part of every key), which refills itself when deleted.
func openTagging(ctx context.Context, cfg yzma.Config, dataDir, samples string, minScore float64, logger *slog.Logger) (*autotag.Notebook, func(), error) {
	logger.Info("loading the text embedding model", "model", yzma.DefaultModel, "accel", cfg.Accel, "dir", filepath.Dir(cfg.ModelsDir))
	emb, err := yzma.New(ctx, cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("text embedding model: %w", err)
	}
	chat := &lazyCompleter{load: func() (*yzma.Completer, error) {
		logger.Info("loading the chat model", "model", yzma.DefaultCompleterModel)
		return yzma.NewCompleter(ctx, cfg) // ctx bounds the download only
	}}
	chat.setState("not loaded")
	closeAll := func() { _ = chat.Close(); _ = emb.Close() }

	db, err := sql.Open("sqlite3", filepath.Join(dataDir, "autotag-cache.sqlite")+"?_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		closeAll()
		return nil, nil, fmt.Errorf("open tag cache database: %w", err)
	}
	fail := func(err error) (*autotag.Notebook, func(), error) {
		_ = db.Close()
		closeAll()
		return nil, nil, err
	}
	cache, err := sqlitecache.New(ctx, db)
	if err != nil {
		return fail(fmt.Errorf("tag cache: %w", err))
	}
	if _, err := os.Stat(samples); samples != "" && err != nil {
		logger.Info("no samples for the auto tagging page", "file", samples, "err", err)
		samples = ""
	}
	nb, err := autotag.New(ctx, autotag.Config{
		Cache:       cache,
		Tagging:     tagging.TaggerConfig{Model: emb.Model(), ChunkBytes: chunkBytes},
		Chat:        chat,
		Embedder:    emb,
		File:        filepath.Join(dataDir, "notebook.json"),
		SamplesFile: samples,
		ChatState:   chat.State,
		Info: autotag.Info{
			Embedder:       yzma.DefaultModel,
			Completer:      yzma.DefaultCompleterModel,
			Accel:          emb.Accel(),
			NameMinScore:   tagging.DefaultNameMinScore,
			MemberMinScore: tagging.DefaultMemberMinScore,
			MinScore:       minScore,
			ChunkBytes:     chunkBytes,
		},
		Logger: logger,
	})
	if err != nil {
		return fail(err)
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
	// cache and the model go
	return nb, func() { stopPruning(); <-pruned; _ = nb.Close(); _ = db.Close(); closeAll() }, nil
}

// lazyCompleter loads the chat model on the first Complete: it is a 2.5 GB
// download and several GB of memory that only the recommend button needs. A
// failed load is retried by the next call.
type lazyCompleter struct {
	load  func() (*yzma.Completer, error)
	mu    sync.Mutex
	c     *yzma.Completer
	state atomic.Pointer[string] // read by the page while a load holds mu
}

func (l *lazyCompleter) setState(s string) { l.state.Store(&s) }

// State says whether the model is loaded: "not loaded", "loading", "ready"
// or the last load error.
func (l *lazyCompleter) State() string { return *l.state.Load() }

// Complete implements langai.Completer.
func (l *lazyCompleter) Complete(ctx context.Context, system, user string) (string, error) {
	l.mu.Lock()
	if l.c == nil {
		l.setState("loading")
		c, err := l.load()
		if err != nil {
			l.setState("load failed: " + err.Error())
			l.mu.Unlock()
			return "", err
		}
		l.c = c
		l.setState("ready")
	}
	c := l.c
	l.mu.Unlock()
	return c.Complete(ctx, system, user)
}

// Close frees the model if it was loaded.
func (l *lazyCompleter) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.c == nil {
		return nil
	}
	err := l.c.Close()
	l.c = nil
	return err
}

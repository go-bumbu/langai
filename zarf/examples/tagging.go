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

	"github.com/go-bumbu/langai/zarf/examples/autotag"
	"github.com/go-bumbu/langai/provider/yzma"
	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/tagging/sqlitestore"
)

// chunkBytes is the most of a note the Tagger embeds at once. EmbeddingGemma
// reads 512 tokens and drops the rest; measured on a 4.9 KB English note, a
// 2,500-byte prefix already embeds as the whole note, so 2,000 bytes stay
// inside the window with room for denser text.
const chunkBytes = 2000

// openTagging builds the auto tagging page: the text embedder now, the chat
// model on the first recommendation. The notebook (notes, tags, hand tags) is
// notebook.json; the Tagger's store is derived from it, one per embedding
// model, and is rebuilt from the notebook when missing.
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

	db, err := sql.Open("sqlite3", filepath.Join(dataDir, "autotag-"+slug(yzma.DefaultModel)+".sqlite")+"?_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		closeAll()
		return nil, nil, fmt.Errorf("open tag database: %w", err)
	}
	fail := func(err error) (*autotag.Notebook, func(), error) {
		_ = db.Close()
		closeAll()
		return nil, nil, err
	}
	store, err := sqlitestore.New(ctx, db)
	if err != nil {
		return fail(fmt.Errorf("tag store: %w", err))
	}
	tagger, err := tagging.NewTagger(emb, store, tagging.TaggerConfig{Model: yzma.DefaultModel, ChunkBytes: chunkBytes})
	if err != nil {
		return fail(err)
	}
	if _, err := os.Stat(samples); samples != "" && err != nil {
		logger.Info("no samples for the auto tagging page", "file", samples, "err", err)
		samples = ""
	}
	nb, err := autotag.New(ctx, autotag.Config{
		Scope:       tagger.Scope("examples", "notebook"),
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
	return nb, func() { _ = db.Close(); closeAll() }, nil
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

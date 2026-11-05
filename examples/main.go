// Command examples is a single-user web service to exercise langai's
// features by hand: run it on a directory of media and click around. Each
// feature is a package with its own page, mounted here; mediasearch is the
// first. See README.md next to this file.
package main

import (
	"context"
	"database/sql"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai/examples/mediasearch"
	"github.com/go-bumbu/langai/media/ffmpeg"
	"github.com/go-bumbu/langai/provider/onnx"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector/sqlitevec"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		addr      = flag.String("addr", "127.0.0.1:8085", "listen address")
		mediaDir  = flag.String("media", "data/media", "directory scanned for images and videos")
		dataDir   = flag.String("data", "data", "where the index database, the catalog and the thumbnail cache go")
		modelsDir = flag.String("models", filepath.Join(homeDir(), ".cache", "semsearch-onnx"), "ONNX models and runtime (shared with make test-images)")
		model     = flag.String("model", onnx.DefaultModel, "image model: "+strings.Join(onnx.Models(), ", "))
		provider  = flag.String("provider", string(onnx.ProviderCPU), "what runs the model: cpu, cuda or coreml")
		limit     = flag.Int("limit", 12, "hits per search")
		minScore  = flag.Float64("min-score", 0, "drop hits scoring below this; 0 shows everything")
	)
	flag.Parse()
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	logger.Info("loading the image model", "model", *model, "provider", *provider, "dir", *modelsDir)
	emb, err := onnx.New(ctx, onnx.Config{
		ModelsDir:     filepath.Join(*modelsDir, "models"),
		Model:         *model,
		LibDir:        filepath.Join(*modelsDir, "onnxruntime", onnx.LibVersion),
		ManagedLibDir: true,
		Provider:      onnx.Provider(*provider),
		Logger:        logger,
	})
	if err != nil {
		return fmt.Errorf("image model: %w", err)
	}
	defer func() { _ = emb.Close() }()

	sampler, err := ffmpeg.New(ffmpeg.Config{})
	if err != nil {
		return fmt.Errorf("ffmpeg: %w", err)
	}

	// one database per model: the spaces never mix, whatever the dimensions
	if err := os.MkdirAll(*dataDir, 0o750); err != nil {
		return err
	}
	base := filepath.Join(*dataDir, "index-"+slug(*model))
	db, err := sql.Open("sqlite3", base+".sqlite?_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer func() { _ = db.Close() }()
	store, err := sqlitevec.New(ctx, db, emb.Dimensions())
	if err != nil {
		return fmt.Errorf("vector store: %w", err)
	}
	ix, err := search.NewMediaIndex(emb, store, "examples", "local", "media",
		search.WithSampler(sampler), search.WithMediaMinScore(*minScore))
	if err != nil {
		return err
	}
	lib, err := mediasearch.New(mediasearch.Config{
		Dir:         *mediaDir,
		Index:       ix,
		Sampler:     sampler,
		CatalogFile: base + ".catalog.json",
		CacheDir:    filepath.Join(*dataDir, "thumbs"),
		Limit:       *limit,
		Info:        mediasearch.Info{Model: *model, Dimensions: emb.Dimensions(), Provider: *provider, MinScore: *minScore},
		Logger:      logger,
	})
	if err != nil {
		return err
	}

	logger.Info("scanning", "dir", *mediaDir)
	if _, err := lib.Scan(ctx); err != nil {
		return fmt.Errorf("scan: %w", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", home)
	mux.Handle("/mediasearch/", http.StripPrefix("/mediasearch", mediasearch.Handler(lib)))
	srv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	logger.Info("serving", "url", "http://"+*addr+"/mediasearch/")
	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// home lists the features.
func home(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = fmt.Fprint(w, `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>langai examples</title>
<style>body{font:16px/1.5 system-ui,sans-serif;background:#15171a;color:#e6e8eb;max-width:640px;margin:48px auto;padding:0 16px}
a{color:#5aa9ff}li{margin:8px 0}</style></head><body>
<h1>langai examples</h1>
<ul>
<li><a href="/mediasearch/">Media search</a> — images and videos that look like an image or a video</li>
</ul>
</body></html>`)
}

func homeDir() string {
	h, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return h
}

// slug turns a model name into a file name: "google/siglip2-base-patch16-224"
// becomes "google-siglip2-base-patch16-224".
func slug(name string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '-' {
			return r
		}
		return '-'
	}, name)
}

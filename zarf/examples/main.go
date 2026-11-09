// Command examples is a single-user web service to exercise langai's
// features by hand: run it on a directory of media and click around. Each
// feature is a package with its own page, mounted here. See README.md next
// to this file.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai/media/ffmpeg"
	"github.com/go-bumbu/langai/provider/onnx"
	"github.com/go-bumbu/langai/provider/yzma"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/zarf/examples/assets"
	"github.com/go-bumbu/langai/zarf/examples/autotag"
	"github.com/go-bumbu/langai/zarf/examples/mediasearch"
)

// options are the command's flags.
type options struct {
	addr      string
	mediaDir  string
	dataDir   string
	modelsDir string
	llamaDir  string
	model     string
	provider  string
	accel     string
	segment   time.Duration
	limit     int
	minScore  float64
	samples   string
	tagScore  float64
}

func parseFlags() options {
	var o options
	flag.StringVar(&o.addr, "addr", "127.0.0.1:8085", "listen address")
	flag.StringVar(&o.mediaDir, "media", "data/media", "directory scanned for images and videos")
	flag.StringVar(&o.dataDir, "data", "data", "where the index databases, the catalogs and the thumbnail cache go")
	flag.StringVar(&o.modelsDir, "models", filepath.Join(homeDir(), ".cache", "semsearch-onnx"), "ONNX models and runtime (shared with make test-images)")
	flag.StringVar(&o.llamaDir, "llama", filepath.Join(homeDir(), ".cache", "semsearch-yzma"), "GGUF models and the llama.cpp build (shared with make test-vision)")
	flag.StringVar(&o.model, "model", onnx.DefaultModel, "media model: "+strings.Join(onnx.Models(), ", ")+" (images, videos as sampled frames) or qwen3vl ("+yzma.DefaultVisionModel+": text, images and videos as segments in one space)")
	flag.StringVar(&o.provider, "provider", string(onnx.ProviderCPU), "what runs an ONNX model: cpu, cuda or coreml")
	flag.StringVar(&o.accel, "accel", "", "what runs the GGUF models: auto (empty), cpu, vulkan, cuda or metal")
	flag.DurationVar(&o.segment, "segment", yzma.DefaultSegmentLength, "how much of a video goes through the vision model at once")
	flag.IntVar(&o.limit, "limit", 12, "hits per search")
	flag.Float64Var(&o.minScore, "min-score", 0, "drop media hits scoring below this; 0 shows everything")
	flag.StringVar(&o.samples, "samples", filepath.Join("..", "eval", "testdata", "golden.yaml"), "labelled texts the auto tagging page can import; empty or missing hides the button")
	flag.Float64Var(&o.tagScore, "tag-min-score", tagging.DefaultGeneratorMinScore, "the cutoff of the recommended tags; the page shows the ones under it greyed")
	flag.Parse()
	return o
}

func main() {
	if err := run(parseFlags()); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(o options) error {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	sampler, err := ffmpeg.New(ffmpeg.Config{})
	if err != nil {
		return fmt.Errorf("ffmpeg: %w", err)
	}
	// the llama.cpp models: the text embedder, the chat model and the
	// vision model share one runtime
	llama := yzma.Config{
		ModelsDir: filepath.Join(o.llamaDir, "models"),
		Runtime: yzma.Runtime{
			LibDir:        filepath.Join(o.llamaDir, "llama-cpp", yzma.LibTag),
			ManagedLibDir: true,
			Accel:         yzma.Accel(o.accel),
			Logger:        logger,
		},
	}

	// the models. Nothing loads here but the vision model: the others sit
	// behind the library's lazy loaders and load on first use
	text, err := newTextModel(llama, logger)
	if err != nil {
		return err
	}
	defer func() { _ = text.Close() }()
	chat, err := newChatModel(llama, logger)
	if err != nil {
		return err
	}
	defer func() { _ = chat.Close() }()
	media, err := newMediaModel(ctx, o, llama, sampler, logger)
	if err != nil {
		return err
	}
	defer func() { _ = media.Close() }()

	st, err := openStores(ctx, o)
	if err != nil {
		return err
	}
	defer st.Close()

	lib, err := openMedia(ctx, o, media, sampler, st, logger)
	if err != nil {
		return err
	}
	notebook, closeTagging, err := openTagging(ctx, o, text, chat, st.cache, logger)
	if err != nil {
		return err
	}
	defer closeTagging()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", home)
	mux.Handle("/assets/", http.StripPrefix("/assets", assets.Handler()))
	mux.Handle("/mediasearch/", http.StripPrefix("/mediasearch", mediasearch.Handler(lib)))
	mux.Handle("/autotag/", http.StripPrefix("/autotag", autotag.Handler(notebook)))

	// the first scans run behind the server, so the service is up at once
	// and a page shows its scan as it goes; what they need of a model loads
	// it. They end before anything they use is closed.
	var scans sync.WaitGroup
	scans.Go(func() {
		logger.Info("scanning", "dir", o.mediaDir)
		if _, err := lib.Scan(ctx); err != nil && ctx.Err() == nil {
			logger.Error("media scan", "err", err)
		}
	})
	defer func() { stop(); scans.Wait() }()
	return serve(ctx, o.addr, mux, logger)
}

// The coordinates of the pages' collections: one index, as a single-user
// service is one tenant.
const (
	indexName      = "examples"
	indexPartition = "local"
)

// openMedia builds the media search page: a MediaIndex over the media
// model in that model's own vector store.
func openMedia(ctx context.Context, o options, m *mediaModel, sampler *ffmpeg.Sampler, st *stores, logger *slog.Logger) (*mediasearch.Library, error) {
	store, _, base, err := st.vectors(ctx, m.name, m.embedder.Dimensions())
	if err != nil {
		return nil, err
	}
	coord := vector.Coord{Index: indexName, Partition: indexPartition, Collection: "media"}
	opts := append([]search.MediaOption{search.WithSampler(sampler), search.WithMediaMinScore(o.minScore)}, m.opts...)
	ix, err := search.NewMediaIndex(m.embedder, store, coord.Index, coord.Partition, coord.Collection, opts...)
	if err != nil {
		return nil, err
	}
	return mediasearch.New(mediasearch.Config{
		Dir:         o.mediaDir,
		Index:       ix,
		Sampler:     sampler,
		CatalogFile: base + ".catalog.json",
		CacheDir:    filepath.Join(o.dataDir, "thumbs"),
		Limit:       o.limit,
		Info:        m.info,
		ModelState:  m.state,
		Logger:      logger,
	})
}

// serve runs the HTTP server until ctx ends.
func serve(ctx context.Context, addr string, handler http.Handler, logger *slog.Logger) error {
	srv := &http.Server{Addr: addr, Handler: handler, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		// ctx has ended: the shutdown gets a few seconds of its own
		shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	logger.Info("serving", "url", "http://"+addr+"/")
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
<li><a href="/mediasearch/">Media search</a> — images and videos that look like an image, a video or a text</li>
<li><a href="/autotag/">Auto tagging</a> — tags for notes, picked among your own tags or written new by a chat model</li>
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

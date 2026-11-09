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

	"github.com/go-bumbu/langai/media/ffmpeg"
	"github.com/go-bumbu/langai/provider/onnx"
	"github.com/go-bumbu/langai/provider/yzma"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/zarf/examples/assets"
	"github.com/go-bumbu/langai/zarf/examples/autotag"
	"github.com/go-bumbu/langai/zarf/examples/duplicates"
	"github.com/go-bumbu/langai/zarf/examples/mediasearch"
	"github.com/go-bumbu/langai/zarf/examples/status"
	"github.com/go-bumbu/langai/zarf/examples/textsearch"
)

// options are the command's flags.
type options struct {
	addr      string
	mediaDir  string
	docsDir   string
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
	// textMinScore is the floor of the text search page.
	textMinScore float64
	// maxImageBytes is the most the media index reads of an image; 0 keeps
	// the library's default.
	maxImageBytes int64
	// postgres is the DSN of a Postgres database with pgvector to keep the
	// vectors and the tagging cache in; empty keeps them in SQLite files.
	postgres string
	// longSamples is the directory of the long sample documents, and
	// writeDocs, when set, makes the command write the sample documents
	// there and exit.
	longSamples string
	writeDocs   string
}

func parseFlags() options {
	var o options
	flag.StringVar(&o.addr, "addr", "127.0.0.1:8085", "listen address")
	flag.StringVar(&o.mediaDir, "media", "data/media", "directory scanned for images and videos")
	flag.StringVar(&o.docsDir, "docs", "data/docs", "directory scanned for the text search page's .md and .txt documents")
	flag.StringVar(&o.dataDir, "data", "data", "where the index databases, the catalogs and the thumbnail cache go")
	flag.StringVar(&o.modelsDir, "models", filepath.Join(homeDir(), ".cache", "semsearch-onnx"), "ONNX models and runtime (shared with make test-images)")
	flag.StringVar(&o.llamaDir, "llama", filepath.Join(homeDir(), ".cache", "semsearch-yzma"), "GGUF models and the llama.cpp build (shared with make test-vision)")
	flag.StringVar(&o.model, "model", onnx.DefaultModel, "media model: "+strings.Join(onnx.Models(), ", ")+" (images, videos as sampled frames) or qwen3vl ("+yzma.DefaultVisionModel+": text, images and videos as segments in one space)")
	flag.StringVar(&o.provider, "provider", string(onnx.ProviderCPU), "what runs an ONNX model: cpu, cuda or coreml")
	flag.StringVar(&o.accel, "accel", "", "what runs the GGUF models: auto (empty), cpu, vulkan, cuda or metal")
	flag.DurationVar(&o.segment, "segment", yzma.DefaultSegmentLength, "how much of a video goes through the vision model at once")
	flag.IntVar(&o.limit, "limit", 12, "hits per search")
	flag.Float64Var(&o.minScore, "min-score", 0, "drop media hits scoring below this; 0 shows everything")
	flag.StringVar(&o.samples, "samples", filepath.Join("..", "eval", "testdata", "golden.yaml"), "labelled texts: the auto tagging page imports them, and -write-docs writes them out as documents; empty or missing hides the button")
	flag.StringVar(&o.longSamples, "long-samples", filepath.Join("..", "eval", "testdata", "long"), "long sample documents -write-docs writes out too; empty leaves them out")
	flag.StringVar(&o.writeDocs, "write-docs", "", "write the sample documents into this directory as Markdown files, for the text search page, and exit (make prepare does)")
	flag.Float64Var(&o.textMinScore, "text-min-score", 0, "drop text search hits scoring below this; 0 shows everything")
	flag.Int64Var(&o.maxImageBytes, "max-image-bytes", 0, "the most the media index reads of an image; a longer one is listed as too large (0: the library's 64 MiB)")
	flag.Float64Var(&o.tagScore, "tag-min-score", tagging.DefaultGeneratorMinScore, "the cutoff of the recommended tags; the page shows the ones under it greyed")
	flag.StringVar(&o.postgres, "postgres", os.Getenv("POSTGRES"), "keep the vectors and the tagging cache in this Postgres database (with pgvector) instead of SQLite files, e.g. postgres://user:pass@localhost:5432/langai (default: $POSTGRES)")
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
	if o.writeDocs != "" {
		n, err := textsearch.WriteSamples(o.writeDocs, o.samples, o.longSamples)
		fmt.Printf("wrote %d sample documents into %s\n", n, o.writeDocs)
		return err
	}
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

	lib, libAt, err := openMedia(ctx, o, media, sampler, st, logger)
	if err != nil {
		return err
	}
	docs, docsAt, err := openText(ctx, o, text, st, logger)
	if err != nil {
		return err
	}
	// the duplicates page hashes the media directory's pictures: no model
	// and no database
	dups, err := duplicates.New(duplicates.Config{Dir: o.mediaDir, Limit: o.limit, Logger: logger})
	if err != nil {
		return err
	}
	notebook, closeTagging, err := openTagging(ctx, o, text, chat, st.cache, logger)
	if err != nil {
		return err
	}
	defer closeTagging()
	// the home page: what is loaded and how
	board, err := openStatus(o, service{
		text: text, chat: chat, media: media, lib: lib, libAt: libAt, docs: docs, docsAt: docsAt,
		dups: dups, notebook: notebook, stores: st,
	})
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.Handle("/", status.Handler(board))
	mux.Handle("/assets/", http.StripPrefix("/assets", assets.Handler()))
	mux.Handle("/mediasearch/", http.StripPrefix("/mediasearch", mediasearch.Handler(lib)))
	mux.Handle("/textsearch/", http.StripPrefix("/textsearch", textsearch.Handler(docs)))
	mux.Handle("/duplicates/", http.StripPrefix("/duplicates", duplicates.Handler(dups)))
	mux.Handle("/autotag/", http.StripPrefix("/autotag", autotag.Handler(notebook)))

	// the first scans run behind the server, one after the other, so the
	// service is up at once and a page shows its scan as it goes; what they
	// need of a model loads it. They end before anything they use is closed.
	var scans sync.WaitGroup
	scans.Go(func() {
		// the cheapest first: hashing takes milliseconds a picture
		if _, err := dups.Scan(ctx); err != nil && ctx.Err() == nil {
			logger.Error("duplicates scan", "err", err)
		}
		logger.Info("scanning", "dir", o.docsDir)
		if _, err := docs.Scan(ctx); err != nil && ctx.Err() == nil {
			logger.Error("documents scan", "err", err)
		}
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
func openMedia(ctx context.Context, o options, m *mediaModel, sampler *ffmpeg.Sampler, st *stores, logger *slog.Logger) (*mediasearch.Library, indexed, error) {
	store, where, base, err := st.vectors(ctx, m.name, m.embedder.Dimensions())
	if err != nil {
		return nil, indexed{}, err
	}
	coord := vector.Coord{Index: indexName, Partition: indexPartition, Collection: "media"}
	opts := append([]search.MediaOption{
		search.WithSampler(sampler), search.WithMediaMinScore(o.minScore), search.WithMaxImageBytes(o.maxImageBytes),
	}, m.opts...)
	ix, err := search.NewMediaIndex(m.embedder, store, coord.Index, coord.Partition, coord.Collection, opts...)
	if err != nil {
		return nil, indexed{}, err
	}
	lib, err := mediasearch.New(mediasearch.Config{
		Dir:         o.mediaDir,
		Index:       ix,
		Sampler:     sampler,
		CatalogFile: base + ".catalog.json",
		CacheDir:    filepath.Join(o.dataDir, "thumbs"),
		Limit:       o.limit,
		Info:        m.info,
		ModelState:  m.state,
		Stored:      func(ctx context.Context) (string, error) { return store.Model(ctx, coord) },
		Logger:      logger,
	})
	return lib, indexed{store: store, coord: coord, where: where}, err
}

// openText builds the text search page: a search.Index over the shared text
// embedder in that model's own vector store. The chunker is the index's
// default, handed to both, so the page shows the chunks the index embeds.
func openText(ctx context.Context, o options, text *textModel, st *stores, logger *slog.Logger) (*textsearch.Library, indexed, error) {
	store, where, _, err := st.vectors(ctx, text.name, text.Dimensions())
	if err != nil {
		return nil, indexed{}, err
	}
	coord := vector.Coord{Index: indexName, Partition: indexPartition, Collection: "docs"}
	chunker := &search.MarkdownChunker{}
	ix, err := search.New(text, store, coord.Index, coord.Partition, coord.Collection,
		search.WithChunker(chunker), search.WithMinScore(o.textMinScore))
	if err != nil {
		return nil, indexed{}, err
	}
	lib, err := textsearch.New(textsearch.Config{
		Dir:        o.docsDir,
		Index:      ix,
		Chunker:    chunker,
		Limit:      o.limit,
		Info:       textsearch.Info{Model: text.Model(), Dimensions: text.Dimensions(), MinScore: o.textMinScore, Store: where},
		ModelState: func() string { return state(text) },
		Stored:     func(ctx context.Context) (string, error) { return store.Model(ctx, coord) },
		Examples:   textsearch.SampleQueries(o.samples, 4),
		Logger:     logger,
	})
	return lib, indexed{store: store, coord: coord, where: where}, err
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

func homeDir() string {
	h, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return h
}

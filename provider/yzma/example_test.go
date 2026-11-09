package yzma_test

import (
	"context"
	"log"
	"path/filepath"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/media/ffmpeg"
	"github.com/go-bumbu/langai/provider/yzma"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/vector"
)

// The examples mirror the README's. They have no output, so they are
// compiled and not run: each would download and load a real model.

// config is the README's: a managed library directory named after the
// pinned build, and the models next to it.
func config() yzma.Config {
	return yzma.Config{
		ModelsDir: "data/models",
		Runtime: yzma.Runtime{
			LibDir:        filepath.Join("data", "llama-cpp", yzma.LibTag), // named after the pinned build
			ManagedLibDir: true,                                            // download the pinned build into it
		},
	}
}

// This example runs the default embedding model behind a search index.
func ExampleNew() {
	ctx := context.Background()
	emb, err := yzma.New(ctx, config()) // Model "" is yzma.DefaultModel; yzma.Models() lists the others
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = emb.Close() }()
	ix, err := search.New(emb, vector.NewMemStore(), "tenant-1", "space-7", "notes", search.WithMinScore(0.3))
	if err != nil {
		log.Fatal(err)
	}
	if err := ix.Index(ctx, "note-42", "Boil the spaghetti for nine minutes."); err != nil {
		log.Fatal(err)
	}
	hits, err := ix.Search(ctx, "how long to boil spaghetti", 10)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(hits)
}

// This example defers the downloads and the load to the first use. The
// LazyEmbedder is configured from the spec New would load: its identity, as
// the loaded Embedder's Model will report it, and its dimensions.
func ExampleLookupSpec() {
	ctx := context.Background()
	cfg := config()
	spec, err := yzma.LookupSpec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		log.Fatal(err)
	}
	emb := langai.NewLazyEmbedder(spec.ModelID(), spec.Dims, func(ctx context.Context) (langai.Embedder, error) {
		return yzma.New(ctx, cfg) // ctx is the LazyEmbedder's own: only Close cancels it
	})
	defer func() { _ = emb.Close() }()
	if err := emb.Load(ctx); err != nil { // optional: load now
		log.Fatal(err)
	}
	state, err := emb.State()
	log.Println(state, err)
}

// This example puts texts, images and videos in one space with the vision
// model, the ffmpeg Sampler cutting the videos into segments.
func ExampleNewVision() {
	ctx := context.Background()
	frames, err := ffmpeg.New(ffmpeg.Config{})
	if err != nil {
		log.Fatal(err)
	}
	v, err := yzma.NewVision(ctx, config(), yzma.VisionOptions{Cutter: frames})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = v.Close() }()
	ix, err := search.NewMediaIndex(v, vector.NewMemStore(), "tenant-1", "space-7", "media",
		search.WithVideoEmbedder(v), search.WithTextEmbedder(v))
	if err != nil {
		log.Fatal(err)
	}
	if err := ix.IndexVideo(ctx, "clip-7", "/media/clip-7.mp4"); err != nil {
		log.Fatal(err)
	}
	hits, err := ix.SearchText(ctx, "a walrus on a beach", 10)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(hits)
}

// This example has the chat model write tags and the embedding model score
// them.
func ExampleNewCompleter() {
	ctx := context.Background()
	emb, err := yzma.New(ctx, config())
	if err != nil {
		log.Fatal(err)
	}
	llm, err := yzma.NewCompleter(ctx, config())
	if err != nil {
		log.Fatal(err)
	}
	gen, err := tagging.NewGenerator(llm, emb, tagging.GeneratorConfig{}, tagging.WithLanguageTags("en"))
	if err != nil {
		log.Fatal(err)
	}
	tags, err := gen.Generate(ctx, "Baking sourdough bread at home", tagging.DefaultGeneratorMinScore)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(tags)
}

package onnx_test

import (
	"context"
	"log"
	"os"
	"path/filepath"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/media/ffmpeg"
	"github.com/go-bumbu/langai/provider/onnx"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
)

// The examples mirror the README's. They have no output, so they are
// compiled and not run: each would download and load a real model.

// cpuRuntime is the README's: a managed library directory named after the
// pinned build, on the CPU.
func cpuRuntime() onnx.Runtime {
	return onnx.Runtime{
		LibDir:        filepath.Join("data", "onnxruntime", onnx.LibVersion), // named after the pinned build
		ManagedLibDir: true,                                                  // download the pinned build into it
		Provider:      onnx.ProviderCPU,                                      // or ProviderCUDA, ProviderCoreML
	}
}

// This example finds the images and videos that look like a photo, a video
// standing for its sampled frames.
func ExampleNew() {
	ctx := context.Background()
	emb, err := onnx.New(ctx, onnx.Config{ModelsDir: "data/models", Runtime: cpuRuntime()}) // Model "" is onnx.DefaultModel
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = emb.Close() }()
	frames, err := ffmpeg.New(ffmpeg.Config{}) // scene cuts, one frame every 5 s, at most 64, 512 px
	if err != nil {
		log.Fatal(err)
	}
	ix, err := search.NewMediaIndex(emb, vector.NewMemStore(), "tenant-1", "space-7", "media",
		search.WithSampler(frames), search.WithMediaMinScore(0.6))
	if err != nil {
		log.Fatal(err)
	}
	if err := ix.IndexVideo(ctx, "clip-7", "/media/clip-7.mp4"); err != nil {
		log.Fatal(err)
	}
	photo, err := os.Open("/media/photo-42.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = photo.Close() }()
	hits, err := ix.SearchImage(ctx, photo, 10)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(hits)
}

// This example defers the download and the load to the first use, with the
// identity and dimensions the spec gives before anything loads.
func ExampleLookupSpec() {
	spec, err := onnx.LookupSpec(onnx.DINOv2Model, "data/models")
	if err != nil {
		log.Fatal(err)
	}
	emb := langai.NewLazyImageEmbedder(spec.ModelID(), spec.Dims, func(ctx context.Context) (langai.ImageEmbedder, error) {
		return onnx.Open(ctx, spec, cpuRuntime())
	})
	defer func() { _ = emb.Close() }()
	log.Println(emb.Model(), emb.Dimensions())
}

package main

import (
	"context"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/media/ffmpeg"
	"github.com/go-bumbu/sense/provider/onnx"
	"github.com/go-bumbu/sense/provider/yzma"
	"github.com/go-bumbu/sense/search"
	"github.com/go-bumbu/sense/zarf/examples/mediasearch"
)

// qwenAliases are the short names -model accepts for the vision model.
var qwenAliases = []string{"qwen3vl", "qwen3-vl-embedding-2b", strings.ToLower(yzma.DefaultVisionModel)}

// isQwen reports whether name selects the vision model.
func isQwen(name string) bool {
	name = strings.ToLower(name)
	for _, a := range qwenAliases {
		if name == a {
			return true
		}
	}
	return false
}

// runsOn remembers what a model runs on once it is loaded: a lazy model
// only knows after its load. Before that it answers with what was asked for.
type runsOn struct {
	asked  string
	loaded atomic.Pointer[string]
}

func (r *runsOn) set(s string) { r.loaded.Store(&s) }

// String says what the model runs on, or what it will be asked to run on.
func (r *runsOn) String() string {
	if s := r.loaded.Load(); s != nil {
		return *s
	}
	if r.asked == "" {
		return "auto: a GPU when the machine has one (known once loaded)"
	}
	return r.asked + " (asked for; known once loaded)"
}

// textModel is the text embedding model (EmbeddingGemma through llama.cpp)
// behind the library's LazyEmbedder: one instance, shared by the text search
// page, the Tagger and the Generator's scoring. Its identity and dimensions
// come from the pinned spec, so the stores and the Tagger's cache are set up
// under the right model before anything loads; the model itself loads on
// the first text that needs embedding. A restart with nothing new to embed
// never loads it.
type textModel struct {
	*sense.LazyEmbedder
	name   string
	runsOn runsOn
}

func newTextModel(cfg yzma.Config, logger *slog.Logger) (*textModel, error) {
	// the spec yzma.New(ctx, cfg) loads: its ModelID is what the loaded
	// model's Model() will report, which the LazyEmbedder checks
	spec, err := yzma.LookupSpec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("text embedding model: %w", err)
	}
	m := &textModel{name: spec.Name, runsOn: runsOn{asked: string(cfg.Accel)}}
	m.LazyEmbedder = sense.NewLazyEmbedder(spec.ModelID(), spec.Dims, func(ctx context.Context) (sense.Embedder, error) {
		start := time.Now()
		logger.Info("loading the text embedding model", "model", spec.Name, "accel", cfg.Accel, "dir", filepath.Dir(cfg.ModelsDir))
		e, err := yzma.New(ctx, cfg) // ctx is the LazyEmbedder's own: only its Close cancels it
		if err != nil {
			return nil, err
		}
		m.runsOn.set(e.Accel())
		logger.Info("text embedding model loaded", "model", spec.Name, "took", time.Since(start).Round(time.Millisecond))
		return e, nil
	})
	return m, nil
}

// chatModel is the chat model (Qwen3 4B Instruct) behind the library's
// LazyCompleter: 2.5 GB that only the recommend button of the auto tagging
// page needs, so it loads on the first recommendation.
type chatModel struct {
	*sense.LazyCompleter
	name string
	// accel is what the model was asked to run on. A yzma.Completer does
	// not say what it ended up on, as the embedders do.
	accel string
}

func newChatModel(cfg yzma.Config, logger *slog.Logger) (*chatModel, error) {
	spec, err := yzma.LookupCompleterSpec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("chat model: %w", err)
	}
	m := &chatModel{name: spec.Name, accel: accelName(cfg.Accel)}
	m.LazyCompleter = sense.NewLazyCompleter(func(ctx context.Context) (sense.Completer, error) {
		start := time.Now()
		logger.Info("loading the chat model", "model", spec.Name)
		c, err := yzma.NewCompleter(ctx, cfg)
		if err != nil {
			return nil, err
		}
		logger.Info("chat model loaded", "model", spec.Name, "took", time.Since(start).Round(time.Millisecond))
		return c, nil
	})
	return m, nil
}

func accelName(a yzma.Accel) string {
	if a == "" {
		return "auto"
	}
	return string(a)
}

// state is a lazy loader's state in words, with the error it keeps when it
// failed.
func state(l interface {
	State() (sense.LazyState, error)
}) string {
	s, err := l.State()
	if s == sense.LazyFailed && err != nil {
		return "load failed: " + err.Error()
	}
	return s.String()
}

// mediaModel is the model of the media search page: an ONNX image model
// behind the library's LazyImageEmbedder, or the vision model, loaded now.
type mediaModel struct {
	// embedder is what the MediaIndex embeds with.
	embedder sense.ImageEmbedder
	// lazy is the ONNX model's loader; nil for the vision model.
	lazy *sense.LazyImageEmbedder
	// vision is the loaded vision model; nil for an ONNX model.
	vision *yzma.Vision
	// name is the model's name and id the identity of its vectors.
	name string
	id   string
	// runtime names what runs it, for the status page.
	runtime string
	runsOn  runsOn
	// opts are the MediaIndex options the model brings: the vision model
	// is also the index's video and text embedder.
	opts []search.MediaOption
	info mediasearch.Info
}

// newMediaModel builds the media page's model.
//
// An ONNX model (SigLIP 2, DINOv2) goes behind a LazyImageEmbedder. The
// page scans its directory at startup, but a file whose bytes are unchanged
// is a no-op that never reaches the model, and "looks like" searches with
// the stored vectors: so a restart over an unchanged library serves the
// whole page without loading the 370 MB model. The first new file, upload
// or "re-embed" loads it.
//
// The vision model (Qwen3-VL) is loaded here, before the service starts. It
// is the index's image, video and text embedder in one object; the library
// has a lazy loader for the first and the last and none for a
// VideoEmbedder, and the fingerprint the index hashes with every video
// (Vision.Fingerprint) exists on the loaded model only.
func newMediaModel(ctx context.Context, o options, llama yzma.Config, sampler *ffmpeg.Sampler, logger *slog.Logger) (*mediaModel, error) {
	if isQwen(o.model) {
		return newVisionModel(ctx, o, llama, sampler, logger)
	}
	cfg := onnx.Config{
		ModelsDir: filepath.Join(o.modelsDir, "models"),
		Model:     o.model,
		Runtime: onnx.Runtime{
			LibDir:        filepath.Join(o.modelsDir, "onnxruntime", onnx.LibVersion),
			ManagedLibDir: true,
			Provider:      onnx.Provider(o.provider),
			Logger:        logger,
		},
	}
	spec, err := onnx.LookupSpec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("image model: %w", err)
	}
	m := &mediaModel{
		name: spec.Name, id: spec.ModelID(), runtime: "ONNX Runtime " + onnx.LibVersion + " (provider/onnx)",
		runsOn: runsOn{asked: o.provider},
		info:   mediasearch.Info{Model: spec.Name, ID: spec.ModelID(), Dimensions: spec.Dims, MinScore: o.minScore, Mode: "frames", Provider: o.provider},
	}
	m.lazy = sense.NewLazyImageEmbedder(spec.ModelID(), spec.Dims, func(ctx context.Context) (sense.ImageEmbedder, error) {
		start := time.Now()
		logger.Info("loading the image model", "model", spec.Name, "provider", o.provider, "dir", o.modelsDir)
		e, err := onnx.New(ctx, cfg)
		if err != nil {
			return nil, err
		}
		m.runsOn.set(o.provider)
		logger.Info("image model loaded", "model", spec.Name, "took", time.Since(start).Round(time.Millisecond))
		return e, nil
	})
	m.embedder = m.lazy
	return m, nil
}

func newVisionModel(ctx context.Context, o options, llama yzma.Config, sampler *ffmpeg.Sampler, logger *slog.Logger) (*mediaModel, error) {
	logger.Info("loading the vision model", "model", yzma.DefaultVisionModel, "accel", o.accel, "dir", o.llamaDir)
	v, err := yzma.NewVision(ctx, llama, yzma.VisionOptions{SegmentLength: o.segment, Cutter: sampler})
	if err != nil {
		return nil, fmt.Errorf("vision model: %w", err)
	}
	m := &mediaModel{
		embedder: v, vision: v, name: yzma.DefaultVisionModel, id: v.Model(),
		runtime: "llama.cpp " + yzma.LibTag + " with libmtmd (provider/yzma)",
		opts:    []search.MediaOption{search.WithVideoEmbedder(v), search.WithTextEmbedder(v)},
		info: mediasearch.Info{
			Model: yzma.DefaultVisionModel, ID: v.Model(), Dimensions: v.Dimensions(), MinScore: o.minScore, Provider: v.Accel(),
			Mode: "segments", SegmentMs: v.SegmentLength().Milliseconds(), Text: true,
		},
	}
	m.runsOn.set(v.Accel())
	return m, nil
}

// state says where the model stands: its loader's state, or "loaded at
// startup" for the vision model.
func (m *mediaModel) state() string {
	if m.lazy == nil {
		return "loaded at startup"
	}
	return state(m.lazy)
}

// Close releases the model.
func (m *mediaModel) Close() error {
	if m.lazy != nil {
		return m.lazy.Close()
	}
	return m.vision.Close()
}

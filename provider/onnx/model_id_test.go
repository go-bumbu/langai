package onnx

import (
	"context"
	"image"
	"strings"
	"testing"

	"github.com/go-bumbu/langai"
)

// the Embedder offers the optional port a store asks for the identity of
// the vectors it is handed (langai.Identified)
var _ interface{ Model() string } = (*Embedder)(nil)

// TestModelSpec_ModelIDIsTheLoadedModel: the identity a spec gives before
// anything loads, defaults unfilled, is exactly what the Embedder opened
// with it reports.
func TestModelSpec_ModelIDIsTheLoadedModel(t *testing.T) {
	libDir, modelFile := fakeInstall(t)
	for name, spec := range map[string]ModelSpec{
		"SigLIP 2": pinned[DefaultModel],
		"DINOv2":   pinned[DINOv2Model],
		"defaults": {Name: "org/m", Side: 8, Dims: 4},
	} {
		t.Run(name, func(t *testing.T) {
			f := &fakeAPI{dims: spec.Dims}
			if spec.Pooling == PoolingCLS || spec.Pooling == PoolingMean {
				f.tokens = 3
			}
			useFake(t, f)
			spec.File, spec.URL, spec.SHA256, spec.Size = modelFile, "", "", 0
			before := spec.ModelID()
			emb, err := Open(context.Background(), spec, Runtime{LibDir: libDir})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = emb.Close() }()
			if emb.Model() != before {
				t.Errorf("ModelID() = %q before the load, Model() = %q after it", before, emb.Model())
			}
		})
	}
}

// TestLazyImageEmbedder_AroundOnnx wires the seam the README shows: a
// LazyImageEmbedder configured from the spec loads the onnx Embedder opened
// with it, and one configured with the bare name refuses the load.
func TestLazyImageEmbedder_AroundOnnx(t *testing.T) {
	libDir, modelFile := fakeInstall(t)
	useFake(t, &fakeAPI{dims: 4})
	spec := ModelSpec{Name: "org/m", File: modelFile, Side: 8, Dims: 4}
	load := func(ctx context.Context) (langai.ImageEmbedder, error) {
		return Open(ctx, spec, Runtime{LibDir: libDir})
	}
	emb := langai.NewLazyImageEmbedder(spec.ModelID(), spec.Dims, load)
	defer func() { _ = emb.Close() }()
	vecs, err := emb.EmbedImages(context.Background(), []image.Image{image.NewRGBA(image.Rect(0, 0, 8, 8))})
	if err != nil || len(vecs) != 1 {
		t.Fatalf("EmbedImages through the LazyImageEmbedder: %d vectors, %v", len(vecs), err)
	}
	byName := langai.NewLazyImageEmbedder(spec.Name, spec.Dims, load)
	defer func() { _ = byName.Close() }()
	if err := byName.Load(context.Background()); err == nil || !strings.Contains(err.Error(), "must not mix") {
		t.Errorf("a LazyImageEmbedder configured with the bare name loaded: %v", err)
	}
}

// TestEmbedder_ModelNamesWhatDecidesAVector: the identity is the model's
// name plus the tensor that is read and how a picture is prepared, since
// each of them moves the vector of a picture, and nothing that does not.
func TestEmbedder_ModelNamesWhatDecidesAVector(t *testing.T) {
	libDir, modelFile := fakeInstall(t)
	id := func(spec ModelSpec, rt Runtime) string {
		t.Helper()
		f := &fakeAPI{dims: spec.Dims}
		if spec.Pooling == PoolingCLS || spec.Pooling == PoolingMean {
			f.tokens = 3
		}
		useFake(t, f)
		rt.LibDir = libDir
		// the fake install's file stands in for the model, whatever its pin says
		spec.File, spec.URL, spec.SHA256, spec.Size = modelFile, "", "", 0
		emb, err := Open(context.Background(), spec, rt)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = emb.Close() }()
		return emb.Model()
	}
	def := id(pinned[DefaultModel], Runtime{})
	want := "google/siglip2-base-patch16-224;output=pooler_output;pooling=none;side=224;resize=squash;edge=224;filter=bilinear;mean=0.5,0.5,0.5;std=0.5,0.5,0.5;prep=2"
	if def != want {
		t.Errorf("default identity = %q\nwant               %q", def, want)
	}
	// the same on every open, whatever runs the model
	if again := id(pinned[DefaultModel], Runtime{Threads: 2, BatchSize: 32, Provider: ProviderCPU}); again != def {
		t.Errorf("threads or batch size changed the identity: %q", again)
	}
	dino := id(pinned[DINOv2Model], Runtime{})
	if want := "facebook/dinov2-base;output=last_hidden_state;pooling=cls;side=224;resize=crop;edge=256;filter=bicubic;mean=0.485,0.456,0.406;std=0.229,0.224,0.225;prep=2"; dino != want {
		t.Errorf("DINOv2 identity = %q\nwant              %q", dino, want)
	}
	base := ModelSpec{Name: "org/m", Side: 8, Dims: 4}
	seen := map[string]string{id(base, Runtime{}): "base"}
	for name, edit := range map[string]func(*ModelSpec){
		"another name":   func(s *ModelSpec) { s.Name = "org/other" },
		"another side":   func(s *ModelSpec) { s.Side = 16 },
		"cropped":        func(s *ModelSpec) { s.Resize = ResizeCrop },
		"a wider edge":   func(s *ModelSpec) { s.Resize, s.ShortestEdge = ResizeCrop, 10 },
		"bicubic":        func(s *ModelSpec) { s.Interpolation = InterpolationBicubic },
		"another mean":   func(s *ModelSpec) { s.Mean = [3]float32{0.5, 0.5, 0.5} },
		"another std":    func(s *ModelSpec) { s.Std = [3]float32{0.5, 0.5, 0.5} },
		"cls pooling":    func(s *ModelSpec) { s.Pooling, s.Output = PoolingCLS, "last_hidden_state" },
		"mean pooling":   func(s *ModelSpec) { s.Pooling, s.Output = PoolingMean, "last_hidden_state" },
		"another tensor": func(s *ModelSpec) { s.Output = "image_embeds" },
	} {
		spec := base
		edit(&spec)
		got := id(spec, Runtime{})
		if other, dup := seen[got]; dup {
			t.Errorf("%s and %s share the identity %q", name, other, got)
		}
		seen[got] = name
	}
}

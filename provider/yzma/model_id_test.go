package yzma

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
)

// TestModelSpec_ModelIDIsTheLoadedModel: the identity a spec gives before
// anything loads is exactly what the Embedder opened with it reports, so a
// LazyEmbedder or a store set up from the spec checks the loaded model
// against the right string.
func TestModelSpec_ModelIDIsTheLoadedModel(t *testing.T) {
	tmp := t.TempDir()
	rt := fakeRuntime(t, newFakeAPI(), tmp)
	file := filepath.Join(tmp, "m.gguf")
	if err := os.WriteFile(file, []byte("GGUF fake model"), 0o600); err != nil {
		t.Fatal(err)
	}
	pinned := DefaultSpec(tmp)
	pinned.File, pinned.URL, pinned.SHA256, pinned.Size, pinned.Dims = file, "", "", 0, 0
	for name, spec := range map[string]ModelSpec{
		"the default model's settings": pinned,
		"mean pooling by default":      {Name: "org/m", File: file},
		"cls and prefixes":             {Name: "org/m", File: file, Pooling: PoolingCLS, QueryPrefix: "q: ", DocumentPrefix: "d: "},
		"a query prefix only":          {Name: "org/m", File: file, Pooling: PoolingLast, QueryPrefix: "q: "},
	} {
		t.Run(name, func(t *testing.T) {
			before := spec.ModelID()
			e, err := Open(context.Background(), spec, rt)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = e.Close() }()
			if e.Model() != before {
				t.Errorf("ModelID() = %q before the load, Model() = %q after it", before, e.Model())
			}
		})
	}
	if got := DefaultSpec("").ModelID(); !regexp.MustCompile(`^google/embeddinggemma-300m;pooling=mean;prefixes=[0-9a-f]{8}$`).MatchString(got) {
		t.Errorf("the default spec's identity is %q", got)
	}
}

// TestVisionSpec_ModelIDIsTheLoadedModel: as for the text embedder, the
// identity a vision spec and its options give before the load is the
// loaded Vision's, defaults filled in or spelled out; with no Dims it is
// unknown.
func TestVisionSpec_ModelIDIsTheLoadedModel(t *testing.T) {
	for name, opts := range map[string]VisionOptions{
		"defaults":            {},
		"defaults spelled":    {Instruction: DefaultInstruction, FPS: DefaultFPS, SegmentLength: DefaultSegmentLength},
		"another instruction": {Instruction: "Find duplicates."},
		"image token bounds":  {ImageMinTokens: 64, ImageMaxTokens: 512},
		"video sampling":      {FPS: 0.5, SegmentLength: 8 * time.Second, MaxSegments: 3, ContextTokens: 8192},
	} {
		t.Run(name, func(t *testing.T) {
			before := testVisionSpec().ModelID(opts)
			if got := openVisionFake(t, newFakeVision(), opts).Model(); got != before {
				t.Errorf("ModelID() = %q before the load, Model() = %q after it", before, got)
			}
		})
	}
	unsized := testVisionSpec()
	unsized.Dims = 0
	if got := unsized.ModelID(VisionOptions{}); got != "" {
		t.Errorf("a spec without Dims gives the identity %q, want none", got)
	}
}

// TestLazyEmbedder_AroundYzma wires the seam the README shows: a
// LazyEmbedder configured from the spec, loading the yzma Embedder opened
// with it. The identity from the spec passes the lazy check; the bare name,
// which the loaded model's identity only starts with, does not.
func TestLazyEmbedder_AroundYzma(t *testing.T) {
	tmp := t.TempDir()
	rt := fakeRuntime(t, newFakeAPI(), tmp)
	file := filepath.Join(tmp, "m.gguf")
	if err := os.WriteFile(file, []byte("GGUF fake model"), 0o600); err != nil {
		t.Fatal(err)
	}
	spec := ModelSpec{Name: "org/m", File: file, QueryPrefix: "q: ", DocumentPrefix: "d: ", Dims: 4}
	load := func(ctx context.Context) (langai.Embedder, error) { return Open(ctx, spec, rt) }

	emb := langai.NewLazyEmbedder(spec.ModelID(), spec.Dims, load)
	defer func() { _ = emb.Close() }()
	vecs, err := emb.Embed(context.Background(), langai.RoleQuery, []string{"hello"})
	if err != nil || len(vecs) != 1 || len(vecs[0]) != spec.Dims {
		t.Fatalf("Embed through the LazyEmbedder: %d vectors, %v", len(vecs), err)
	}

	byName := langai.NewLazyEmbedder(spec.Name, spec.Dims, load)
	defer func() { _ = byName.Close() }()
	if err := byName.Load(context.Background()); err == nil || !strings.Contains(err.Error(), "must not mix") {
		t.Errorf("a LazyEmbedder configured with the bare name loaded: %v", err)
	}
}

// identified is the optional port a store asks for the identity of the
// vectors it is handed (langai.Identified); both embedders offer it.
type identified interface{ Model() string }

var (
	_ identified = (*Embedder)(nil)
	_ identified = (*Vision)(nil)
)

// TestEmbedderModel_NamesWhatDecidesAVector: the identity is the model's
// name plus the pooling and the prefixes, since each of them moves the
// vector of a text, and nothing that does not.
func TestEmbedderModel_NamesWhatDecidesAVector(t *testing.T) {
	spec := DefaultSpec("")
	def := embedderModel(spec.Name, spec.Pooling, spec.QueryPrefix, spec.DocumentPrefix)
	if !regexp.MustCompile(`^google/embeddinggemma-300m;pooling=mean;prefixes=[0-9a-f]{8}$`).MatchString(def) {
		t.Errorf("the default model's identity is %q", def)
	}
	if got := embedderModel("org/m", PoolingCLS, "", ""); got != "org/m;pooling=cls;prefixes=none" {
		t.Errorf("a model without prefixes: %q", got)
	}
	seen := map[string]string{}
	for name, id := range map[string]string{
		"default":               def,
		"another name":          embedderModel("org/other", spec.Pooling, spec.QueryPrefix, spec.DocumentPrefix),
		"another pooling":       embedderModel(spec.Name, PoolingLast, spec.QueryPrefix, spec.DocumentPrefix),
		"another query prefix":  embedderModel(spec.Name, spec.Pooling, "query: ", spec.DocumentPrefix),
		"another doc prefix":    embedderModel(spec.Name, spec.Pooling, spec.QueryPrefix, "passage: "),
		"no prefixes":           embedderModel(spec.Name, spec.Pooling, "", ""),
		"the prefixes swapped":  embedderModel(spec.Name, spec.Pooling, spec.DocumentPrefix, spec.QueryPrefix),
		"one prefix, as query":  embedderModel(spec.Name, spec.Pooling, "ab", ""),
		"one prefix, as doc":    embedderModel(spec.Name, spec.Pooling, "", "ab"),
		"split between the two": embedderModel(spec.Name, spec.Pooling, "a", "b"),
	} {
		if other, dup := seen[id]; dup {
			t.Errorf("%s and %s share the identity %q", name, other, id)
		}
		seen[id] = name
	}
}

// TestOpen_SetsTheModelIdentity: an opened Embedder reports the identity of
// its spec, the same every time it is opened.
func TestOpen_SetsTheModelIdentity(t *testing.T) {
	tmp := t.TempDir()
	rt := fakeRuntime(t, newFakeAPI(), tmp)
	modelPath := filepath.Join(tmp, "custom.gguf")
	if err := os.WriteFile(modelPath, []byte("GGUF fake model"), 0o600); err != nil {
		t.Fatal(err)
	}
	spec := ModelSpec{Name: "org/custom", File: modelPath, Pooling: PoolingCLS, QueryPrefix: "q: ", DocumentPrefix: "d: "}
	var ids []string
	for range 2 {
		e, err := Open(context.Background(), spec, rt)
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, e.Model())
		_ = e.Close()
	}
	if want := embedderModel("org/custom", PoolingCLS, "q: ", "d: "); ids[0] != want || ids[1] != want {
		t.Errorf("Model() = %q, want %q both times", ids, want)
	}
	// "" is mean pooling, and says so
	spec.Pooling = ""
	e, err := Open(context.Background(), spec, rt)
	if err != nil {
		t.Fatal(err)
	}
	if want := embedderModel("org/custom", PoolingMean, "q: ", "d: "); e.Model() != want {
		t.Errorf("Model() = %q, want %q", e.Model(), want)
	}
}

// TestVisionModel_NamesWhatDecidesAVector: the identity of a Vision's
// vectors is its model plus the instruction, the vector length, the image
// token bounds and how videos are sampled and cut.
func TestVisionModel_NamesWhatDecidesAVector(t *testing.T) {
	id := func(opts VisionOptions) string {
		t.Helper()
		return openVisionFake(t, newFakeVision(), opts).Model()
	}
	def := id(VisionOptions{})
	if want := "org/vision;instruction=" + shortDigest(DefaultInstruction) + ";dims=4;image_tokens=0-0;fps=1;segment=20s"; def != want {
		t.Errorf("default identity = %q\nwant               %q", def, want)
	}
	if again := id(VisionOptions{}); again != def {
		t.Errorf("the identity changed between two opens: %q, %q", def, again)
	}
	// the zero values and the defaults they mean are one identity
	if same := id(VisionOptions{Instruction: DefaultInstruction, FPS: DefaultFPS, SegmentLength: DefaultSegmentLength}); same != def {
		t.Errorf("spelling the defaults out changed the identity: %q", same)
	}
	// what moves no vector is not part of it
	if same := id(VisionOptions{ContextTokens: 8192, MaxSegments: 3, Cutter: &fakeCutter{}}); same != def {
		t.Errorf("the context size or the segment cap changed the identity: %q", same)
	}
	seen := map[string]string{def: "default"}
	for name, cfg := range map[string]VisionOptions{
		"another instruction": {Instruction: "Find duplicates."},
		"image token cap":     {ImageMaxTokens: 512},
		"image token floor":   {ImageMinTokens: 512},
		"another frame rate":  {FPS: 0.5},
		"another segment":     {SegmentLength: 8 * time.Second},
	} {
		got := id(cfg)
		if other, dup := seen[got]; dup {
			t.Errorf("%s and %s share the identity %q", name, other, got)
		}
		seen[got] = name
	}
}

package yzma

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"
)

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

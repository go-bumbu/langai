package yzma

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
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
		"a token cap of its own":       {Name: "org/m", File: file, MaxTokens: 128},
		"a cap beyond the window":      {Name: "org/m", File: file, MaxTokens: 1 << 20},
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
	if got := DefaultSpec("").ModelID(); !regexp.MustCompile(`^google/embeddinggemma-300m;pooling=mean;prefixes=[0-9a-f]{8};max_tokens=512$`).MatchString(got) {
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
		"a small context":     {ContextTokens: 64},
		// the identity names the context asked for, also where the model's
		// window (32768 in the fake) grants less
		"a context beyond the window": {ContextTokens: 1 << 20},
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
// name plus the pooling, the prefixes and the token cap, since each of them
// moves the vector of some text (the cap that of a text longer than it,
// which is cut), and nothing that moves none.
func TestEmbedderModel_NamesWhatDecidesAVector(t *testing.T) {
	spec := DefaultSpec("")
	def := spec.ModelID()
	if !regexp.MustCompile(`^google/embeddinggemma-300m;pooling=mean;prefixes=[0-9a-f]{8};max_tokens=512$`).MatchString(def) {
		t.Errorf("the default model's identity is %q", def)
	}
	if got := (ModelSpec{Name: "org/m", Pooling: PoolingCLS}).ModelID(); got != "org/m;pooling=cls;prefixes=none;max_tokens=512" {
		t.Errorf("a model without prefixes: %q", got)
	}
	// the zero cap and the default it means are one identity; the file, the
	// download and the vector length the spec states are not part of it
	same := spec
	same.MaxTokens, same.File, same.URL, same.SHA256, same.Size, same.Dims = 0, "/elsewhere/m.gguf", "", "", 0, 0
	if got := same.ModelID(); got != def {
		t.Errorf("the default cap spelled as 0, in another file: %q, want %q", got, def)
	}
	with := func(change func(*ModelSpec)) string {
		s := spec
		change(&s)
		return s.ModelID()
	}
	seen := map[string]string{}
	for name, id := range map[string]string{
		"default":               def,
		"another name":          with(func(s *ModelSpec) { s.Name = "org/other" }),
		"another pooling":       with(func(s *ModelSpec) { s.Pooling = PoolingLast }),
		"another query prefix":  with(func(s *ModelSpec) { s.QueryPrefix = "query: " }),
		"another doc prefix":    with(func(s *ModelSpec) { s.DocumentPrefix = "passage: " }),
		"no prefixes":           with(func(s *ModelSpec) { s.QueryPrefix, s.DocumentPrefix = "", "" }),
		"the prefixes swapped":  with(func(s *ModelSpec) { s.QueryPrefix, s.DocumentPrefix = s.DocumentPrefix, s.QueryPrefix }),
		"one prefix, as query":  with(func(s *ModelSpec) { s.QueryPrefix, s.DocumentPrefix = "ab", "" }),
		"one prefix, as doc":    with(func(s *ModelSpec) { s.QueryPrefix, s.DocumentPrefix = "", "ab" }),
		"split between the two": with(func(s *ModelSpec) { s.QueryPrefix, s.DocumentPrefix = "a", "b" }),
		"a smaller token cap":   with(func(s *ModelSpec) { s.MaxTokens = 256 }),
		"a larger token cap":    with(func(s *ModelSpec) { s.MaxTokens = 2048 }),
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
	if want := "org/custom;pooling=cls;prefixes=" + shortDigest("q: ", "d: ") + ";max_tokens=512"; ids[0] != want || ids[1] != want {
		t.Errorf("Model() = %q, want %q both times", ids, want)
	}
	// "" is mean pooling, and says so; a token cap of its own is named
	spec.Pooling, spec.MaxTokens = "", 128
	e, err := Open(context.Background(), spec, rt)
	if err != nil {
		t.Fatal(err)
	}
	if want := "org/custom;pooling=mean;prefixes=" + shortDigest("q: ", "d: ") + ";max_tokens=128"; e.Model() != want {
		t.Errorf("Model() = %q, want %q", e.Model(), want)
	}
}

// TestVisionModel_NamesWhatDecidesAVector: the identity of a Vision's
// vectors is its model plus the instruction, the vector length, the image
// token bounds, how videos are sampled and cut, and the context size, which
// decides where a long text is cut.
func TestVisionModel_NamesWhatDecidesAVector(t *testing.T) {
	id := func(opts VisionOptions) string {
		t.Helper()
		return openVisionFake(t, newFakeVision(), opts).Model()
	}
	def := id(VisionOptions{})
	if want := "org/vision;instruction=" + shortDigest(DefaultInstruction) + ";dims=4;image_tokens=0-0;fps=1;segment=20s;ctx=16384"; def != want {
		t.Errorf("default identity = %q\nwant               %q", def, want)
	}
	if again := id(VisionOptions{}); again != def {
		t.Errorf("the identity changed between two opens: %q, %q", def, again)
	}
	// the zero values and the defaults they mean are one identity
	if same := id(VisionOptions{Instruction: DefaultInstruction, FPS: DefaultFPS, SegmentLength: DefaultSegmentLength, ContextTokens: DefaultVisionContextTokens, MaxSegments: DefaultMaxSegments}); same != def {
		t.Errorf("spelling the defaults out changed the identity: %q", same)
	}
	// what moves no vector is not part of it: the segment cap decides which
	// segments there are (see Fingerprint), not what one holds
	if same := id(VisionOptions{MaxSegments: 3, Cutter: &fakeCutter{}, TempDir: t.TempDir()}); same != def {
		t.Errorf("the segment cap, the cutter or the temp directory changed the identity: %q", same)
	}
	seen := map[string]string{def: "default"}
	for name, cfg := range map[string]VisionOptions{
		"another instruction": {Instruction: "Find duplicates."},
		"image token cap":     {ImageMaxTokens: 512},
		"image token floor":   {ImageMinTokens: 512},
		"another frame rate":  {FPS: 0.5},
		"another segment":     {SegmentLength: 8 * time.Second},
		"a smaller context":   {ContextTokens: 64},
		"another context":     {ContextTokens: 4096},
	} {
		got := id(cfg)
		if other, dup := seen[got]; dup {
			t.Errorf("%s and %s share the identity %q", name, other, got)
		}
		seen[got] = name
	}
}

// the Vision is a search.Fingerprinter
var _ search.Fingerprinter = (*Vision)(nil)

// TestVisionFingerprint_NamesWhichSegmentsAVideoYields: the fingerprint is
// the same for two Visions that cut a video into the same segments and
// another one whenever they would not: another segment length or another
// cap. What only changes a segment's vector is the identity's business.
func TestVisionFingerprint_NamesWhichSegmentsAVideoYields(t *testing.T) {
	fp := func(opts VisionOptions) string {
		t.Helper()
		return openVisionFake(t, newFakeVision(), opts).Fingerprint()
	}
	def := fp(VisionOptions{})
	if want := fmt.Sprintf("yzma-vision/%d segment=20s max_segments=720", segmenting); def != want {
		t.Errorf("default fingerprint = %q, want %q", def, want)
	}
	for name, opts := range map[string]VisionOptions{
		"the defaults spelled out": {SegmentLength: DefaultSegmentLength, MaxSegments: DefaultMaxSegments},
		"another instruction":      {Instruction: "Find duplicates."},
		"another context":          {ContextTokens: 4096},
		"another temp directory":   {TempDir: t.TempDir()},
		"a cutter":                 {Cutter: &fakeCutter{}},
	} {
		if got := fp(opts); got != def {
			t.Errorf("%s changed the fingerprint: %q", name, got)
		}
	}
	seen := map[string]string{def: "default"}
	for name, opts := range map[string]VisionOptions{
		"a cap of two":             {MaxSegments: 2},
		"a larger cap":             {MaxSegments: 5000},
		"another segment length":   {SegmentLength: 8 * time.Second},
		"another length and a cap": {SegmentLength: 8 * time.Second, MaxSegments: 2},
	} {
		got := fp(opts)
		if other, dup := seen[got]; dup {
			t.Errorf("%s and %s share the fingerprint %q", name, other, got)
		}
		seen[got] = name
	}
}

// TestMediaIndex_ReindexesWhenTheSegmentCapChanges: a MediaIndex finds a
// video unchanged by a hash that covers the VideoEmbedder's fingerprint. A
// video indexed under a cap of two segments must be indexed again once the
// cap is lifted, or it would keep its two segments for good; under the same
// settings it is left alone.
func TestMediaIndex_ReindexesWhenTheSegmentCapChanges(t *testing.T) {
	ctx := context.Background()
	clip := filepath.Join(t.TempDir(), "clip.mp4")
	if err := os.WriteFile(clip, []byte("pretend video bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := vector.NewMemStore()
	cutter := &fakeCutter{durations: map[string]time.Duration{"clip.mp4": 70 * time.Second}}
	index := func(maxSegments int) int {
		t.Helper()
		f := newFakeVision()
		v := openVisionFake(t, f, VisionOptions{Cutter: cutter, MaxSegments: maxSegments})
		ix, err := search.NewMediaIndex(v, store, "t", "p", "c", search.WithVideoEmbedder(v))
		if err != nil {
			t.Fatal(err)
		}
		if err := ix.IndexVideo(ctx, "clip", clip); err != nil {
			t.Fatal(err)
		}
		return f.count("bitmapVideo")
	}
	for _, step := range []struct {
		name        string
		maxSegments int
		embedded    int // segments that went through the model
	}{
		{"first, capped at two", 2, 2},
		{"again, as before", 2, 0},
		{"the cap lifted", 0, 4},
		{"again, as before", 0, 0},
		{"capped at three", 3, 3},
	} {
		if got := index(step.maxSegments); got != step.embedded {
			t.Errorf("%s (MaxSegments %d): %d segments embedded, want %d", step.name, step.maxSegments, got, step.embedded)
		}
	}
}

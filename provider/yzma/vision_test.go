package yzma

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/media"
)

// Handles the vision fake hands out.
const (
	fakeProj   uintptr = 0x3000
	fakeBitmap uintptr = 0x4000
	fakeVideo  uintptr = 0x5000
	fakeChunks uintptr = 0x6000
)

// fakeVision is a visionAPI without native code. It records every call and
// turns each failure mode into a knob.
type fakeVision struct {
	callLog
	devs           []device
	loadErr        error
	loadModelErr   error
	trainCtx       int32
	newContextErr  error
	grantCtx       int
	projectorErr   error
	template       string
	templateErr    error
	bitmapErr      error
	videoErr       error
	tokenizeErr    error
	tokensPerImage int // tokens a bitmap becomes; text is len(prompt)/4
	evalErr        error
	embeddingsErr  error
	buf            []float32 // what embeddings returns
	nEmbdValue     int32     // what nEmbdOut reports; 0 = len(buf)
	lastPrompt     string
	lastBitmaps    int
}

var _ visionAPI = (*fakeVision)(nil)

func newFakeVision() *fakeVision {
	return &fakeVision{trainCtx: 32768, tokensPerImage: 300, buf: []float32{3, 4, 0, 0}}
}

func (f *fakeVision) load(dir string, _ *slog.Logger) error {
	f.record("load", dir)
	return f.loadErr
}
func (f *fakeVision) version() string { return libVersion }
func (f *fakeVision) devices() []device {
	f.record("devices")
	return f.devs
}
func (f *fakeVision) loadModel(path string, gpuLayers int32) (uintptr, error) {
	f.record("loadModel", path, gpuLayers)
	if f.loadModelErr != nil {
		return 0, f.loadModelErr
	}
	return fakeModel, nil
}
func (f *fakeVision) freeModel(model uintptr) { f.record("freeModel", model) }
func (f *fakeVision) nCtxTrain(model uintptr) int32 {
	f.record("nCtxTrain", model)
	return f.trainCtx
}
func (f *fakeVision) nEmbdOut(model uintptr) int32 {
	f.record("nEmbdOut", model)
	if f.nEmbdValue != 0 {
		return f.nEmbdValue
	}
	return int32(len(f.buf))
}
func (f *fakeVision) newVisionContext(model uintptr, nCtx, batch int32, threads int) (uintptr, int, int, uintptr, error) {
	f.record("newVisionContext", model, nCtx, batch, threads)
	if f.newContextErr != nil {
		return 0, 0, 0, 0, f.newContextErr
	}
	got := int(nCtx)
	if f.grantCtx != 0 {
		got = f.grantCtx
	}
	return fakeCtx, got, int(batch), fakeMem, nil
}
func (f *fakeVision) freeContext(ctx uintptr) { f.record("freeContext", ctx) }
func (f *fakeVision) memoryClear(mem uintptr) error {
	f.record("memoryClear", mem)
	return nil
}
func (f *fakeVision) embeddings(ctx uintptr, n int32) ([]float32, error) {
	f.record("embeddings", ctx, n)
	if f.embeddingsErr != nil {
		return nil, f.embeddingsErr
	}
	return f.buf, nil
}
func (f *fakeVision) chatTemplate(model uintptr) string {
	f.record("chatTemplate", model)
	return f.template
}
func (f *fakeVision) applyTemplate(template, system, user string) (string, error) {
	f.record("applyTemplate", template, system, user)
	if f.templateErr != nil {
		return "", f.templateErr
	}
	return "<s>" + system + "\n" + user + "\n<a>", nil
}
func (f *fakeVision) loadProjector(path string, model uintptr, threads int, useGPU bool, minTok, maxTok int) (uintptr, error) {
	f.record("loadProjector", path, model, threads, useGPU, minTok, maxTok)
	if f.projectorErr != nil {
		return 0, f.projectorErr
	}
	return fakeProj, nil
}
func (f *fakeVision) freeProjector(proj uintptr) { f.record("freeProjector", proj) }
func (f *fakeVision) marker() string             { return "<__media__>" }
func (f *fakeVision) bitmapRGB(w, h int, rgb []byte) (uintptr, error) {
	f.record("bitmapRGB", w, h, len(rgb))
	if f.bitmapErr != nil {
		return 0, f.bitmapErr
	}
	return fakeBitmap, nil
}
func (f *fakeVision) bitmapVideo(proj uintptr, path string, fps float64) (uintptr, uintptr, error) {
	f.record("bitmapVideo", proj, filepath.Base(path), fps)
	if f.videoErr != nil {
		return 0, 0, f.videoErr
	}
	return fakeBitmap, fakeVideo, nil
}
func (f *fakeVision) freeBitmap(bitmap uintptr) { f.record("freeBitmap", bitmap) }
func (f *fakeVision) freeVideo(video uintptr)   { f.record("freeVideo", video) }
func (f *fakeVision) tokenizeMedia(proj uintptr, prompt string, bitmaps []uintptr) (uintptr, int, error) {
	f.record("tokenizeMedia", proj, prompt, len(bitmaps))
	f.lastPrompt, f.lastBitmaps = prompt, len(bitmaps)
	if f.tokenizeErr != nil {
		return 0, 0, f.tokenizeErr
	}
	return fakeChunks, len(prompt)/4 + f.tokensPerImage*len(bitmaps), nil
}
func (f *fakeVision) freeChunks(chunks uintptr) { f.record("freeChunks", chunks) }
func (f *fakeVision) evalChunks(proj, ctx, chunks uintptr, batch int32) error {
	f.record("evalChunks", proj, ctx, chunks, batch)
	return f.evalErr
}

// fakeCutter answers Duration from a map and records the cuts, creating
// the output files so the next step finds them.
type fakeCutter struct {
	durations map[string]time.Duration
	cuts      []string // "start/length/base"
	err       error
}

func (c *fakeCutter) Duration(_ context.Context, path string) (time.Duration, error) {
	d, ok := c.durations[filepath.Base(path)]
	if !ok {
		return 0, errors.New("fake cutter: unknown " + path)
	}
	return d, nil
}

func (c *fakeCutter) Cut(_ context.Context, _ string, start, length time.Duration, out string) error {
	c.cuts = append(c.cuts, start.String()+"/"+length.String()+"/"+filepath.Base(out))
	if c.err != nil {
		return c.err
	}
	return os.WriteFile(out, []byte("cut"), 0o600)
}

func testVisionSpec() VisionSpec {
	return VisionSpec{Name: "org/vision", File: "/opt/models/vision.gguf", Projector: "/opt/models/mmproj.gguf", Dims: 4}
}

func openVisionFake(t *testing.T, f *fakeVision, cfg VisionConfig) *Vision {
	t.Helper()
	v, err := openVision(f, &runtimeLoader{}, testLibDir, testVisionSpec(), cfg, Runtime{Threads: 2, Accel: AccelCPU})
	if err != nil {
		t.Fatal(err)
	}
	f.calls = nil
	return v
}

func TestOpenVision_LoadsProbesAndFreesOnFailure(t *testing.T) {
	f := newFakeVision()
	f.template = "{{ jinja }}"
	v, err := openVision(f, &runtimeLoader{}, testLibDir, testVisionSpec(), VisionConfig{ImageMaxTokens: 512}, Runtime{Threads: 2, Accel: AccelCPU})
	if err != nil {
		t.Fatal(err)
	}
	if v.Dimensions() != 4 || v.Accel() != "cpu" || v.SegmentLength() != DefaultSegmentLength || v.template != "{{ jinja }}" || v.instruction != DefaultInstruction {
		t.Errorf("vision = %+v", v)
	}
	want := []string{"load", "devices", "loadModel", "chatTemplate", "nCtxTrain", "newVisionContext", "loadProjector", "applyTemplate", "tokenizeMedia", "memoryClear", "evalChunks", "nEmbdOut", "embeddings", "freeChunks"}
	if got := f.names(); strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("calls = %v\nwant    %v", got, want)
	}
	if args := f.only(t, "newVisionContext"); args[1] != int32(DefaultVisionContextTokens) || args[2] != int32(visionBatch) {
		t.Errorf("context = %v, want %d tokens and %d-token batches", args, DefaultVisionContextTokens, visionBatch)
	}
	if args := f.only(t, "loadProjector"); args[3] != false || args[5] != 512 {
		t.Errorf("loadProjector = %v, want no GPU and 512 max tokens", args)
	}
	if args := f.only(t, "applyTemplate"); args[1] != DefaultInstruction || args[2] != "dimension probe" {
		t.Errorf("probe prompt = %v", args)
	}
}

// TestVision_CloseFreesOnceAndBlocksCalls: Close frees the projector, the
// context and the model once, and every later call fails with
// langai.ErrClosed without reaching llama.cpp.
func TestVision_CloseFreesOnceAndBlocksCalls(t *testing.T) {
	f := newFakeVision()
	v := openVisionFake(t, f, VisionConfig{Cutter: &fakeCutter{durations: map[string]time.Duration{"clip.mp4": time.Second}}})
	for range 2 {
		if err := v.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}
	if f.count("freeProjector") != 1 || f.count("freeContext") != 1 || f.count("freeModel") != 1 {
		t.Errorf("Close freed %v", f.names())
	}
	f.calls = nil
	ctx := context.Background()
	if _, err := v.Embed(ctx, langai.RoleQuery, []string{"x"}); !errors.Is(err, langai.ErrClosed) {
		t.Errorf("Embed after Close = %v, want langai.ErrClosed", err)
	}
	if _, err := v.EmbedImages(ctx, []image.Image{pixel(2, 2, color.RGBA{})}); !errors.Is(err, langai.ErrClosed) {
		t.Errorf("EmbedImages after Close = %v, want langai.ErrClosed", err)
	}
	if _, err := v.EmbedVideo(ctx, "/media/clip.mp4"); !errors.Is(err, langai.ErrClosed) {
		t.Errorf("EmbedVideo after Close = %v, want langai.ErrClosed", err)
	}
	if len(f.calls) != 0 {
		t.Errorf("a closed Vision reached llama.cpp: %v", f.names())
	}
}

// TestOpenVision_GPUAndTemplateDefaults: a GPU offloads the projector too,
// and a GGUF without a template gets chatml.
func TestOpenVision_GPUAndTemplateDefaults(t *testing.T) {
	f := newFakeVision()
	f.devs = []device{{name: "Vulkan0", backend: "Vulkan", description: "GPU", gpu: true}}
	v, err := openVision(f, &runtimeLoader{}, testLibDir, testVisionSpec(), VisionConfig{Instruction: "Find duplicates."}, Runtime{Threads: 2})
	if err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "loadProjector"); args[3] != true {
		t.Errorf("loadProjector = %v, want the GPU", args)
	}
	if args := f.only(t, "loadModel"); args[1] != allLayers {
		t.Errorf("loadModel = %v, want all layers offloaded", args)
	}
	if v.template != "chatml" || v.instruction != "Find duplicates." || v.Accel() != "Vulkan0: GPU" {
		t.Errorf("vision = %+v", v)
	}

}

// TestOpenVision_FreesOnFailure: every failure frees what was made before it.
func TestOpenVision_FreesOnFailure(t *testing.T) {
	boom := errors.New("boom")
	cases := []struct {
		name  string
		set   func(f *fakeVision)
		freed []string
	}{
		{"load", func(f *fakeVision) { f.loadErr = boom }, nil},
		{"loadModel", func(f *fakeVision) { f.loadModelErr = boom }, nil},
		{"no window", func(f *fakeVision) { f.trainCtx = 0 }, []string{"freeModel"}},
		{"context", func(f *fakeVision) { f.newContextErr = boom }, []string{"freeModel"}},
		{"empty context", func(f *fakeVision) { f.grantCtx = -1 }, []string{"freeContext", "freeModel"}},
		{"projector", func(f *fakeVision) { f.projectorErr = boom }, []string{"freeContext", "freeModel"}},
		{"probe", func(f *fakeVision) { f.evalErr = boom }, []string{"freeProjector", "freeContext", "freeModel"}},
		{"dims", func(f *fakeVision) { f.buf = []float32{1, 2, 3} }, []string{"freeProjector", "freeContext", "freeModel"}},
	}
	for _, c := range cases {
		f := newFakeVision()
		c.set(f)
		if _, err := openVision(f, &runtimeLoader{}, testLibDir, testVisionSpec(), VisionConfig{}, Runtime{Threads: 2, Accel: AccelCPU}); err == nil {
			t.Errorf("%s: want error", c.name)
			continue
		}
		var frees []string
		for _, n := range f.names() {
			if strings.HasPrefix(n, "free") && n != "freeChunks" && n != "freeBitmap" {
				frees = append(frees, n)
			}
		}
		if strings.Join(frees, " ") != strings.Join(c.freed, " ") {
			t.Errorf("%s: freed %v, want %v", c.name, frees, c.freed)
		}
	}
}

func TestVision_EmbedTextsAndImages(t *testing.T) {
	f := newFakeVision()
	v := openVisionFake(t, f, VisionConfig{})
	ctx := context.Background()

	vecs, err := v.Embed(ctx, langai.RoleDocument, []string{"a dog", "a cat"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 || math.Abs(float64(vecs[0][0])-0.6) > 1e-6 || math.Abs(float64(vecs[0][1])-0.8) > 1e-6 {
		t.Errorf("vectors = %v, want the fake's (3,4,0,0) normalized", vecs)
	}
	if f.lastPrompt != "<s>"+DefaultInstruction+"\na cat\n<a>" || f.lastBitmaps != 0 {
		t.Errorf("text prompt = %q with %d bitmaps", f.lastPrompt, f.lastBitmaps)
	}
	if f.count("tokenizeMedia") != 2 || f.count("evalChunks") != 2 || f.count("freeChunks") != 2 || f.count("memoryClear") != 2 {
		t.Errorf("calls = %v", f.names())
	}

}

func TestVision_EmbedImages(t *testing.T) {
	f := newFakeVision()
	v := openVisionFake(t, f, VisionConfig{})
	ctx := context.Background()
	red := color.RGBA{200, 30, 30, 255}
	big := image.NewRGBA(image.Rect(0, 0, 4000, 2000))
	vecs, err := v.EmbedImages(ctx, []image.Image{pixel(8, 6, red), big})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 || f.lastPrompt != "<s>"+DefaultInstruction+"\n<__media__>\n<a>" || f.lastBitmaps != 1 {
		t.Errorf("image prompt = %q with %d bitmaps", f.lastPrompt, f.lastBitmaps)
	}
	bitmaps := f.argsOf("bitmapRGB")
	if len(bitmaps) != 2 || bitmaps[0][0] != 8 || bitmaps[0][1] != 6 || bitmaps[0][2] != 8*6*3 {
		t.Errorf("first bitmap = %v, want 8x6 RGB", bitmaps)
	}
	if bitmaps[1][0] != maxImageSide || bitmaps[1][1] != maxImageSide/2 {
		t.Errorf("large picture = %v, want scaled to %d on the long side", bitmaps[1], maxImageSide)
	}
	// the bitmap is freed after the chunks made from it
	names := f.names()
	if idx(names, "freeBitmap") < idx(names, "freeChunks") {
		t.Errorf("bitmap freed before its chunks: %v", names)
	}
	if _, err := v.EmbedImages(ctx, []image.Image{nil}); err == nil || !strings.Contains(err.Error(), "empty image") {
		t.Errorf("nil image: %v", err)
	}
	if _, err := v.EmbedImages(ctx, []image.Image{image.NewRGBA(image.Rect(0, 0, 0, 0))}); err == nil {
		t.Error("empty image: want error")
	}
	f.bitmapErr = errors.New("boom")
	if _, err := v.EmbedImages(ctx, []image.Image{pixel(2, 2, red)}); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("bitmap error: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := v.Embed(cancelled, langai.RoleQuery, []string{"x"}); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled: %v", err)
	}
}

func TestVision_TokenBudgetAndErrors(t *testing.T) {
	f := newFakeVision()
	f.tokensPerImage = 20000
	v := openVisionFake(t, f, VisionConfig{ContextTokens: 8192})
	ctx := context.Background()
	if v.nCtx != 8192 {
		t.Errorf("nCtx = %d", v.nCtx)
	}
	_, err := v.EmbedImages(ctx, []image.Image{pixel(2, 2, color.RGBA{})})
	if err == nil || !strings.Contains(err.Error(), "exceeds the 8192-token context") {
		t.Errorf("over budget: %v", err)
	}
	if f.count("evalChunks") != 0 || f.count("freeChunks") != 1 || f.count("freeBitmap") != 1 {
		t.Errorf("over budget still evaluated, or leaked: %v", f.names())
	}
	for _, c := range []struct {
		name string
		set  func()
	}{
		{"template", func() { f.templateErr = errors.New("boom") }},
		{"tokenize", func() { f.tokenizeErr = errors.New("boom") }},
		{"eval", func() { f.evalErr = errors.New("boom") }},
		{"embeddings", func() { f.embeddingsErr = errors.New("boom") }},
		{"empty vector", func() { f.buf = nil }},
	} {
		f = newFakeVision()
		v = openVisionFake(t, f, VisionConfig{})
		c.set()
		if _, err := v.Embed(ctx, langai.RoleQuery, []string{"x"}); err == nil {
			t.Errorf("%s: want error", c.name)
		}
	}
	if _, err := openVision(newFakeVision(), &runtimeLoader{}, testLibDir, testVisionSpec(), VisionConfig{}, Runtime{Threads: 2, Accel: AccelCUDA}); err == nil || !strings.Contains(err.Error(), "no CUDA device") {
		t.Errorf("missing accelerator: %v", err)
	}
}

func TestVision_EmbedVideoInSegments(t *testing.T) {
	f := newFakeVision()
	cutter := &fakeCutter{durations: map[string]time.Duration{"long.mp4": 45 * time.Second, "short.mp4": 12 * time.Second, "exact.mp4": 20 * time.Second}}
	v := openVisionFake(t, f, VisionConfig{Cutter: cutter, FPS: 2})
	ctx := context.Background()

	segs, err := v.EmbedVideo(ctx, "/media/long.mp4")
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 3 || segs[0].Start != 0 || segs[0].End != 20*time.Second || segs[2].Start != 40*time.Second || segs[2].End != 45*time.Second {
		t.Errorf("segments = %+v", segs)
	}
	if strings.Join(cutter.cuts, " ") != "0s/20s/segment-000.mp4 20s/20s/segment-001.mp4 40s/20s/segment-002.mp4" {
		t.Errorf("cuts = %v", cutter.cuts)
	}
	if videos := f.argsOf("bitmapVideo"); len(videos) != 3 || videos[0][1] != "segment-000.mp4" || videos[0][2] != 2.0 {
		t.Errorf("bitmapVideo = %v", videos)
	}
	if f.count("freeVideo") != 3 || f.count("freeBitmap") != 3 || f.lastPrompt != "<s>"+DefaultInstruction+"\n<__media__>\n<a>" {
		t.Errorf("frees %d/%d, prompt %q", f.count("freeVideo"), f.count("freeBitmap"), f.lastPrompt)
	}

}

// TestVision_EmbedVideoShortAndCapped: a short video is one input, uncut, one
// exactly a segment long too, and MaxSegments caps the work.
func TestVision_EmbedVideoShortAndCapped(t *testing.T) {
	f := newFakeVision()
	cutter := &fakeCutter{durations: map[string]time.Duration{"long.mp4": 45 * time.Second, "short.mp4": 12 * time.Second, "exact.mp4": 20 * time.Second}}
	v := openVisionFake(t, f, VisionConfig{Cutter: cutter})
	ctx := context.Background()
	var segs []media.Segment
	var err error
	for _, name := range []string{"short.mp4", "exact.mp4"} {
		segs, err = v.EmbedVideo(ctx, "/media/"+name)
		if err != nil || len(segs) != 1 || segs[0].Start != 0 || segs[0].End != cutter.durations[name] {
			t.Errorf("%s: %+v, %v", name, segs, err)
		}
	}
	if len(cutter.cuts) != 0 {
		t.Errorf("short videos were cut: %v", cutter.cuts)
	}
	if videos := f.argsOf("bitmapVideo"); videos[len(videos)-1][1] != "exact.mp4" {
		t.Errorf("the whole file should be embedded: %v", videos[len(videos)-1])
	}

	// MaxSegments caps the work
	v = openVisionFake(t, f, VisionConfig{Cutter: cutter, MaxSegments: 2})
	if segs, err = v.EmbedVideo(ctx, "/media/long.mp4"); err != nil || len(segs) != 2 {
		t.Errorf("capped: %d segments, %v", len(segs), err)
	}

}

// TestVision_EmbedVideoErrors: no cutter, an unknown file, a failing cut, a
// failing read.
func TestVision_EmbedVideoErrors(t *testing.T) {
	f := newFakeVision()
	cutter := &fakeCutter{durations: map[string]time.Duration{"long.mp4": 45 * time.Second, "short.mp4": 12 * time.Second}}
	ctx := context.Background()
	v := openVisionFake(t, f, VisionConfig{})
	if _, err := v.EmbedVideo(ctx, "/media/long.mp4"); err == nil || !strings.Contains(err.Error(), "Cutter") {
		t.Errorf("no cutter: %v", err)
	}
	v = openVisionFake(t, f, VisionConfig{Cutter: cutter})
	if _, err := v.EmbedVideo(ctx, "/media/nope.mp4"); err == nil {
		t.Error("unknown file: want error")
	}
	cutter.err = errors.New("boom")
	if _, err := v.EmbedVideo(ctx, "/media/long.mp4"); err == nil || !strings.Contains(err.Error(), "cut segment 0") {
		t.Errorf("cut error: %v", err)
	}
	cutter.err = nil
	f.videoErr = errors.New("unreadable")
	if _, err := v.EmbedVideo(ctx, "/media/short.mp4"); err == nil || !strings.Contains(err.Error(), "unreadable") {
		t.Errorf("read error: %v", err)
	}
}

func TestOpenVision_ValidatesSpecAndConfig(t *testing.T) {
	ctx := context.Background()
	if _, err := OpenVision(ctx, VisionSpec{}, VisionConfig{}); err == nil {
		t.Error("empty spec: want error")
	}
	if _, err := OpenVision(ctx, VisionSpec{Name: "x", File: "m"}, VisionConfig{}); err == nil {
		t.Error("no projector: want error")
	}
	if _, err := OpenVision(ctx, VisionSpec{Name: "x", File: "m", Projector: "p", URL: "https://x"}, VisionConfig{}); err == nil {
		t.Error("URL without digest: want error")
	}
	if _, err := OpenVision(ctx, testVisionSpec(), VisionConfig{FPS: -1}); err == nil {
		t.Error("negative fps: want error")
	}
	if _, err := OpenVision(ctx, testVisionSpec(), VisionConfig{}); err == nil {
		t.Error("no LibDir: want error")
	}
	if _, err := NewVision(ctx, VisionConfig{Config: Config{Model: "org/other"}}); err == nil || !strings.Contains(err.Error(), DefaultVisionModel) {
		t.Errorf("unknown model: %v", err)
	}
}

func TestLookupVisionSpec(t *testing.T) {
	spec, err := LookupVisionSpec("", "")
	if err != nil {
		t.Fatal(err)
	}
	if spec.Name != DefaultVisionModel || spec.File != filepath.Join("models", "Qwen3-VL-Embedding-2B-Q8_0.gguf") || spec.Projector != filepath.Join("models", "mmproj-Qwen3-VL-Embedding-2B-f16.gguf") || spec.Dims != 2048 {
		t.Errorf("default spec = %+v", spec)
	}
	for name, a := range map[string]asset{"model": {URL: spec.URL, SHA256: spec.SHA256, Size: spec.Size}, "projector": {URL: spec.ProjectorURL, SHA256: spec.ProjectorSHA256, Size: spec.ProjectorSize}} {
		if !strings.HasPrefix(a.URL, "https://") || len(a.SHA256) != 64 || a.Size <= 0 {
			t.Errorf("%s: %+v is not well pinned", name, a)
		}
	}
	if d := DefaultVisionSpec("/m"); d.File != "/m/Qwen3-VL-Embedding-2B-Q8_0.gguf" {
		t.Errorf("DefaultVisionSpec = %+v", d)
	}
}

// TestNewVision_DownloadsAndLoads: the model and the projector are fetched
// into ModelsDir, the libraries into LibDir, and the load is logged.
func TestNewVision_DownloadsAndLoads(t *testing.T) {
	_, libAsset := serve(t, tarGz(t, "llama-b11429", releaseLibs()))
	_, modelAsset := serve(t, []byte("GGUF fake model"))
	_, projAsset := serve(t, []byte("GGUF fake projector"))
	offline(t)
	cpuOnly(t)
	key := platform() + "/cpu"
	origLib, hadLib := libAssets[key]
	libAssets[key] = libAsset
	origVision, origLibs := visionLib, libs
	f := newFakeVision()
	f.buf = make([]float32, 2048)
	f.buf[0] = 1
	visionLib, libs = f, &runtimeLoader{}
	t.Cleanup(func() {
		if hadLib {
			libAssets[key] = origLib
		} else {
			delete(libAssets, key)
		}
		visionLib, libs = origVision, origLibs
	})
	spec := DefaultVisionSpec(filepath.Join(t.TempDir(), "models"))
	spec.URL, spec.SHA256, spec.Size = modelAsset.URL, modelAsset.SHA256, modelAsset.Size
	spec.ProjectorURL, spec.ProjectorSHA256, spec.ProjectorSize = projAsset.URL, projAsset.SHA256, projAsset.Size
	var logs bytes.Buffer
	libDir := filepath.Join(t.TempDir(), "llama-cpp", LibTag)
	v, err := OpenVision(context.Background(), spec, VisionConfig{Config: Config{LibDir: libDir, ManagedLibDir: true, Threads: 2, Logger: slog.New(slog.NewTextHandler(&logs, nil))}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = v.Close() }()
	for _, p := range []string{spec.File, spec.Projector, filepath.Join(libDir, "libllama.so")} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s was not installed: %v", p, err)
		}
	}
	if args := f.only(t, "loadModel"); args[0] != spec.File {
		t.Errorf("loadModel = %v", args)
	}
	if args := f.only(t, "loadProjector"); args[0] != spec.Projector {
		t.Errorf("loadProjector = %v", args)
	}
	if !strings.Contains(logs.String(), "llama.cpp loaded") || !strings.Contains(logs.String(), "dims=2048") {
		t.Errorf("logs:\n%s", logs.String())
	}
}

func TestRGBBytes(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	img.Set(0, 0, color.NRGBA{255, 0, 0, 255})
	img.Set(1, 0, color.NRGBA{0, 0, 255, 0}) // transparent: composed on white
	w, h, rgb := rgbBytes(img)
	if w != 2 || h != 1 || !bytes.Equal(rgb, []byte{255, 0, 0, 255, 255, 255}) {
		t.Errorf("rgbBytes = %d×%d %v", w, h, rgb)
	}
	w, h, rgb = rgbBytes(image.NewRGBA(image.Rect(10, 10, 10+2*maxImageSide, 10+maxImageSide)))
	if w != maxImageSide || h != maxImageSide/2 || len(rgb) != w*h*3 {
		t.Errorf("scaled = %d×%d, %d bytes", w, h, len(rgb))
	}
}

// pixel is a w×h picture of one colour.
func pixel(w, h int, c color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, c)
		}
	}
	return img
}

// idx is the position of name in names, or -1.
func idx(names []string, name string) int {
	for i, n := range names {
		if n == name {
			return i
		}
	}
	return -1
}

// argsOf returns the arguments of every call to the named method, in order.
func (l *callLog) argsOf(name string) [][]any {
	var out [][]any
	for _, c := range l.calls {
		if c.name == name {
			out = append(out, c.args)
		}
	}
	return out
}

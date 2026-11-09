package onnx

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"log/slog"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-bumbu/langai"
)

// fakeAPI is a sessionAPI without native code. It records every call and
// turns each ONNX Runtime failure mode into a knob, so runtimeLoader, Open
// and Embedder run their real code against it.
type fakeAPI struct {
	loadErr  error  // load fails with it
	reported string // the version reported; "" reports LibVersion
	openErr  error  // open fails with it
	dims     int    // what open reports the model file declares; 0 = dynamic
	vecDims  int    // the vector length run produces; 0 = dims, or 4 when that is dynamic
	tokens   int    // > 0: run answers a token sequence [n, tokens, dims] instead of [n, dims]
	runErr   error  // run fails with it
	short    bool   // run answers one value too few

	loads    []string
	opens    []sessionOptions
	sessions []*fakeSession
}

// verify the fake actually satisfies the interface
var _ sessionAPI = (*fakeAPI)(nil)

type fakeSession struct {
	api    *fakeAPI
	runs   []int // the batch size of every run
	closed int
}

func (f *fakeAPI) load(p string, _ *slog.Logger) error {
	f.loads = append(f.loads, p)
	return f.loadErr
}

func (f *fakeAPI) version() string {
	if f.reported == "" {
		return LibVersion
	}
	return f.reported
}

func (f *fakeAPI) open(_ string, o sessionOptions) (session, int, error) {
	f.opens = append(f.opens, o)
	if f.openErr != nil {
		return nil, 0, f.openErr
	}
	s := &fakeSession{api: f}
	f.sessions = append(f.sessions, s)
	return s, f.dims, nil
}

// run answers each picture with its three channel means followed by ones,
// so different pictures get different vectors and the same picture always
// the same. With tokens set, that vector is the first token and every
// further token is all twos, so CLS and mean pooling differ.
func (s *fakeSession) run(pixels []float32, n, side int) ([]float32, []int64, error) {
	s.runs = append(s.runs, n)
	if s.api.runErr != nil {
		return nil, nil, s.api.runErr
	}
	dims := s.api.vecDims
	if dims == 0 {
		dims = max(s.api.dims, 4)
	}
	tokens := max(s.api.tokens, 1)
	plane := side * side
	out := make([]float32, 0, n*tokens*dims)
	for i := range n {
		px := pixels[i*3*plane : (i+1)*3*plane]
		vec := make([]float32, dims)
		for c := range 3 {
			var sum float32
			for j := range plane {
				sum += px[c*plane+j]
			}
			if c < dims {
				vec[c] = sum / float32(plane)
			}
		}
		for j := 3; j < dims; j++ {
			vec[j] = 1
		}
		out = append(out, vec...)
		for range tokens - 1 {
			for range dims {
				out = append(out, 2)
			}
		}
	}
	if s.api.short {
		out = out[:len(out)-1]
	}
	// a short answer keeps the intended shape, which then disagrees with
	// the values
	shape := []int64{int64(n), int64(dims)}
	if s.api.tokens > 0 {
		shape = []int64{int64(n), int64(tokens), int64(dims)}
	}
	return out, shape, nil
}

func (s *fakeSession) close() { s.closed++ }

// useFake swaps the package's runtime surface and loader for the fake for
// the duration of the test, and keeps the test from downloading a real
// model or runtime: requests to anything but the loopback test servers fail.
func useFake(t *testing.T, f *fakeAPI) {
	t.Helper()
	origAPI, origLibs, origClient := api, libs, httpClient
	api, libs, httpClient = f, &runtimeLoader{}, &http.Client{Transport: loopbackOnly{}}
	t.Cleanup(func() { api, libs, httpClient = origAPI, origLibs, origClient })
}

// loopbackOnly is a RoundTripper that only reaches httptest servers.
type loopbackOnly struct{}

func (loopbackOnly) RoundTrip(r *http.Request) (*http.Response, error) {
	if h := r.URL.Hostname(); h != "127.0.0.1" && h != "::1" && h != "localhost" {
		return nil, fmt.Errorf("test is offline, refusing %s", r.URL)
	}
	return http.DefaultTransport.RoundTrip(r)
}

// fakeInstall lays out an operator library directory with an empty library
// file and a model file, so Open passes the on-disk checks without
// downloading.
func fakeInstall(t *testing.T) (libDir, modelFile string) {
	t.Helper()
	libDir = filepath.Join(t.TempDir(), "onnxruntime")
	if err := os.MkdirAll(libDir, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(libDir, libFile()), []byte("not really"), 0o600); err != nil {
		t.Fatal(err)
	}
	modelFile = filepath.Join(t.TempDir(), "model.onnx")
	if err := os.WriteFile(modelFile, []byte("not really"), 0o600); err != nil {
		t.Fatal(err)
	}
	return libDir, modelFile
}

// testSpec is a small model: 8-pixel squares, 4-dimensional vectors.
func testSpec(file string) ModelSpec {
	return ModelSpec{Name: "test/model", File: file, Side: 8, Dims: 4}
}

func solid(c color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for i := range 16 * 16 {
		img.Pix[i*4], img.Pix[i*4+1], img.Pix[i*4+2], img.Pix[i*4+3] = c.R, c.G, c.B, 255
	}
	return img
}

func TestRuntimeLoader_LoadsOnceAndLocksVersionAndPath(t *testing.T) {
	f := &fakeAPI{}
	l := &runtimeLoader{}
	for range 2 {
		if err := l.load(f, "/opt/ort/lib.so", discardLogger()); err != nil {
			t.Fatal(err)
		}
	}
	if len(f.loads) != 1 || f.loads[0] != "/opt/ort/lib.so" {
		t.Errorf("loads = %v, want one", f.loads)
	}
	if l.version != LibVersion {
		t.Errorf("version = %q", l.version)
	}
	err := l.load(f, "/elsewhere/lib.so", discardLogger())
	if err == nil || !strings.Contains(err.Error(), "already loaded") {
		t.Errorf("second path: err = %v", err)
	}

	f = &fakeAPI{reported: "1.30.0"}
	l = &runtimeLoader{}
	err = l.load(f, "/opt/ort/lib.so", discardLogger())
	if err == nil || !strings.Contains(err.Error(), "1.30.0") || !strings.Contains(err.Error(), LibVersion) {
		t.Errorf("other version: err = %v", err)
	}
	f = &fakeAPI{reported: libMajorMinor}
	if err := (&runtimeLoader{}).load(f, "/opt/ort/lib.so", discardLogger()); err != nil {
		t.Errorf("a bare major.minor must pass: %v", err)
	}

	f = &fakeAPI{loadErr: errors.New("dlopen failed")}
	l = &runtimeLoader{}
	for range 2 {
		if err := l.load(f, "/opt/ort/lib.so", discardLogger()); err == nil || !strings.Contains(err.Error(), "dlopen failed") {
			t.Errorf("err = %v, want the cached load error", err)
		}
	}
	if len(f.loads) != 1 {
		t.Errorf("a failed load was retried: %v", f.loads)
	}
}

func TestOpen_LoadsOpensProbesAndLogs(t *testing.T) {
	f := &fakeAPI{dims: 4}
	useFake(t, f)
	libDir, modelFile := fakeInstall(t)
	var log bytes.Buffer
	emb, err := Open(context.Background(), testSpec(modelFile), Runtime{
		LibDir: libDir, Provider: ProviderCUDA, DeviceID: 1, Threads: 2, Logger: slog.New(slog.NewTextHandler(&log, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = emb.Close() }()
	if emb.Dimensions() != 4 {
		t.Errorf("Dimensions = %d", emb.Dimensions())
	}
	if len(f.loads) != 1 || f.loads[0] != filepath.Join(libDir, libFile()) {
		t.Errorf("loads = %v", f.loads)
	}
	want := sessionOptions{input: "pixel_values", output: "pooler_output", provider: ProviderCUDA, deviceID: 1, threads: 2}
	if len(f.opens) != 1 || f.opens[0] != want {
		t.Errorf("opens = %+v, want %+v", f.opens, want)
	}
	if runs := f.sessions[0].runs; len(runs) != 1 || runs[0] != 1 {
		t.Errorf("probe runs = %v, want one picture", runs)
	}
	for _, s := range []string{"onnxruntime loaded", "provider=cuda", "model=test/model", "version=" + LibVersion} {
		if !strings.Contains(log.String(), s) {
			t.Errorf("log lacks %q:\n%s", s, log.String())
		}
	}
	// a second embedder shares the loaded runtime
	emb2, err := Open(context.Background(), testSpec(modelFile), Runtime{LibDir: libDir})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = emb2.Close() }()
	if len(f.loads) != 1 || len(f.opens) != 2 {
		t.Errorf("second Open: %d loads, %d opens", len(f.loads), len(f.opens))
	}
	if f.opens[1].provider != ProviderCPU || f.opens[1].threads != defaultThreads() {
		t.Errorf("defaults: %+v", f.opens[1])
	}
}

func TestOpen_Errors(t *testing.T) {
	ctx := context.Background()
	libDir, modelFile := fakeInstall(t)

	f := &fakeAPI{dims: 4}
	useFake(t, f)
	cases := map[string]struct {
		spec ModelSpec
		rt   Runtime
		want string
	}{
		"bad spec":        {ModelSpec{File: modelFile, Side: 8}, Runtime{LibDir: libDir}, "Name"},
		"bad provider":    {testSpec(modelFile), Runtime{LibDir: libDir, Provider: "tpu"}, "provider"},
		"no lib dir":      {testSpec(modelFile), Runtime{}, "LibDir"},
		"empty lib dir":   {testSpec(modelFile), Runtime{LibDir: t.TempDir()}, libFile()},
		"no model":        {testSpec(filepath.Join(t.TempDir(), "gone.onnx")), Runtime{LibDir: libDir}, "no URL"},
		"dims mismatch":   {ModelSpec{Name: "m", File: modelFile, Side: 8, Dims: 5}, Runtime{LibDir: libDir}, "declares 4"},
		"another lib dir": {testSpec(modelFile), Runtime{LibDir: t.TempDir()}, "already loaded"},
	}
	// "another lib dir" needs a loaded runtime first, and a library file
	another := cases["another lib dir"]
	if err := os.WriteFile(filepath.Join(another.rt.LibDir, libFile()), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(ctx, testSpec(modelFile), Runtime{LibDir: libDir}); err != nil {
		t.Fatal(err)
	}
	for name, c := range cases {
		_, err := Open(ctx, c.spec, c.rt)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", name, err, c.want)
		}
	}
	// a session that was opened but fails its probe or its size check is closed again
	for name, f := range map[string]*fakeAPI{
		"probe fails":   {dims: 4, runErr: errors.New("cuda out of memory")},
		"dims mismatch": {dims: 5},
		"short output":  {dims: 4, short: true},
	} {
		useFake(t, f)
		if _, err := Open(ctx, testSpec(modelFile), Runtime{LibDir: libDir}); err == nil {
			t.Errorf("%s: want an error", name)
		}
		if len(f.sessions) != 1 || f.sessions[0].closed != 1 {
			t.Errorf("%s: the session was not closed once: %+v", name, f.sessions)
		}
	}
	useFake(t, &fakeAPI{openErr: errors.New("no such tensor")})
	if _, err := Open(ctx, testSpec(modelFile), Runtime{LibDir: libDir}); err == nil || !strings.Contains(err.Error(), "no such tensor") {
		t.Errorf("open fails: err = %v", err)
	}
	useFake(t, &fakeAPI{loadErr: errors.New("dlopen")})
	if _, err := Open(ctx, testSpec(modelFile), Runtime{LibDir: libDir}); err == nil || !strings.Contains(err.Error(), "dlopen") {
		t.Errorf("load fails: err = %v", err)
	}
}

func TestEmbedder_BatchesAndNormalizes(t *testing.T) {
	f := &fakeAPI{dims: 4}
	useFake(t, f)
	libDir, modelFile := fakeInstall(t)
	emb, err := Open(context.Background(), testSpec(modelFile), Runtime{LibDir: libDir, BatchSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = emb.Close() }()
	red, blue := solid(color.RGBA{255, 0, 0, 255}), solid(color.RGBA{0, 0, 255, 255})
	vecs, err := emb.EmbedImages(context.Background(), []image.Image{red, blue, red, red, blue})
	if err != nil {
		t.Fatal(err)
	}
	if runs := f.sessions[0].runs; len(runs) != 4 || runs[1] != 2 || runs[2] != 2 || runs[3] != 1 {
		t.Errorf("runs = %v, want the probe then 2, 2, 1", runs)
	}
	if len(vecs) != 5 {
		t.Fatalf("got %d vectors", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != 4 {
			t.Fatalf("vector %d has %d dims", i, len(v))
		}
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		if math.Abs(sum-1) > 1e-5 {
			t.Errorf("vector %d is not unit length: %f", i, sum)
		}
	}
	if vecs[0][0] != vecs[2][0] || vecs[0][0] == vecs[1][0] {
		t.Errorf("same pictures must agree and different ones differ: %v", vecs)
	}
	if out, err := emb.EmbedImages(context.Background(), nil); err != nil || len(out) != 0 {
		t.Errorf("no images: %v, %v", out, err)
	}
}

func TestEmbedder_DynamicDims(t *testing.T) {
	libDir, modelFile := fakeInstall(t)
	// the model file leaves the size open and so does the spec: the probe decides
	useFake(t, &fakeAPI{dims: 0, vecDims: 6})
	emb, err := Open(context.Background(), ModelSpec{Name: "m", File: modelFile, Side: 8}, Runtime{LibDir: libDir})
	if err != nil {
		t.Fatal(err)
	}
	if emb.Dimensions() != 6 {
		t.Errorf("Dimensions = %d, want the probe's 6", emb.Dimensions())
	}
	_ = emb.Close()
	// the spec says: the probe must agree
	useFake(t, &fakeAPI{dims: 0, vecDims: 6})
	if _, err := Open(context.Background(), ModelSpec{Name: "m", File: modelFile, Side: 8, Dims: 5}, Runtime{LibDir: libDir}); err == nil || !strings.Contains(err.Error(), "expected 5") {
		t.Errorf("err = %v, want the probe's size refused", err)
	}
	useFake(t, &fakeAPI{dims: 0, vecDims: 5})
	emb, err = Open(context.Background(), ModelSpec{Name: "m", File: modelFile, Side: 8, Dims: 5}, Runtime{LibDir: libDir})
	if err != nil {
		t.Fatal(err)
	}
	if emb.Dimensions() != 5 {
		t.Errorf("Dimensions = %d, want 5", emb.Dimensions())
	}
	_ = emb.Close()
}

func TestEmbedder_ClosedCancelledAndBadInput(t *testing.T) {
	f := &fakeAPI{dims: 4}
	useFake(t, f)
	libDir, modelFile := fakeInstall(t)
	emb, err := Open(context.Background(), testSpec(modelFile), Runtime{LibDir: libDir})
	if err != nil {
		t.Fatal(err)
	}
	img := solid(color.RGBA{255, 0, 0, 255})
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := emb.EmbedImages(cancelled, []image.Image{img}); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled: err = %v", err)
	}
	if _, err := emb.EmbedImages(context.Background(), []image.Image{nil}); err == nil {
		t.Error("nil image: want an error")
	}
	f.runErr = errors.New("device lost")
	if _, err := emb.EmbedImages(context.Background(), []image.Image{img}); err == nil || !strings.Contains(err.Error(), "device lost") {
		t.Errorf("run fails: err = %v", err)
	}
	f.runErr = nil
	f.short = true
	if _, err := emb.EmbedImages(context.Background(), []image.Image{img}); err == nil || !strings.Contains(err.Error(), "values") {
		t.Errorf("short output: err = %v", err)
	}
	f.short = false
	for range 2 {
		if err := emb.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if f.sessions[0].closed != 1 {
		t.Errorf("closed %d times", f.sessions[0].closed)
	}
	if _, err := emb.EmbedImages(context.Background(), []image.Image{img}); !errors.Is(err, langai.ErrClosed) {
		t.Errorf("after Close: err = %v, want langai.ErrClosed", err)
	}
}

func TestEmbedder_PoolsTokenSequences(t *testing.T) {
	libDir, modelFile := fakeInstall(t)
	img := solid(color.RGBA{255, 0, 0, 255})
	// cls: the first token, the channel means then ones
	useFake(t, &fakeAPI{dims: 4, tokens: 3})
	spec := ModelSpec{Name: "m", File: modelFile, Side: 8, Dims: 4, Output: "last_hidden_state", Pooling: PoolingCLS}
	emb, err := Open(context.Background(), spec, Runtime{LibDir: libDir})
	if err != nil {
		t.Fatal(err)
	}
	cls, err := emb.EmbedImages(context.Background(), []image.Image{img})
	if err != nil {
		t.Fatal(err)
	}
	_ = emb.Close()
	// mean: (token0 + 2 + 2) / 3 per dimension, normalized: a different direction
	useFake(t, &fakeAPI{dims: 4, tokens: 3})
	spec.Pooling = PoolingMean
	emb, err = Open(context.Background(), spec, Runtime{LibDir: libDir})
	if err != nil {
		t.Fatal(err)
	}
	mean, err := emb.EmbedImages(context.Background(), []image.Image{img})
	if err != nil {
		t.Fatal(err)
	}
	_ = emb.Close()
	if len(cls[0]) != 4 || len(mean[0]) != 4 {
		t.Fatalf("cls = %v, mean = %v", cls, mean)
	}
	var dot float64
	for i := range 4 {
		dot += float64(cls[0][i]) * float64(mean[0][i])
	}
	if dot > 0.999 {
		t.Errorf("cls %v and mean %v pooling agree, they must not", cls[0], mean[0])
	}
	// the pooling and the output's shape must fit each other
	for name, c := range map[string]struct {
		tokens  int
		pooling Pooling
	}{"sequence without pooling": {3, PoolingNone}, "cls on vectors": {0, PoolingCLS}} {
		useFake(t, &fakeAPI{dims: 4, tokens: c.tokens})
		spec.Pooling = c.pooling
		if _, err := Open(context.Background(), spec, Runtime{LibDir: libDir}); err == nil || !strings.Contains(err.Error(), "pooling") {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

func TestPool_RejectsShapesThatDoNotFit(t *testing.T) {
	cases := map[string]struct {
		p      Pooling
		values []float32
		shape  []int64
		n      int
	}{
		"rank 1":        {PoolingNone, []float32{1, 2}, []int64{2}, 1},
		"rank 4":        {PoolingNone, []float32{1, 2}, []int64{1, 1, 1, 2}, 1},
		"wrong batch":   {PoolingNone, []float32{1, 2}, []int64{2, 1}, 1},
		"wrong count":   {PoolingNone, []float32{1, 2, 3}, []int64{1, 2}, 1},
		"empty vectors": {PoolingNone, nil, []int64{1, 0}, 1},
		"no tokens":     {PoolingCLS, nil, []int64{1, 0, 4}, 1},
	}
	for name, c := range cases {
		if _, err := pool(c.p, c.values, c.shape, c.n); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
	out, err := pool(PoolingMean, []float32{1, 2, 3, 4, 10, 20, 30, 40}, []int64{2, 2, 2}, 2)
	if err != nil || len(out) != 2 || out[0][0] != 2 || out[0][1] != 3 || out[1][0] != 20 || out[1][1] != 30 {
		t.Errorf("mean = %v, %v", out, err)
	}
}

func TestNew_SelectsAPinnedModel(t *testing.T) {
	f := &fakeAPI{dims: 768}
	useFake(t, f)
	libDir, _ := fakeInstall(t)
	modelsDir := t.TempDir()
	if _, err := New(context.Background(), Config{Model: "nobody/nothing", LibDir: libDir, ModelsDir: modelsDir}); err == nil {
		t.Error("unknown model: want an error")
	}
	// a pre-seeded file that is not the pinned size is what a crash left: it is
	// fetched again (here: refused, the test is offline) rather than loaded
	seeded := filepath.Join(modelsDir, defaultModelFile)
	if err := os.WriteFile(seeded, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(context.Background(), Config{LibDir: libDir, ModelsDir: modelsDir}); err == nil || !strings.Contains(err.Error(), "offline") {
		t.Errorf("truncated model: err = %v, want a download attempt", err)
	}
	// the default model, pre-seeded (a sparse file of the pinned size) so
	// nothing is downloaded
	if err := os.Truncate(seeded, pinned[DefaultModel].Size); err != nil {
		t.Fatal(err)
	}
	emb, err := New(context.Background(), Config{LibDir: libDir, ModelsDir: modelsDir, Threads: 3, BatchSize: 16})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = emb.Close() }()
	if emb.Dimensions() != 768 || emb.batch != 16 || f.opens[0].threads != 3 {
		t.Errorf("New: dims %d, batch %d, opens %+v", emb.Dimensions(), emb.batch, f.opens)
	}
}

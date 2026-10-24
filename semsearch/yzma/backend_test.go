package yzma

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/semsearch"
)

// Handles the fake hands out. They are distinct so a test can tell which one
// a free call received.
const (
	fakeModel uintptr = 0x1000
	fakeCtx   uintptr = 0x2000
)

// Paths handed to openLlama; the fake only records them.
const (
	testLibDir    = "/opt/llama-cpp/" + LibTag
	testModelPath = "/opt/models/" + defaultModelFile
)

// fakeCall is one llamaAPI call the fake saw: the method and its arguments.
type fakeCall struct {
	name string
	args []any
}

// fakeAPI is a llamaAPI without native code. It records every call in order
// and turns each llama.cpp failure mode into a knob, so runtimeLoader,
// openLlama and llamaBackend run their real code against it.
type fakeAPI struct {
	loadErr       error   // load fails with it
	reported      string  // the version reported; "" reports libVersion
	loadModelErr  error   // loadModel fails with it
	trainCtx      int32   // the model's n_ctx_train
	grantedLimit  int     // the limit newContext grants
	mem           uintptr // the memory handle newContext returns; 0 = no KV cache
	newContextErr error   // newContext fails with it
	decodeRet     int32   // decode's return value
	decodeErr     error   // decode fails with it
	embeddingsErr error   // embeddings fails with it
	nilEmbeddings bool    // embeddings returns no vector
	// buf is the one buffer every embeddings call returns, standing in for
	// the llama.cpp memory the real view aliases; nEmbd is its length.
	buf []float32

	// backtraceAtLoad is GGML_NO_BACKTRACE as load saw it.
	backtraceAtLoad string
	calls           []fakeCall
}

// verify the fake actually satisfies the interface
var _ llamaAPI = (*fakeAPI)(nil)

// newFakeAPI returns a fake that loads, opens and embeds successfully: a
// model with a 512-token window that the context grants in full, no KV cache
// and 4-dimensional vectors.
func newFakeAPI() *fakeAPI {
	return &fakeAPI{trainCtx: 512, grantedLimit: 512, buf: []float32{1, 2, 3, 4}}
}

func (f *fakeAPI) record(name string, args ...any) {
	f.calls = append(f.calls, fakeCall{name: name, args: args})
}

// names lists the methods called, in order.
func (f *fakeAPI) names() []string {
	out := make([]string, len(f.calls))
	for i, c := range f.calls {
		out[i] = c.name
	}
	return out
}

// count reports how often the named method was called.
func (f *fakeAPI) count(name string) int {
	n := 0
	for _, c := range f.calls {
		if c.name == name {
			n++
		}
	}
	return n
}

// only returns the arguments of the single call to the named method and
// fails the test unless there was exactly one.
func (f *fakeAPI) only(t *testing.T, name string) []any {
	t.Helper()
	if n := f.count(name); n != 1 {
		t.Fatalf("%s was called %d times, want once; calls: %v", name, n, f.names())
	}
	for _, c := range f.calls {
		if c.name == name {
			return c.args
		}
	}
	return nil
}

func (f *fakeAPI) load(dir string, _ *slog.Logger) error {
	f.record("load", dir)
	f.backtraceAtLoad = os.Getenv("GGML_NO_BACKTRACE")
	return f.loadErr
}

func (f *fakeAPI) version() string {
	f.record("version")
	if f.reported == "" {
		return libVersion
	}
	return f.reported
}

func (f *fakeAPI) loadModel(path string) (uintptr, error) {
	f.record("loadModel", path)
	if f.loadModelErr != nil {
		return 0, f.loadModelErr
	}
	return fakeModel, nil
}

func (f *fakeAPI) freeModel(model uintptr) { f.record("freeModel", model) }

func (f *fakeAPI) nCtxTrain(model uintptr) int32 {
	f.record("nCtxTrain", model)
	return f.trainCtx
}

func (f *fakeAPI) nEmbd(model uintptr) int32 {
	f.record("nEmbd", model)
	return int32(len(f.buf))
}

func (f *fakeAPI) newContext(model uintptr, nCtx int32, threads int, pooling Pooling) (uintptr, int, uintptr, error) {
	f.record("newContext", model, nCtx, threads, pooling)
	if f.newContextErr != nil {
		return 0, 0, 0, f.newContextErr
	}
	return fakeCtx, f.grantedLimit, f.mem, nil
}

func (f *fakeAPI) freeContext(ctx uintptr) { f.record("freeContext", ctx) }

// tokenize uses stubBackend's scheme: [CLS], one token per rune, [SEP].
func (f *fakeAPI) tokenize(model uintptr, text string) []int32 {
	f.record("tokenize", model, text)
	return (&stubBackend{}).tokenize(text)
}

func (f *fakeAPI) memoryClear(mem uintptr) error {
	f.record("memoryClear", mem)
	return nil
}

func (f *fakeAPI) decode(ctx uintptr, tokens []int32) (int32, error) {
	f.record("decode", ctx, len(tokens))
	return f.decodeRet, f.decodeErr
}

func (f *fakeAPI) embeddings(ctx uintptr, n int32) ([]float32, error) {
	f.record("embeddings", ctx, n)
	if f.embeddingsErr != nil {
		return nil, f.embeddingsErr
	}
	if f.nilEmbeddings {
		return nil, nil
	}
	return f.buf[:n], nil
}

// openFake opens a backend over f through the real runtimeLoader and
// openLlama, then forgets the calls that made, so a test sees only what embed
// does.
func openFake(t *testing.T, f *fakeAPI) *llamaBackend {
	t.Helper()
	b, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	f.calls = nil
	return b
}

// TestRuntimeLoader_LoadsOnceAndRefusesAnotherDir: yzma binds the libraries
// into process globals, so a second load must not reload them, and a second
// directory cannot be honoured at all.
func TestRuntimeLoader_LoadsOnceAndRefusesAnotherDir(t *testing.T) {
	f := newFakeAPI()
	rt := &runtimeLoader{}
	for i := 1; i <= 2; i++ {
		if err := rt.load(f, "/libs/first", nil); err != nil {
			t.Fatalf("load %d: %v", i, err)
		}
	}
	if n := f.count("load"); n != 1 {
		t.Fatalf("api.load ran %d times for two loads of one directory, want 1", n)
	}
	err := rt.load(f, "/libs/second", nil)
	if err == nil || !strings.Contains(err.Error(), "/libs/first") || !strings.Contains(err.Error(), "/libs/second") {
		t.Fatalf("err = %v, want a refusal naming both directories", err)
	}
	if n := f.count("load"); n != 1 {
		t.Fatalf("the refused directory reached api.load (%d loads)", n)
	}
}

// TestRuntimeLoader_LoadErrorIsCached: a failed load is permanent for the
// process, matching semsearch.LazyEmbedder's cached error, and nothing else is
// called on libraries that did not load.
func TestRuntimeLoader_LoadErrorIsCached(t *testing.T) {
	boom := errors.New("dlopen failed")
	f := newFakeAPI()
	f.loadErr = boom
	rt := &runtimeLoader{}
	first := rt.load(f, testLibDir, nil)
	second := rt.load(f, testLibDir, nil)
	if !errors.Is(first, boom) || !errors.Is(second, boom) {
		t.Fatalf("first = %v, second = %v; want the load error both times", first, second)
	}
	if n := f.count("load"); n != 1 {
		t.Fatalf("api.load ran %d times, want 1: a failed load must not be retried", n)
	}
	if n := f.count("version"); n != 0 {
		t.Fatal("version was read from libraries that failed to load")
	}
}

// TestRuntimeLoader_RefusesOtherVersion: yzma's structs mirror one llama.cpp
// ABI, so libraries reporting another version are refused before a model is
// loaded through them.
func TestRuntimeLoader_RefusesOtherVersion(t *testing.T) {
	f := newFakeAPI()
	f.reported = "0.7.0"
	err := (&runtimeLoader{}).load(f, testLibDir, nil)
	if err == nil {
		t.Fatal("libraries reporting another version were accepted")
	}
	for _, want := range []string{`"0.7.0"`, libVersion, LibTag} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name %s", err, want)
		}
	}
}

// TestRuntimeLoader_SetsGGMLNoBacktrace: on an assertion failure ggml forks a
// debugger unless GGML_NO_BACKTRACE is set, so the loader sets it before the
// libraries load, while an operator's own value wins.
func TestRuntimeLoader_SetsGGMLNoBacktrace(t *testing.T) {
	cases := []struct{ name, before, want string }{
		{"empty", "", "1"},
		{"operator value", "0", "0"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv("GGML_NO_BACKTRACE", c.before)
			f := newFakeAPI()
			if err := (&runtimeLoader{}).load(f, testLibDir, nil); err != nil {
				t.Fatal(err)
			}
			if f.backtraceAtLoad != c.want {
				t.Errorf("GGML_NO_BACKTRACE = %q when the libraries loaded, want %q", f.backtraceAtLoad, c.want)
			}
			if got := os.Getenv("GGML_NO_BACKTRACE"); got != c.want {
				t.Errorf("GGML_NO_BACKTRACE = %q after load, want %q", got, c.want)
			}
		})
	}
}

// TestOpenLlama_StopsAtAFailedLoad: nothing else may be called on libraries
// that did not load (the bindings' functions are unbound), and a model that
// failed to load is reported with its path and never freed.
func TestOpenLlama_StopsAtAFailedLoad(t *testing.T) {
	boom := errors.New("boom")
	t.Run("libraries", func(t *testing.T) {
		f := newFakeAPI()
		f.loadErr = boom
		if _, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, 0, nil); !errors.Is(err, boom) {
			t.Fatalf("err = %v, want the load error", err)
		}
		if got := strings.Join(f.names(), ","); got != "load" {
			t.Fatalf("calls after a failed library load: %s, want only load", got)
		}
	})
	t.Run("model", func(t *testing.T) {
		f := newFakeAPI()
		f.loadModelErr = boom
		_, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, 0, nil)
		if !errors.Is(err, boom) || !strings.Contains(err.Error(), testModelPath) {
			t.Fatalf("err = %v, want the load error naming %s", err, testModelPath)
		}
		if f.count("freeModel") != 0 || f.count("newContext") != 0 {
			t.Fatalf("calls after a failed model load: %v", f.names())
		}
	})
}

// TestOpenLlama_RejectsModelWithoutTrainingContext: n_ctx_train sizes the
// context and the token limit, so without it nothing safe can be built, and
// the model already loaded must not leak.
func TestOpenLlama_RejectsModelWithoutTrainingContext(t *testing.T) {
	f := newFakeAPI()
	f.trainCtx = 0
	if _, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, 0, nil); err == nil {
		t.Fatal("a model without a training context length was accepted")
	}
	if args := f.only(t, "freeModel"); args[0] != fakeModel {
		t.Fatalf("freeModel(%v), want the loaded model %v", args[0], fakeModel)
	}
	if n := f.count("newContext"); n != 0 {
		t.Fatalf("newContext was called %d times for a model without a training context", n)
	}
}

func TestOpenLlama_FreesModelWhenContextFails(t *testing.T) {
	boom := errors.New("cannot allocate the compute buffer")
	f := newFakeAPI()
	f.newContextErr = boom
	_, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, 0, nil)
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want it to wrap the context error", err)
	}
	if args := f.only(t, "freeModel"); args[0] != fakeModel {
		t.Fatalf("freeModel(%v), want the loaded model %v", args[0], fakeModel)
	}
	if n := f.count("freeContext"); n != 0 {
		t.Fatalf("freeContext was called %d times although no context was created", n)
	}
}

// TestOpenLlama_LimitIsMinOfGrantedAndTrain: llama.cpp may grant smaller
// sizes than it was asked for, and nothing longer than the training window
// is ever embedded even when the context could hold it.
func TestOpenLlama_LimitIsMinOfGrantedAndTrain(t *testing.T) {
	cases := []struct{ granted, want int }{
		{granted: 256, want: 256},
		{granted: 1024, want: 512},
	}
	for _, c := range cases {
		f := newFakeAPI()
		f.trainCtx = 512
		f.grantedLimit = c.granted
		b, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, 0, nil)
		if err != nil {
			t.Fatal(err)
		}
		if b.limit() != c.want {
			t.Errorf("granted %d of a 512-token window: limit() = %d, want %d", c.granted, b.limit(), c.want)
		}
		if args := f.only(t, "newContext"); args[1] != int32(512) {
			t.Errorf("newContext asked for nCtx %v, want the training window 512", args[1])
		}
	}
}

// TestOpenLlama_RejectsZeroLimit: truncateTokens reads a limit <= 0 as "no
// limit", which would disable the only guard against llama.cpp's abort on an
// oversized batch, so such a context is freed (before its model) instead.
func TestOpenLlama_RejectsZeroLimit(t *testing.T) {
	f := newFakeAPI()
	f.grantedLimit = 0
	if _, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, 0, nil); err == nil {
		t.Fatal("a context granting a 0-token batch was accepted")
	}
	if args := f.only(t, "freeContext"); args[0] != fakeCtx {
		t.Fatalf("freeContext(%v), want the created context %v", args[0], fakeCtx)
	}
	if args := f.only(t, "freeModel"); args[0] != fakeModel {
		t.Fatalf("freeModel(%v), want the loaded model %v", args[0], fakeModel)
	}
	names := strings.Join(f.names(), ",")
	if !strings.HasSuffix(names, "freeContext,freeModel") {
		t.Fatalf("calls %s: the context must be freed before its model", names)
	}
}

// TestLlamaBackend_EmbedRefusesOversizedAndEmptyBatch: llama.cpp aborts the
// process on a batch larger than n_ubatch, so embed refuses one (and an empty
// one) before anything reaches llama.cpp.
func TestLlamaBackend_EmbedRefusesOversizedAndEmptyBatch(t *testing.T) {
	f := newFakeAPI()
	f.trainCtx, f.grantedLimit = 4, 4
	b := openFake(t, f)
	for _, toks := range [][]int32{{1, 2, 3, 4, 5}, {}} {
		if _, err := b.embed(toks); err == nil {
			t.Errorf("embed accepted %d tokens with a limit of 4", len(toks))
		}
	}
	if len(f.calls) != 0 {
		t.Fatalf("a refused batch reached llama.cpp: %v", f.names())
	}
	if _, err := b.embed([]int32{1, 2, 3, 4}); err != nil {
		t.Fatalf("a batch of exactly the limit must pass: %v", err)
	}
}

// TestLlamaBackend_EmbedSkipsMemoryClearWithoutMemory: BERT encoders have no
// KV cache (GetMemory returns 0, and clearing that handle is an error); a
// model that has one gets it cleared before each text is decoded.
func TestLlamaBackend_EmbedSkipsMemoryClearWithoutMemory(t *testing.T) {
	f := newFakeAPI()
	b := openFake(t, f)
	if _, err := b.embed([]int32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if n := f.count("memoryClear"); n != 0 {
		t.Fatalf("memoryClear was called %d times without a KV cache", n)
	}

	f = newFakeAPI()
	f.mem = 7
	b = openFake(t, f)
	if _, err := b.embed([]int32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "memoryClear"); args[0] != uintptr(7) {
		t.Fatalf("memoryClear(%v), want the context's memory 7", args[0])
	}
	if names := strings.Join(f.names(), ","); !strings.HasPrefix(names, "memoryClear,decode,") {
		t.Fatalf("calls %s: the KV cache must be cleared right before decode", names)
	}
}

// TestLlamaBackend_EmbedCopiesTheView: the vector llama.cpp returns is a view
// into its own buffer, which the next decode overwrites, so embed hands out
// copies the caller owns.
func TestLlamaBackend_EmbedCopiesTheView(t *testing.T) {
	f := newFakeAPI()
	b := openFake(t, f)
	first, err := b.embed([]int32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	second, err := b.embed([]int32{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if &first[0] == &f.buf[0] || &second[0] == &f.buf[0] {
		t.Fatal("embed returned llama.cpp's buffer instead of a copy")
	}
	first[0] = 99
	if second[0] != 1 || f.buf[0] != 1 {
		t.Fatalf("mutating the first vector changed the second (%v) or the buffer (%v)", second, f.buf)
	}
	for i := range f.buf {
		if second[i] != f.buf[i] {
			t.Fatalf("copy = %v, want the full vector %v", second, f.buf)
		}
	}
}

// TestLlamaBackend_EmbedReportsDecodeFailure: each way llama.cpp can fail a
// text surfaces as an error, never as a missing or stale vector; after a
// failed decode the previous text's embeddings are not read.
func TestLlamaBackend_EmbedReportsDecodeFailure(t *testing.T) {
	boom := errors.New("boom")
	cases := []struct {
		name  string
		set   func(*fakeAPI)
		check func(error) bool
		reads int // embeddings calls expected
	}{
		{"decode error", func(f *fakeAPI) { f.decodeErr = boom }, func(err error) bool { return errors.Is(err, boom) }, 0},
		{"decode status", func(f *fakeAPI) { f.decodeRet = 1 }, func(err error) bool { return err != nil && strings.Contains(err.Error(), "decode returned 1") }, 0},
		{"embeddings error", func(f *fakeAPI) { f.embeddingsErr = boom }, func(err error) bool { return errors.Is(err, boom) }, 1},
		{"no vector", func(f *fakeAPI) { f.nilEmbeddings = true }, func(err error) bool { return err != nil }, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newFakeAPI()
			b := openFake(t, f)
			c.set(f)
			vec, err := b.embed([]int32{1, 2})
			if !c.check(err) || vec != nil {
				t.Fatalf("embed = %v, %v", vec, err)
			}
			if n := f.count("embeddings"); n != c.reads {
				t.Fatalf("embeddings read %d times, want %d", n, c.reads)
			}
		})
	}
}

// TestNew_LoadsThroughTheAPI runs the whole New path (pinned downloads,
// runtime load, model open, dimension probe) without network or native code:
// the downloads come from httptest servers and llama.cpp is the fake. First
// use must not be silent: each download is logged before it starts, the
// install and the model once verified, then the load.
func TestNew_LoadsThroughTheAPI(t *testing.T) {
	_, libAsset := serve(t, tarGz(t, "llama-b11429", releaseLibs()))
	_, modelAsset := serve(t, []byte("GGUF fake model"))
	plat := platform()
	origLib, hadLib := libAssets[plat]
	origModel := defaultModelAsset
	origAPI, origLibs := api, libs
	f := newFakeAPI()
	f.buf = make([]float32, 768) // the default model's size, which its spec checks
	for i := range f.buf {
		f.buf[i] = 1
	}
	offline(t)
	libAssets[plat] = libAsset
	defaultModelAsset = modelAsset
	api, libs = f, &runtimeLoader{}
	t.Cleanup(func() {
		if hadLib {
			libAssets[plat] = origLib
		} else {
			delete(libAssets, plat)
		}
		defaultModelAsset = origModel
		api, libs = origAPI, origLibs
	})

	tmp := t.TempDir()
	modelsDir := filepath.Join(tmp, "models")
	modelPath := filepath.Join(modelsDir, defaultModelFile)
	libDir := filepath.Join(tmp, "llama-cpp", LibTag)
	var logs bytes.Buffer
	e, err := New(context.Background(), Config{ModelsDir: modelsDir, LibDir: libDir, ManagedLibDir: true, Threads: 2, Logger: slog.New(slog.NewTextHandler(&logs, nil))})
	if err != nil {
		t.Fatal(err)
	}
	if e.Dimensions() != len(f.buf) {
		t.Errorf("Dimensions() = %d, want the probed width %d", e.Dimensions(), len(f.buf))
	}
	if args := f.only(t, "load"); args[0] != libDir || !filepath.IsAbs(libDir) {
		t.Errorf("load(%v), want the absolute LibDir %s", args[0], libDir)
	}
	if args := f.only(t, "loadModel"); args[0] != modelPath {
		t.Errorf("loadModel(%v), want the model downloaded into %s", args[0], modelsDir)
	}
	if args := f.only(t, "newContext"); args[2] != 2 {
		t.Errorf("newContext ran with %v threads, want the configured 2", args[2])
	}
	checkLog(t, logs.String(), []logLine{
		{"downloading llama.cpp " + LibTag, []string{"url=" + libAsset.URL, fmt.Sprintf("size=%d", libAsset.Size), "sha256=" + libAsset.SHA256, "dest="}},
		{"installed llama.cpp " + LibTag, []string{"dir=" + libDir}},
		{"downloading model " + DefaultModel, []string{"url=" + modelAsset.URL, fmt.Sprintf("size=%d", modelAsset.Size), "sha256=" + modelAsset.SHA256, "dest=" + modelPath}},
		{"downloaded model", []string{"path=" + modelPath}},
		{"llama.cpp loaded", []string{"dir=" + libDir, "version=" + libVersion, "threads=2", "model=" + DefaultModel}},
	})
	// the load line reuses the version the loader read: no second FFI call
	f.only(t, "version")

	vecs, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"first text", "second text"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 {
		t.Fatalf("got %d vectors for two texts", len(vecs))
	}
	for i, v := range vecs {
		if l := vecLen(v); math.Abs(l-1) > 1e-6 {
			t.Errorf("vector %d has length %f, want unit length", i, l)
		}
	}
}

// logLine is one expected log line: its message and the key=value
// attributes it must carry.
type logLine struct {
	msg   string
	attrs []string
}

// checkLog checks that logs (slog text output) holds exactly the want lines,
// in order, each tagged with this package's component.
func checkLog(t *testing.T, logs string, want []logLine) {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(logs), "\n")
	if len(lines) != len(want) {
		t.Fatalf("logged %d lines, want %d:\n%s", len(lines), len(want), logs)
	}
	for i, w := range want {
		for _, s := range append([]string{fmt.Sprintf("msg=%q", w.msg), "component=semsearch"}, w.attrs...) {
			if !strings.Contains(lines[i], s) {
				t.Errorf("log line %d lacks %s: %s", i, s, lines[i])
			}
		}
	}
}

// TestOpenLlama_CapsContextAtMaxTokens: the context is sized to the training
// window but never above the cap (512 by default), so a model trained on
// 8192 tokens does not get multi-gigabyte batch buffers for inputs the
// callers keep far shorter; a smaller window still wins over the cap.
func TestOpenLlama_CapsContextAtMaxTokens(t *testing.T) {
	cases := []struct {
		name      string
		trainCtx  int32
		maxTokens int
		want      int32
	}{
		{"window below default cap", 512, 0, 512},
		{"window above default cap", 8192, 0, 512},
		{"explicit cap below window", 8192, 2048, 2048},
		{"explicit cap above window", 256, 2048, 256},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newFakeAPI()
			f.trainCtx, f.grantedLimit = c.trainCtx, int(c.trainCtx)
			b, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, c.maxTokens, nil)
			if err != nil {
				t.Fatal(err)
			}
			if args := f.only(t, "newContext"); args[1] != c.want {
				t.Errorf("newContext asked for nCtx %v, want %d", args[1], c.want)
			}
			if b.limit() != int(c.want) {
				t.Errorf("limit() = %d, want %d", b.limit(), c.want)
			}
		})
	}
}

// TestOpenLlama_PassesPoolingAndRefusesUnknown: pooling is a property of the
// model, so the requested type must reach llama.cpp, and an unknown one must
// fail before a model is loaded (an unpooled context yields no vector).
func TestOpenLlama_PassesPoolingAndRefusesUnknown(t *testing.T) {
	for _, p := range []Pooling{PoolingMean, PoolingCLS, PoolingLast} {
		f := newFakeAPI()
		if _, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, p, 0, nil); err != nil {
			t.Fatal(err)
		}
		if args := f.only(t, "newContext"); args[3] != p {
			t.Errorf("newContext ran with pooling %v, want %s", args[3], p)
		}
	}
	f := newFakeAPI()
	_, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, Pooling("average"), 0, nil)
	if err == nil || !strings.Contains(err.Error(), `"average"`) || !strings.Contains(err.Error(), "mean") {
		t.Fatalf("err = %v, want a refusal naming the value and the accepted ones", err)
	}
	if n := f.count("loadModel"); n != 0 {
		t.Fatalf("a model was loaded for an unknown pooling (%d loads)", n)
	}
}

// TestLlamaBackend_CloseFreesContextThenModel: the evaluation command opens
// several models in one process, so each must give its context and model
// back, in that order.
func TestLlamaBackend_CloseFreesContextThenModel(t *testing.T) {
	f := newFakeAPI()
	b := openFake(t, f)
	b.close()
	if args := f.only(t, "freeContext"); args[0] != fakeCtx {
		t.Fatalf("freeContext(%v), want %v", args[0], fakeCtx)
	}
	if args := f.only(t, "freeModel"); args[0] != fakeModel {
		t.Fatalf("freeModel(%v), want %v", args[0], fakeModel)
	}
	if names := strings.Join(f.names(), ","); names != "freeContext,freeModel" {
		t.Fatalf("calls %s: want the context freed before its model and nothing else", names)
	}
}

// fakeRuntime serves the pinned libraries from an httptest server, routes
// llama.cpp calls to f, and returns a managed library directory under tmp;
// the globals are restored when the test ends.
func fakeRuntime(t *testing.T, f *fakeAPI, tmp string) Runtime {
	t.Helper()
	_, libAsset := serve(t, tarGz(t, "llama-b11429", releaseLibs()))
	plat := platform()
	origLib, hadLib := libAssets[plat]
	origAPI, origLibs := api, libs
	libAssets[plat] = libAsset
	api, libs = f, &runtimeLoader{}
	t.Cleanup(func() {
		if hadLib {
			libAssets[plat] = origLib
		} else {
			delete(libAssets, plat)
		}
		api, libs = origAPI, origLibs
	})
	return Runtime{LibDir: filepath.Join(tmp, "llama-cpp", LibTag), ManagedLibDir: true, Threads: 2}
}

// TestOpen_PassesPoolingPrefixesAndCap: a custom spec reaches llama.cpp as
// given — its pooling, its cap on the context, and its prefixes per role —
// and a local file needs no URL.
func TestOpen_PassesPoolingPrefixesAndCap(t *testing.T) {
	tmp := t.TempDir()
	f := newFakeAPI()
	rt := fakeRuntime(t, f, tmp)
	modelPath := filepath.Join(tmp, "custom.gguf")
	if err := os.WriteFile(modelPath, []byte("GGUF fake model"), 0o600); err != nil {
		t.Fatal(err)
	}
	spec := ModelSpec{Name: "org/custom", File: modelPath, Pooling: PoolingCLS, QueryPrefix: "q: ", DocumentPrefix: "d: ", MaxTokens: 128}
	e, err := Open(context.Background(), spec, rt)
	if err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "loadModel"); args[0] != modelPath {
		t.Errorf("loadModel(%v), want %s", args[0], modelPath)
	}
	if args := f.only(t, "newContext"); args[1] != int32(128) || args[3] != PoolingCLS {
		t.Errorf("newContext(nCtx=%v, pooling=%v), want 128 and cls", args[1], args[3])
	}
	f.calls = nil
	if _, err := e.Embed(context.Background(), semsearch.RoleQuery, []string{"x"}); err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "tokenize"); args[1] != "q: x" {
		t.Errorf("tokenize(%v), want the query prefix applied", args[1])
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if f.count("freeContext") != 1 || f.count("freeModel") != 1 {
		t.Errorf("Close freed context %d times and model %d times, want once each", f.count("freeContext"), f.count("freeModel"))
	}
}

// TestOpen_UsesExistingFileWithoutDownload: a model already in place is
// opened as is, even when the spec carries a URL, so a second run of the
// evaluation command never re-downloads.
func TestOpen_UsesExistingFileWithoutDownload(t *testing.T) {
	tmp := t.TempDir()
	f := newFakeAPI()
	rt := fakeRuntime(t, f, tmp)
	requests := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		_, _ = w.Write([]byte("GGUF fake model"))
	}))
	t.Cleanup(srv.Close)
	modelPath := filepath.Join(tmp, "models", "m.gguf")
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modelPath, []byte("GGUF fake model"), 0o600); err != nil {
		t.Fatal(err)
	}
	spec := ModelSpec{Name: "org/m", File: modelPath, URL: srv.URL + "/m.gguf", SHA256: strings.Repeat("0", 64)}
	if _, err := Open(context.Background(), spec, rt); err != nil {
		t.Fatal(err)
	}
	if requests != 0 {
		t.Fatalf("the model was requested %d times although the file exists", requests)
	}
}

// TestOpen_DownloadsMissingModelVerified: a missing file is fetched from the
// spec's URL and verified; the wrong digest never produces a usable file.
func TestOpen_DownloadsMissingModelVerified(t *testing.T) {
	tmp := t.TempDir()
	f := newFakeAPI()
	rt := fakeRuntime(t, f, tmp)
	_, a := serve(t, []byte("GGUF fake model"))
	modelPath := filepath.Join(tmp, "models", "m.gguf")
	spec := ModelSpec{Name: "org/m", File: modelPath, URL: a.URL, SHA256: a.SHA256, Size: a.Size}
	if _, err := Open(context.Background(), spec, rt); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Fatalf("model not downloaded into place: %v", err)
	}
	bad := spec
	bad.File = filepath.Join(tmp, "models", "bad.gguf")
	bad.SHA256 = strings.Repeat("1", 64)
	if _, err := Open(context.Background(), bad, rt); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("err = %v, want a digest error", err)
	}
	if _, err := os.Stat(bad.File); err == nil {
		t.Fatal("a file with the wrong digest appeared under its final name")
	}
}

func TestOpen_MissingFileWithoutURLNamesTheFile(t *testing.T) {
	tmp := t.TempDir()
	rt := fakeRuntime(t, newFakeAPI(), tmp)
	missing := filepath.Join(tmp, "nowhere.gguf")
	_, err := Open(context.Background(), ModelSpec{Name: "org/m", File: missing}, rt)
	if err == nil || !strings.Contains(err.Error(), missing) {
		t.Fatalf("err = %v, want it to name %s", err, missing)
	}
}

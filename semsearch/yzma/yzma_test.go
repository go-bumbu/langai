package yzma

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/go-bumbu/langai/semsearch"
)

// stubBackend is a backend that answers with canned vectors, so the tests
// exercise this package's own logic (clamping, truncation, normalization,
// ordering, error wrapping, serialization) without the native libraries.
// Its tokens are the text's runes between a CLS and a SEP marker, so embed
// can map a token sequence back to the text that produced it.
type stubBackend struct {
	mu sync.Mutex
	// lim is the token limit reported to the Embedder (0 = effectively none).
	lim int
	// vectors maps a text to the raw (un-normalized) vector to return for it;
	// an absent text gets defaultVec.
	vectors    map[string][]float32
	defaultVec []float32
	// err, when set, is returned for errText instead of a vector.
	err     error
	errText string
	// nilVectorFor names a text the backend answers with a nil vector.
	nilVectorFor string

	texts  []string  // texts seen by embed (reconstructed from tokens), in order
	tokens [][]int32 // token sequences seen by embed, in order
	// concurrent records whether two embed calls ever overlapped.
	inFlight   int
	concurrent bool
	closes     int // how often close was called
}

// The markers are negative because a rune decoded from a string is never
// negative, so embed's reconstruction of a text from its tokens can never
// drop a real character.
const (
	stubCLS int32 = -1
	stubSEP int32 = -2
)

func (s *stubBackend) limit() int {
	if s.lim == 0 {
		return 1 << 20
	}
	return s.lim
}

func (s *stubBackend) tokenize(text string) []int32 {
	toks := []int32{stubCLS}
	for _, r := range text {
		toks = append(toks, int32(r))
	}
	return append(toks, stubSEP)
}

func (s *stubBackend) embed(tokens []int32) ([]float32, error) {
	s.mu.Lock()
	s.inFlight++
	if s.inFlight > 1 {
		s.concurrent = true
	}
	var b strings.Builder
	for _, t := range tokens {
		if t != stubCLS && t != stubSEP {
			b.WriteRune(rune(t))
		}
	}
	text := b.String()
	s.texts = append(s.texts, text)
	s.tokens = append(s.tokens, append([]int32(nil), tokens...))
	s.mu.Unlock()
	// widen the window in which an unserialized second call would be observed
	time.Sleep(time.Millisecond)
	defer func() {
		s.mu.Lock()
		s.inFlight--
		s.mu.Unlock()
	}()

	if s.err != nil && text == s.errText {
		return nil, s.err
	}
	if text == s.nilVectorFor {
		return nil, nil
	}
	data := s.defaultVec
	if v, ok := s.vectors[text]; ok {
		data = v
	}
	return append([]float32(nil), data...), nil
}

func (s *stubBackend) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closes++
}

// newStubbed returns an Embedder over b with dims already probed, the state
// New leaves behind after a successful load.
func newStubbed(t *testing.T, b backend, dims int) *Embedder {
	t.Helper()
	return &Embedder{b: b, dims: dims}
}

func TestNewEmbedder_ProbesDimensions(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{1, 2, 3, 4}}
	e, err := newEmbedder(stub, "stub", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if e.Dimensions() != 4 {
		t.Fatalf("Dimensions() = %d, want 4 from the probe", e.Dimensions())
	}
	if len(stub.texts) != 1 || stub.texts[0] != "dimension probe" {
		t.Fatalf("probe ran with %v, want exactly one %q call", stub.texts, "dimension probe")
	}
}

func TestNewEmbedder_ProbeFailureIsReported(t *testing.T) {
	stub := &stubBackend{err: errors.New("boom"), errText: "dimension probe"}
	_, err := newEmbedder(stub, "stub", "", "")
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("probe error should be reported and wrap the cause, got %v", err)
	}
}

func TestNewEmbedder_EmptyVectorIsRejected(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{}}
	if _, err := newEmbedder(stub, "stub", "", ""); err == nil {
		t.Fatal("a backend returning an empty vector must not yield a usable embedder")
	}
}

func TestEmbed_NormalizesAndKeepsOrder(t *testing.T) {
	stub := &stubBackend{vectors: map[string][]float32{
		"first":  {3, 4, 0},  // length 5
		"second": {0, 0, -2}, // length 2
	}}
	e := newStubbed(t, stub, 3)
	got, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"first", "second"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d vectors, want 2", len(got))
	}
	for i, v := range got {
		if l := vecLen(v); math.Abs(l-1) > 1e-6 {
			t.Errorf("vector %d has length %f, want unit length", i, l)
		}
	}
	// direction must be preserved, not just magnitude
	if math.Abs(float64(got[0][0])-0.6) > 1e-6 || math.Abs(float64(got[0][1])-0.8) > 1e-6 {
		t.Errorf("first vector = %v, want [0.6 0.8 0]", got[0])
	}
	if got[1][2] >= 0 {
		t.Errorf("second vector lost its sign: %v", got[1])
	}
	if len(stub.texts) != 2 || stub.texts[0] != "first" || stub.texts[1] != "second" {
		t.Errorf("texts were not encoded in order: %v", stub.texts)
	}
}

func TestEmbed_ModelErrorNamesTheText(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{1, 0}, err: errors.New("boom"), errText: "bad"}
	e := newStubbed(t, stub, 2)
	_, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"ok", "bad"})
	if err == nil || !strings.Contains(err.Error(), "embed text 1") || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("error should name the failing index and wrap the cause, got %v", err)
	}
}

func TestEmbed_MissingVectorIsAnError(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{1, 0}, nilVectorFor: "empty"}
	e := newStubbed(t, stub, 2)
	if _, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"empty"}); err == nil {
		t.Fatal("a nil vector must be an error, not a nil embedding")
	}
}

// TestEmbed_RejectsWrongDimensions guards the store contract: a vector of the
// wrong width would be rejected (or worse, silently mis-indexed) downstream,
// so it must fail here with a message naming both widths.
func TestEmbed_RejectsWrongDimensions(t *testing.T) {
	stub := &stubBackend{vectors: map[string][]float32{"short": {1, 0}}, defaultVec: []float32{1, 0, 0}}
	e := newStubbed(t, stub, 3)
	_, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"fine", "short"})
	if err == nil || !strings.Contains(err.Error(), "2 dimensions") || !strings.Contains(err.Error(), "expected 3") {
		t.Fatalf("error should name both widths, got %v", err)
	}
}

// TestEmbed_StopsOnCancelledContext keeps a cancelled sweep from spending
// minutes of CPU on the remaining texts.
func TestEmbed_StopsOnCancelledContext(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{1, 0}}
	e := newStubbed(t, stub, 2)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := e.Embed(ctx, semsearch.RoleDocument, []string{"a", "b"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(stub.texts) != 0 {
		t.Fatalf("backend was called %d times after cancellation", len(stub.texts))
	}
}

// TestEmbed_SerializesBackendAccess covers the mutex in encode: a llama.cpp
// context is not safe for concurrent use, so overlapping Embed calls must
// still reach the backend one at a time.
func TestEmbed_SerializesBackendAccess(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{1, 0}}
	e := newStubbed(t, stub, 2)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"a", "b", "c"}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	stub.mu.Lock()
	defer stub.mu.Unlock()
	if stub.concurrent {
		t.Fatal("the backend was entered concurrently; encode must serialize access")
	}
	if len(stub.texts) != 24 {
		t.Fatalf("backend saw %d calls, want 24", len(stub.texts))
	}
}

// TestEncode_TruncatesToLimitKeepingLastToken pins the invariant that keeps
// the process alive: llama.cpp aborts on a batch larger than its limit
// instead of returning an error, so the Embedder must never hand it one. The
// final token ([SEP]) survives the cut so the model still sees a well-formed
// sequence.
func TestEncode_TruncatesToLimitKeepingLastToken(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{1, 0}, lim: 5}
	e := newStubbed(t, stub, 2)
	if _, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"abcdefghij"}); err != nil {
		t.Fatal(err)
	}
	got := stub.tokens[0]
	want := []int32{stubCLS, 'a', 'b', 'c', stubSEP}
	if len(got) != len(want) {
		t.Fatalf("backend saw %d tokens, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("tokens = %v, want %v", got, want)
		}
	}
}

// TestEncode_ClampsInputText: the tokenizer side of llama.cpp sizes buffers
// from the byte length and expects UTF-8, so both are enforced before it.
func TestEncode_ClampsInputText(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{1, 0}}
	e := newStubbed(t, stub, 2)
	huge := strings.Repeat("é", maxInputBytes) // 2 bytes per rune: twice the cap
	if _, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"bad \xff byte", huge}); err != nil {
		t.Fatal(err)
	}
	if got := stub.texts[0]; got != "bad � byte" {
		t.Errorf("invalid UTF-8 reached the tokenizer: %q", got)
	}
	if got := stub.texts[1]; len(got) > maxInputBytes || !utf8.ValidString(got) || len(got) < maxInputBytes-utf8.UTFMax {
		t.Errorf("oversized text was not cut to the cap on a rune boundary: len=%d valid=%v", len(got), utf8.ValidString(got))
	}
}

func TestTruncateTokens(t *testing.T) {
	cases := []struct {
		name  string
		in    []int32
		limit int
		want  []int32
	}{
		{"under limit", []int32{1, 2, 3}, 5, []int32{1, 2, 3}},
		{"at limit", []int32{1, 2, 3}, 3, []int32{1, 2, 3}},
		{"over limit keeps last", []int32{1, 2, 3, 4, 5}, 3, []int32{1, 2, 5}},
		{"limit one keeps last", []int32{1, 2, 3}, 1, []int32{3}},
		{"no limit", []int32{1, 2, 3}, 0, []int32{1, 2, 3}},
		{"empty", nil, 3, nil},
	}
	for _, c := range cases {
		got := truncateTokens(c.in, c.limit)
		if len(got) != len(c.want) {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("%s: got %v, want %v", c.name, got, c.want)
				break
			}
		}
	}
}

func TestTruncateTokens_DoesNotAliasInput(t *testing.T) {
	in := []int32{1, 2, 3, 4, 5}
	got := truncateTokens(in, 3)
	got[0] = 99
	if in[0] != 1 {
		t.Fatal("truncateTokens must not write into the caller's slice")
	}
}

func TestClampText(t *testing.T) {
	if got := clampText("plain"); got != "plain" {
		t.Errorf("clampText(plain) = %q", got)
	}
	if got := clampText("a\xffb"); got != "a�b" {
		t.Errorf("invalid byte not replaced: %q", got)
	}
	s := strings.Repeat("x", maxInputBytes-1) + "é" // the 2-byte rune straddles the cap
	got := clampText(s)
	if len(got) != maxInputBytes-1 || !utf8.ValidString(got) {
		t.Errorf("straddling rune not dropped: len=%d valid=%v", len(got), utf8.ValidString(got))
	}
}

// vecLen returns the Euclidean length of v.
func vecLen(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func TestNew_RejectsUnknownModel(t *testing.T) {
	_, err := New(context.Background(), Config{Model: "some/other-model", LibDir: t.TempDir()})
	if err == nil || !strings.Contains(err.Error(), DefaultModel) {
		t.Fatalf("err = %v, want an error naming the only supported model", err)
	}
}

func TestNew_RequiresLibDir(t *testing.T) {
	if _, err := New(context.Background(), Config{}); err == nil {
		t.Fatal("New must refuse an empty LibDir instead of letting the bindings read $YZMA_LIB")
	}
}

// TestNew_OperatorLibDirWithoutLibraries: an operator-supplied LibDir is only
// checked, never written to, so one without the libraries fails with the
// install instructions before anything is downloaded or loaded. ModelsDir and
// Threads stay empty to run New's defaults; the test works in a temporary
// directory so the relative default "models" cannot land in the package.
func TestNew_OperatorLibDirWithoutLibraries(t *testing.T) {
	wd := t.TempDir()
	t.Chdir(wd)
	dir := t.TempDir()
	_, err := New(context.Background(), Config{LibDir: dir})
	if err == nil || !strings.Contains(err.Error(), "yzma install") {
		t.Fatalf("err = %v, want an error telling the operator to run yzma install", err)
	}
	for _, d := range []string{dir, wd} {
		entries, err := os.ReadDir(d)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 0 {
			t.Fatalf("New wrote into %s: %v", d, entries)
		}
	}
}

// TestClose_FreesOnceAndBlocksEmbed: Close releases the backend exactly once
// (a second Close is a no-op) and a later Embed is an error rather than a
// use of freed llama.cpp memory.
func TestClose_FreesOnceAndBlocksEmbed(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{1, 0}}
	e := newStubbed(t, stub, 2)
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if stub.closes != 1 {
		t.Fatalf("backend closed %d times, want 1", stub.closes)
	}
	if _, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"a"}); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("Embed after Close = %v, want an error saying the embedder is closed", err)
	}
	if len(stub.texts) != 0 {
		t.Fatal("a closed embedder reached the backend")
	}
}

func TestPooling_Normalized(t *testing.T) {
	if p, err := Pooling("").normalized(); err != nil || p != PoolingMean {
		t.Fatalf("empty pooling = %q, %v; want mean", p, err)
	}
	if p, err := PoolingCLS.normalized(); err != nil || p != PoolingCLS {
		t.Fatalf("cls = %q, %v", p, err)
	}
	if _, err := Pooling("avg").normalized(); err == nil || !strings.Contains(err.Error(), "cls") || !strings.Contains(err.Error(), "last") {
		t.Fatalf("err = %v, want it to list the accepted values", err)
	}
}

// TestEmbed_PrefixesByRole: asymmetric models are trained with a prefix per
// role; the prefix is applied before the text is clamped and tokenized, and a
// symmetric model (empty prefixes) sees the text untouched.
func TestEmbed_PrefixesByRole(t *testing.T) {
	stub := &stubBackend{defaultVec: []float32{1, 0}}
	e := &Embedder{b: stub, dims: 2, queryPrefix: "query: ", documentPrefix: "passage: "}
	if _, err := e.Embed(context.Background(), semsearch.RoleQuery, []string{"a"}); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Embed(context.Background(), semsearch.RoleDocument, []string{"b"}); err != nil {
		t.Fatal(err)
	}
	if len(stub.texts) != 2 || stub.texts[0] != "query: a" || stub.texts[1] != "passage: b" {
		t.Fatalf("backend saw %q, want [query: a, passage: b]", stub.texts)
	}
	plain := &stubBackend{defaultVec: []float32{1, 0}}
	e = newStubbed(t, plain, 2)
	if _, err := e.Embed(context.Background(), semsearch.RoleQuery, []string{"a"}); err != nil {
		t.Fatal(err)
	}
	if plain.texts[0] != "a" {
		t.Fatalf("symmetric embedder changed the text: %q", plain.texts[0])
	}
}

func TestDefaultSpec_IsThePinnedModel(t *testing.T) {
	spec := DefaultSpec("/data/models")
	if spec.Name != DefaultModel || spec.File != filepath.Join("/data/models", defaultModelFile) {
		t.Fatalf("spec = %+v, want the default model inside /data/models", spec)
	}
	if spec.URL != defaultModelAsset.URL || spec.SHA256 != defaultModelAsset.SHA256 || spec.Size != defaultModelAsset.Size {
		t.Fatalf("spec = %+v, want the pinned asset", spec)
	}
	if spec.Pooling != PoolingMean || spec.QueryPrefix != "task: search result | query: " || spec.DocumentPrefix != "title: none | text: " || spec.MaxTokens != 0 || spec.Dims != 768 {
		t.Fatalf("spec = %+v, want mean pooling, EmbeddingGemma's prompts, default cap, 768 dims", spec)
	}
	if got := DefaultSpec("").File; got != filepath.Join("models", defaultModelFile) {
		t.Fatalf("DefaultSpec(\"\").File = %s, want the models default", got)
	}
}

// TestOpen_RefusesBadSpecsBeforeTouchingAnything: every refusal happens
// before libraries or files are touched, and names the model.
func TestOpen_RefusesBadSpecsBeforeTouchingAnything(t *testing.T) {
	rt := Runtime{LibDir: t.TempDir()}
	cases := []struct {
		name string
		spec ModelSpec
		want string
	}{
		{"no name", ModelSpec{File: "/m.gguf"}, "Name"},
		{"no file", ModelSpec{Name: "m"}, "File"},
		{"url without digest", ModelSpec{Name: "m", File: "/m.gguf", URL: "http://x/m.gguf"}, "SHA256"},
		{"unknown pooling", ModelSpec{Name: "m", File: "/m.gguf", Pooling: "average"}, `"average"`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Open(context.Background(), c.spec, rt)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want it to mention %s", err, c.want)
			}
		})
	}
	_, err := Open(context.Background(), ModelSpec{Name: "m", File: "/m.gguf"}, Runtime{})
	if err == nil || !strings.Contains(err.Error(), "LibDir") {
		t.Fatalf("err = %v, want the LibDir requirement", err)
	}
}

// TestOpen_RefusesAModelOfAnotherSize: a spec that names its vector size is
// what a caller sizes its store from before the model loads, so a model whose
// probe disagrees is refused, naming both sizes, and the context and model it
// loaded go back instead of leaking.
func TestOpen_RefusesAModelOfAnotherSize(t *testing.T) {
	tmp := t.TempDir()
	f := newFakeAPI() // 4-dimensional vectors
	rt := fakeRuntime(t, f, tmp)
	modelPath := filepath.Join(tmp, "m.gguf")
	if err := os.WriteFile(modelPath, []byte("GGUF fake model"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := Open(context.Background(), ModelSpec{Name: "org/m", File: modelPath, Dims: 384}, rt)
	if err == nil || !strings.Contains(err.Error(), "384") || !strings.Contains(err.Error(), "4") {
		t.Fatalf("err = %v, want a refusal naming the spec's 384 and the model's 4 dimensions", err)
	}
	if f.count("freeContext") != 1 || f.count("freeModel") != 1 {
		t.Errorf("a refused model freed context %d times and model %d times, want once each", f.count("freeContext"), f.count("freeModel"))
	}
}

// TestOpen_AcceptsAModelOfTheNamedSize: the size check passes a model that
// matches the spec.
func TestOpen_AcceptsAModelOfTheNamedSize(t *testing.T) {
	tmp := t.TempDir()
	rt := fakeRuntime(t, newFakeAPI(), tmp)
	modelPath := filepath.Join(tmp, "m.gguf")
	if err := os.WriteFile(modelPath, []byte("GGUF fake model"), 0o600); err != nil {
		t.Fatal(err)
	}
	e, err := Open(context.Background(), ModelSpec{Name: "org/m", File: modelPath, Dims: 4}, rt)
	if err != nil {
		t.Fatal(err)
	}
	if e.Dimensions() != 4 {
		t.Fatalf("Dimensions() = %d, want 4", e.Dimensions())
	}
}

// TestOpen_FreesTheBackendWhenTheProbeFails: a model that loads but cannot
// produce a vector is refused, and the context and model it loaded go back
// instead of leaking.
func TestOpen_FreesTheBackendWhenTheProbeFails(t *testing.T) {
	tmp := t.TempDir()
	f := newFakeAPI()
	f.nilEmbeddings = true
	rt := fakeRuntime(t, f, tmp)
	modelPath := filepath.Join(tmp, "m.gguf")
	if err := os.WriteFile(modelPath, []byte("GGUF fake model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(context.Background(), ModelSpec{Name: "org/m", File: modelPath}, rt); err == nil {
		t.Fatal("Open succeeded although the dimension probe returned no vector")
	}
	if f.count("freeContext") != 1 || f.count("freeModel") != 1 {
		t.Errorf("a failed probe freed context %d times and model %d times, want once each", f.count("freeContext"), f.count("freeModel"))
	}
}

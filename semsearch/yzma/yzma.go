// Package yzma is a semsearch.Embedder that runs a GGUF sentence-embedding
// model locally through llama.cpp. The llama.cpp shared libraries are loaded
// at runtime by github.com/hybridgroup/yzma (purego + libffi, no CGO), so
// nothing leaves the process: the first New downloads the pinned model and,
// unless the operator supplies a library directory, the pinned llama.cpp build
// (see fetch.go); later loads are offline.
//
// New runs one of the pinned models (see Spec), DefaultModel unless
// Config.Model selects another; Open runs any GGUF described by a ModelSpec
// (pooling, prefixes, context cap) for the evaluation tooling. An Embedder
// runs the model it was opened with until Close. Running another model means
// opening another Embedder, and two run side by side, e.g. while a migration
// builds the next model's vectors into another store generation.
//
// The pinned llama.cpp build (LibTag) matches the yzma version this module
// requires, whose Go bindings mirror exactly that build's ABI. A module that
// imports this package must not require another yzma version: minimal version
// selection would hand these bindings to libraries they do not match.
package yzma

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/go-bumbu/langai/semsearch"
)

// DefaultModel is the logical identity of the model New runs unless
// Config.Model selects another: google/embeddinggemma-300m, a multilingual
// encoder producing 768-dimensional vectors, trained with a prompt per role.
// It is the string to cache alongside vectors (see autotag.Config.Model); the
// GGUF file behind it is defaultModelAsset in fetch.go.
const DefaultModel = "google/embeddinggemma-300m"

// maxInputBytes caps the text handed to the tokenizer. Callers already keep
// inputs small (a tagged page's essence is ~2 KB), but llama.cpp sizes tokenizer
// buffers from the byte length, so an absurd input is cut here rather than
// discovered there.
const maxInputBytes = 32 << 10

// Pooling is how a model's token outputs are reduced to one vector. It is a
// property of the model: sentence-transformers models are trained for mean
// pooling, BGE models for the [CLS] token, decoder-based embedders for the
// last token. Asking llama.cpp for the wrong one yields a valid-looking but
// meaningless vector, so it is never guessed.
type Pooling string

const (
	PoolingMean Pooling = "mean"
	PoolingCLS  Pooling = "cls"
	PoolingLast Pooling = "last"
)

// poolings lists the accepted values, for error messages.
var poolings = []Pooling{PoolingMean, PoolingCLS, PoolingLast}

// normalized returns p with "" read as mean, or an error naming the accepted
// values.
func (p Pooling) normalized() (Pooling, error) {
	if p == "" {
		return PoolingMean, nil
	}
	for _, known := range poolings {
		if p == known {
			return p, nil
		}
	}
	return "", fmt.Errorf("unknown pooling %q: use one of %v", p, poolings)
}

// defaultMaxTokens caps the context when a ModelSpec sets no MaxTokens.
// Callers keep inputs within it (chunks of ~300 tokens, essences of ~500), so
// a model with a longer window (the default model's is 2048) runs capped here
// rather than sizing its context for inputs that never come.
const defaultMaxTokens = 512

// Config selects the model, where models and libraries live, and how the
// model runs.
type Config struct {
	// ModelsDir is the directory the GGUF model is downloaded to and loaded
	// from. Defaults to "models". Pre-seed it for offline installs.
	ModelsDir string
	// Model selects one of the pinned models by name (see Spec); "" means
	// DefaultModel. The Embedder runs it for its whole life.
	Model string
	// LibDir holds the llama.cpp shared libraries (libllama.so and friends).
	// Required.
	LibDir string
	// ManagedLibDir marks LibDir as owned by this package: when the libraries
	// are missing, the pinned build is downloaded into it. An operator-supplied
	// LibDir (false) is never written to, only checked.
	ManagedLibDir bool
	// Threads is the thread count for model evaluation; <= 0 picks a default
	// of at most four.
	Threads int
	// Logger receives this package's progress at Info (each first-use
	// download before it starts, the installed libraries and the downloaded
	// model, the llama.cpp load) and llama.cpp's own warnings and errors; nil
	// discards all of them.
	Logger *slog.Logger
}

// ModelSpec describes one GGUF embedding model: where its file is (or comes
// from) and how it must be run. Spec returns the pinned ones New selects
// from; the evaluation command builds others from its models file.
type ModelSpec struct {
	// Name is the logical identity cached alongside vectors.
	Name string
	// File is the GGUF path. When it is missing and URL is set, it is
	// downloaded there first.
	File string
	// URL is where File is downloaded from when absent. SHA256 is required
	// with it, and Size, when non-zero, is checked too: Open never downloads
	// unverified bytes (an unpinned file goes through Download first).
	URL    string
	SHA256 string
	Size   int64
	// Pooling reduces token outputs to one vector; "" means PoolingMean.
	Pooling Pooling
	// QueryPrefix and DocumentPrefix are prepended to texts embedded as
	// semsearch.RoleQuery and semsearch.RoleDocument. Symmetric models leave
	// both empty.
	QueryPrefix    string
	DocumentPrefix string
	// MaxTokens caps the context (and so the longest input) below the model's
	// training window; 0 means min(window, 512).
	MaxTokens int
	// Dims is the size of the model's vectors, known before it loads so a
	// caller can size its vector store from the spec. Open refuses a model
	// whose vectors come out another size; 0 skips the check.
	Dims int
}

// Runtime is what Open needs besides the model: the llama.cpp libraries and
// how to run them. The fields mirror Config's.
type Runtime struct {
	LibDir        string
	ManagedLibDir bool
	Threads       int
	Logger        *slog.Logger
}

// DefaultSpec is Spec(DefaultModel, modelsDir): the pinned EmbeddingGemma
// Q8_0 GGUF inside modelsDir ("" means "models"), mean pooling, its query and
// document prompts, the 512-token cap, 768 dimensions.
func DefaultSpec(modelsDir string) ModelSpec {
	spec, _ := Spec(DefaultModel, modelsDir) // DefaultModel is always pinned
	return spec
}

// logComponent tags this package's own log lines the way the server tags its
// subsystems; the llama.cpp lines logCallback forwards carry "llama.cpp".
var logComponent = slog.String("component", "semsearch")

// New prepares the libraries and the model cfg.Model selects (downloading
// whatever is missing, see fetch.go), loads them, and probes the model once
// for its dimensionality. ctx bounds the downloads only; the returned
// Embedder outlives it. It is Open with Spec(cfg.Model, cfg.ModelsDir).
func New(ctx context.Context, cfg Config) (*Embedder, error) {
	spec, err := Spec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		return nil, err
	}
	return Open(ctx, spec, Runtime{LibDir: cfg.LibDir, ManagedLibDir: cfg.ManagedLibDir, Threads: cfg.Threads, Logger: cfg.Logger})
}

// Open prepares the libraries and the model described by spec (downloading
// whatever is missing, see fetch.go), loads them with the spec's pooling and
// context cap, and probes the model once for its dimensionality. ctx bounds
// the downloads only; the returned Embedder outlives it and is released
// with Close.
func Open(ctx context.Context, spec ModelSpec, rt Runtime) (*Embedder, error) {
	// This package logs to logger. rt.Logger itself goes to llama.cpp as
	// given, so a nil one keeps llama.cpp's lines silenced on the C side
	// instead of routing each through a Go callback only to discard it.
	logger := rt.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	if spec.Name == "" {
		return nil, errors.New("yzma: ModelSpec.Name is required")
	}
	if spec.File == "" {
		return nil, fmt.Errorf("yzma: model %q: File is required", spec.Name)
	}
	if spec.URL != "" && spec.SHA256 == "" {
		return nil, fmt.Errorf("yzma: model %q: URL without SHA256; pin the digest (Download reports it)", spec.Name)
	}
	pooling, err := spec.Pooling.normalized()
	if err != nil {
		return nil, fmt.Errorf("yzma: model %q: %w", spec.Name, err)
	}
	if rt.LibDir == "" {
		return nil, errors.New("yzma: LibDir is required")
	}
	libDir, err := filepath.Abs(rt.LibDir)
	if err != nil {
		return nil, err
	}
	if rt.Threads <= 0 {
		rt.Threads = defaultThreads()
	}
	ctx, cancel := context.WithTimeout(ctx, downloadTimeout)
	defer cancel()
	if err := ensureLibs(ctx, libDir, rt.ManagedLibDir, logger); err != nil {
		return nil, err
	}
	if err := ensureModelFile(ctx, spec, logger); err != nil {
		return nil, err
	}
	b, err := openLlama(api, libs, libDir, spec.File, rt.Threads, pooling, spec.MaxTokens, rt.Logger)
	if err != nil {
		return nil, err
	}
	logger.Info("llama.cpp loaded", logComponent, slog.String("dir", libDir), slog.String("version", libs.version), slog.Int("threads", rt.Threads), slog.String("model", spec.Name))
	e, err := newEmbedder(b, spec.Name, spec.QueryPrefix, spec.DocumentPrefix)
	if err != nil {
		// the probe failed after the model and its context were created:
		// give them back instead of leaking them
		b.close()
		return nil, err
	}
	if spec.Dims != 0 && e.Dimensions() != spec.Dims {
		_ = e.Close()
		return nil, fmt.Errorf("yzma: model %q produces %d-dimensional vectors, its spec says %d", spec.Name, e.Dimensions(), spec.Dims)
	}
	return e, nil
}

// backend is the seam between Embedder and llama.cpp: the only real
// implementation is llamaBackend (backend.go), and tests substitute a stub so
// this package's own logic runs without the native libraries.
type backend interface {
	// limit is the most tokens one embed call may carry. Exceeding it is not
	// an error in llama.cpp but a process abort, so Embedder never does.
	limit() int
	// tokenize converts text to model tokens, special tokens included.
	tokenize(text string) []int32
	// embed runs one tokenized text through the model and returns the pooled,
	// un-normalized vector as a fresh slice the caller may modify.
	embed(tokens []int32) ([]float32, error)
	// close frees the model and its context; called at most once.
	close()
}

// Embedder implements semsearch.Embedder over a backend.
type Embedder struct {
	// a llama.cpp context is not safe for concurrent use, so encoding is
	// serialized; embedding is CPU-bound anyway
	mu     sync.Mutex
	b      backend
	dims   int
	closed bool // set by Close and read by encode, both under mu
	// prefixes per role, "" for symmetric models (see ModelSpec)
	queryPrefix    string
	documentPrefix string
}

// verify the interface is actually satisfied
var _ semsearch.Embedder = (*Embedder)(nil)

// newEmbedder wraps a backend and probes it once to learn its vector
// dimensionality. Split from Open so the probe and its failure modes stay
// testable with a stub backend.
func newEmbedder(b backend, name, queryPrefix, documentPrefix string) (*Embedder, error) {
	e := &Embedder{b: b, queryPrefix: queryPrefix, documentPrefix: documentPrefix}
	probe, err := e.encode(context.Background(), "dimension probe")
	if err != nil {
		return nil, fmt.Errorf("probe embedding model %q: %w", name, err)
	}
	if len(probe) == 0 {
		return nil, fmt.Errorf("embedding model %q returned an empty vector", name)
	}
	e.dims = len(probe)
	return e, nil
}

// Dimensions implements semsearch.Embedder.
func (e *Embedder) Dimensions() int { return e.dims }

// Embed implements semsearch.Embedder: each text gets the role's prefix and
// is encoded on its own.
func (e *Embedder) Embed(ctx context.Context, role semsearch.Role, texts []string) ([][]float32, error) {
	prefix := e.documentPrefix
	if role == semsearch.RoleQuery {
		prefix = e.queryPrefix
	}
	out := make([][]float32, len(texts))
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vec, err := e.encode(ctx, prefix+text)
		if err != nil {
			return nil, fmt.Errorf("embed text %d: %w", i, err)
		}
		if e.dims != 0 && len(vec) != e.dims {
			return nil, fmt.Errorf("embed text %d: got %d dimensions, expected %d", i, len(vec), e.dims)
		}
		out[i] = vec
	}
	return out, nil
}

// Close frees the model and its llama.cpp context; a later Embed fails. The
// server never closes its embedder (the model lives as long as the process);
// the evaluation command does, between models. Closing twice is a no-op.
func (e *Embedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	e.b.close()
	return nil
}

// encode runs one text through the model (pooling is the one
// ModelSpec.Pooling selected when the backend was opened, mean when empty)
// and returns a unit-length vector. The text is clamped and the token
// sequence truncated to the backend's limit here, before anything reaches
// llama.cpp.
func (e *Embedder) encode(_ context.Context, text string) ([]float32, error) {
	text = clampText(text)
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("embedder is closed")
	}
	toks := truncateTokens(e.b.tokenize(text), e.b.limit())
	if len(toks) == 0 {
		return nil, errors.New("tokenizer returned no tokens")
	}
	vec, err := e.b.embed(toks)
	if err != nil {
		return nil, err
	}
	if len(vec) == 0 {
		return nil, errors.New("model returned no vector")
	}
	semsearch.Normalize(vec)
	return vec, nil
}

// clampText makes text safe for the tokenizer: invalid UTF-8 is replaced and
// anything past maxInputBytes is dropped on a rune boundary.
func clampText(s string) string {
	s = strings.ToValidUTF8(s, "�")
	if len(s) <= maxInputBytes {
		return s
	}
	cut := maxInputBytes
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut]
}

// truncateTokens keeps the first limit-1 tokens plus the final one, so a
// sequence cut for length still ends with its [SEP]. limit <= 0 means no
// limit. The result never aliases toks when a cut was made.
func truncateTokens(toks []int32, limit int) []int32 {
	if limit <= 0 || len(toks) <= limit {
		return toks
	}
	out := make([]int32, 0, limit)
	out = append(out, toks[:limit-1]...)
	return append(out, toks[len(toks)-1])
}

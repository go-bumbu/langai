// Package yzma runs GGUF models locally through llama.cpp: a sentence-embedding
// model as a langai.Embedder, and an instruction-tuned chat model as a
// langai.Completer, which a tagging.Generator has write tags. The llama.cpp
// shared libraries are loaded at runtime by github.com/hybridgroup/yzma
// (purego + libffi, no CGO), so nothing leaves the process: the first New or
// NewCompleter downloads the pinned model and, unless the operator supplies a
// library directory, the pinned llama.cpp build (see fetch.go); later loads
// are offline. The libraries load once per process, so an Embedder and a
// Completer run side by side.
//
// Each kind of model opens two ways, in one shape. New, NewCompleter and
// NewVision take a Config, which names one of the pinned models (Models,
// CompleterModels, VisionModels; the default of each kind when it names
// none) and embeds the Runtime; Open, OpenCompleter and OpenVision take a
// spec instead (ModelSpec, CompleterSpec, VisionSpec: any GGUF, e.g. for the
// evaluation tooling) and the Runtime. LookupSpec, LookupCompleterSpec and
// LookupVisionSpec return the pinned specs. A Vision also takes its
// VisionOptions, both ways. A model runs until Close. Running another model
// means opening another one, and two run side by side, e.g. while a
// migration builds the next model's vectors into another store generation.
//
// The pinned llama.cpp build (LibTag) matches the yzma version this module
// requires, whose Go bindings mirror exactly that build's ABI. A module that
// imports this package must not require another yzma version: minimal version
// selection would hand these bindings to libraries they do not match.
package yzma

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/internal/fetch"
	"github.com/go-bumbu/langai/internal/hw"
)

// DefaultModel is the logical identity of the model New runs unless
// Config.Model selects another: google/embeddinggemma-300m, a multilingual
// encoder producing 768-dimensional vectors, trained with a prompt per role.
// It is the name the identity to store next to vectors starts with (see
// Embedder.Model and tagging.TaggerConfig.Model); the
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

// Config selects a pinned model by name and the runtime that runs it: what
// New, NewCompleter and NewVision take. Open, OpenCompleter and OpenVision
// take a spec and the Runtime instead.
type Config struct {
	// Runtime is the llama.cpp libraries and how the model runs.
	Runtime
	// ModelsDir is the directory the GGUF model is downloaded to and loaded
	// from. Defaults to "models". Pre-seed it for offline installs.
	ModelsDir string
	// Model selects one of the pinned models by name; "" means the default
	// of the kind being opened. New reads it as an embedding model (see
	// Models and LookupSpec, "" meaning DefaultModel), NewCompleter as a chat
	// model (CompleterModels, DefaultCompleterModel) and NewVision as a
	// vision model (VisionModels, DefaultVisionModel), so a Config that
	// leaves it empty builds all three.
	Model string
}

// ModelSpec describes one GGUF embedding model: where its file is (or comes
// from) and how it must be run. LookupSpec returns the pinned ones New selects
// from; the evaluation command builds others from its models file.
type ModelSpec struct {
	// Name names the model. It is the first part of the identity to store
	// next to vectors, which Embedder.Model completes.
	Name string
	// File is the GGUF path. When it is missing and URL is set, it is
	// downloaded there first.
	File string
	// URL is where File is downloaded from when it is absent or, with a Size
	// set, not that long (a truncated file). SHA256 is required
	// with it, and Size, when non-zero, is checked too: Open never downloads
	// unverified bytes (the evaluation tooling downloads an unpinned file itself
	// and logs its digest).
	URL    string
	SHA256 string
	Size   int64
	// Pooling reduces token outputs to one vector; "" means PoolingMean.
	Pooling Pooling
	// QueryPrefix and DocumentPrefix are prepended to texts embedded as
	// langai.RoleQuery and langai.RoleDocument. Symmetric models leave
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

// validate reports a spec Open cannot run.
func (spec ModelSpec) validate() error {
	switch {
	case spec.Name == "":
		return errors.New("yzma: ModelSpec.Name is required")
	case spec.File == "":
		return fmt.Errorf("yzma: model %q: File is required", spec.Name)
	case spec.URL != "" && spec.SHA256 == "":
		return fmt.Errorf("yzma: model %q: URL without SHA256; pin the digest (make semeval logs it for an unpinned entry)", spec.Name)
	case spec.MaxTokens < 0 || spec.Dims < 0 || spec.Size < 0:
		return fmt.Errorf("yzma: model %q: MaxTokens, Dims and Size must not be negative", spec.Name)
	}
	if _, err := spec.Pooling.normalized(); err != nil {
		return fmt.Errorf("yzma: model %q: %w", spec.Name, err)
	}
	return nil
}

// Runtime is what Open, OpenCompleter and OpenVision need besides the
// model: the llama.cpp libraries and how to run them. The libraries load
// once per process, so every model opened in it shares them.
type Runtime struct {
	// LibDir holds the llama.cpp shared libraries (libllama.so and friends).
	// Required.
	LibDir string
	// ManagedLibDir marks LibDir as owned by this package: when the libraries
	// are missing, the pinned build is downloaded into it. Give it a directory
	// of its own: only a missing or empty one, or one an earlier install left
	// its record in, is ever filled or replaced, and anything else is an
	// error rather than a deletion. Installs are serialized by a lock file
	// next to it ("<LibDir>.lock"), between goroutines and processes, and the
	// libraries a process has loaded are never replaced under it: asking for
	// another build by name in the same process is an error. An
	// operator-supplied LibDir (false) is never written to, only checked.
	ManagedLibDir bool
	// Threads is the thread count for model evaluation; 0 picks a default of
	// at most four. Negative is an error.
	Threads int
	// Accel is what runs the model: "" picks from the hardware (a GPU when
	// one registers, the CPU otherwise), AccelCPU stays on the CPU, and
	// AccelVulkan, AccelCUDA or AccelMetal require that backend. A managed
	// LibDir gets the llama.cpp build the choice needs. Any other value is an
	// error.
	Accel Accel
	// GPULayers caps how many layers go to the GPU when one is used; 0 means
	// all of them. Negative is an error.
	GPULayers int
	// Logger receives this package's progress at Info (each first-use
	// download before it starts, the installed libraries and the downloaded
	// model, the llama.cpp load) and llama.cpp's own warnings and errors; nil
	// discards this package's lines. llama.cpp has one log sink for the whole
	// process, so its lines go to the Logger of the latest open that set one,
	// whichever model they are about: a nil Logger leaves them where an
	// earlier open pointed them.
	Logger *slog.Logger
}

// validate reports a Runtime no open can run with, before anything is
// downloaded or loaded.
func (rt Runtime) validate() error {
	if rt.LibDir == "" {
		return errors.New("yzma: Runtime.LibDir is required")
	}
	if _, err := rt.Accel.normalized(); err != nil {
		return fmt.Errorf("yzma: %w", err)
	}
	if rt.Threads < 0 {
		return fmt.Errorf("yzma: Runtime.Threads %d is negative: use 0 for the default", rt.Threads)
	}
	if rt.GPULayers < 0 {
		return fmt.Errorf("yzma: Runtime.GPULayers %d is negative: use 0 for all layers", rt.GPULayers)
	}
	return nil
}

// prepare validates rt, fills its defaults, makes sure libDir holds the
// build rt.Accel needs (downloading it into a managed directory) and loads
// the libraries from it through api, once per process (see runtimeLoader).
// Install and load are one step under the directory's lock (see
// ensureLibs): nothing replaces the libraries between the check and the
// load, and loaded libraries are never replaced. It returns the absolute
// library directory and the logger this package logs to. ctx bounds the
// download.
func (rt *Runtime) prepare(ctx context.Context, api runtimeAPI, loader *runtimeLoader) (libDir string, logger *slog.Logger, err error) {
	logger = rt.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	if err := rt.validate(); err != nil {
		return "", nil, err
	}
	libDir, err = filepath.Abs(rt.LibDir)
	if err != nil {
		return "", nil, err
	}
	if rt.Threads == 0 {
		rt.Threads = defaultThreads()
	}
	if from := loader.loadedDir(); rt.ManagedLibDir && from != "" && from != libDir {
		// before anything is installed into a directory that cannot be loaded
		return "", nil, fmt.Errorf("yzma: llama.cpp is already loaded from %s; cannot load %s in the same process", from, libDir)
	}
	report := detectHW()
	b, err := chooseBackend(rt.Accel, report, platform())
	if err != nil {
		if rt.ManagedLibDir {
			return "", nil, fmt.Errorf("yzma: %w", err)
		}
		b = "cpu" // nothing to install; whatever the operator put there registers
	}
	// the load line says what runs the model; the findings behind the choice
	// are for whoever asks
	logger.Debug("hardware", logComponent, slog.String("found", report.String()), slog.String("accel", string(rt.Accel)), slog.String("build", b))
	err = ensureLibs(ctx, libsWant{
		plat: platform(), dir: libDir, managed: rt.ManagedLibDir, build: b, explicit: rt.Accel != AccelAuto,
		loadedDir: loader.loadedDir,
		use:       func() error { return loader.load(api, libDir, rt.Logger) },
	}, logger)
	if err != nil {
		return "", nil, err
	}
	if rt.ManagedLibDir && b != "cpu" && rt.Accel != AccelCPU && !hasGPU(loader.devices) {
		// the build was chosen for a GPU the hardware seemed to have; say so
		// when it did not materialize, or the model quietly runs on the CPU
		logger.Warn("the "+b+" build of llama.cpp registered no GPU device: models run on the CPU", logComponent,
			slog.String("dir", libDir), slog.String("devices", deviceList(loader.devices)), slog.String("found", report.String()))
	}
	return libDir, logger, nil
}

// hasGPU reports whether a hardware GPU is among the devices the loaded
// libraries registered.
func hasGPU(devices []device) bool {
	for _, d := range devices {
		if d.gpu && !d.software() {
			return true
		}
	}
	return false
}

// DefaultSpec is LookupSpec(DefaultModel, modelsDir): the pinned EmbeddingGemma
// Q8_0 GGUF inside modelsDir ("" means "models"), mean pooling, its query and
// document prompts, the 512-token cap, 768 dimensions.
func DefaultSpec(modelsDir string) ModelSpec {
	spec, _ := LookupSpec(DefaultModel, modelsDir) // DefaultModel is always pinned
	return spec
}

// logComponent tags this package's own log lines the way the server tags its
// subsystems; the llama.cpp lines logCallback forwards carry "llama.cpp".
var logComponent = slog.String("component", "semsearch")

// detectHW looks at the machine for what the pinned builds can run on: CUDA
// only with the runtime the CUDA build links against (see cudaRuntimeMajor).
// Tests substitute a fixed report.
var detectHW = func() hw.Report { return hw.Detect(hw.Needs{CUDARuntime: cudaRuntimeMajor}) }

// New prepares the libraries and the model cfg.Model selects (downloading
// whatever is missing, see fetch.go), loads them, and probes the model once
// for its dimensionality. ctx bounds the downloads only; the returned
// Embedder outlives it. It is Open with LookupSpec(cfg.Model, cfg.ModelsDir)
// and cfg.Runtime.
func New(ctx context.Context, cfg Config) (*Embedder, error) {
	spec, err := LookupSpec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		return nil, err
	}
	return Open(ctx, spec, cfg.Runtime)
}

// Open prepares the libraries and the model described by spec (downloading
// whatever is missing, see fetch.go), loads them with the spec's pooling and
// context cap, and probes the model once for its dimensionality. ctx bounds
// the downloads only, and gets a deadline of 15 minutes when it has none of
// its own (give it one to allow a slow link more); the returned Embedder
// outlives it and is released with Close. A download that was cut short is
// picked up where it stopped by the next Open.
func Open(ctx context.Context, spec ModelSpec, rt Runtime) (*Embedder, error) {
	if err := spec.validate(); err != nil {
		return nil, err
	}
	if err := rt.validate(); err != nil {
		return nil, err
	}
	pooling, _ := spec.Pooling.normalized() // validate checked it
	ctx, cancel := fetch.WithDefaultTimeout(ctx, downloadTimeout)
	defer cancel()
	// This package logs to logger, which discards when rt.Logger is nil.
	// rt.Logger itself is what llama.cpp's lines are pointed at, and only
	// when it is set (see llamaLog).
	libDir, logger, err := rt.prepare(ctx, api, libs)
	if err != nil {
		return nil, err
	}
	if err := ensureModelFile(ctx, spec, logger); err != nil {
		return nil, err
	}
	b, err := openLlama(api, libs, libDir, spec.File, rt.Threads, pooling, spec.MaxTokens, rt.Accel, rt.GPULayers, rt.Logger)
	if err != nil {
		return nil, err
	}
	logger.Info("llama.cpp loaded", logComponent, slog.String("dir", libDir), slog.String("version", libs.version), slog.Int("threads", rt.Threads), slog.String("accel", b.accel), slog.String("model", spec.Name))
	e, err := newEmbedder(b, spec.Name, spec.QueryPrefix, spec.DocumentPrefix)
	if err != nil {
		// the probe failed after the model and its context were created:
		// give them back instead of leaking them
		b.close()
		return nil, err
	}
	e.accel = b.accel
	e.model = embedderModel(spec.Name, pooling, spec.QueryPrefix, spec.DocumentPrefix)
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
	embed(ctx context.Context, tokens []int32) ([]float32, error)
	// close frees the model and its context; called at most once.
	close()
}

// Embedder implements langai.Embedder over a backend.
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
	accel          string // what runs the model, see Accel
	model          string // the identity of its vectors, see Model
}

// verify the interface is actually satisfied
var _ langai.Embedder = (*Embedder)(nil)

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

// Dimensions implements langai.Embedder.
func (e *Embedder) Dimensions() int { return e.dims }

// Accel says what runs the model, as the load log line does: "cpu" or the
// GPU device.
func (e *Embedder) Accel() string { return e.accel }

// Model identifies the vectors this Embedder produces: the string to store
// next to them (see tagging.TaggerConfig.Model), the same on every run and
// different whenever the vectors would be. It is the spec's name followed
// by what else decides the vector of a text:
//
//	<ModelSpec.Name>;pooling=<mean|cls|last>;prefixes=<id>
//
// where <id> is "none" for a model without prefixes and otherwise the first
// eight hex digits of the SHA-256 of the query prefix, a NUL byte and the
// document prefix. The default model is
// "google/embeddinggemma-300m;pooling=mean;prefixes=<id>". What does not
// move a vector (threads, the accelerator, the library directory) is not
// part of it; neither is the GGUF file, so two quantizations of one model
// need two names.
func (e *Embedder) Model() string { return e.model }

// embedderModel builds the identity Model documents.
func embedderModel(name string, pooling Pooling, queryPrefix, documentPrefix string) string {
	prefixes := "none"
	if queryPrefix != "" || documentPrefix != "" {
		prefixes = shortDigest(queryPrefix, documentPrefix)
	}
	return name + ";pooling=" + string(pooling) + ";prefixes=" + prefixes
}

// shortDigest is the first eight hex digits of the SHA-256 of the parts
// joined by NUL bytes: enough to tell settings apart in an identity string
// without spelling them out.
func shortDigest(parts ...string) string {
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x00")))
	return hex.EncodeToString(sum[:4])
}

// Embed implements langai.Embedder: each text gets the role's prefix and
// is encoded on its own.
func (e *Embedder) Embed(ctx context.Context, role langai.Role, texts []string) ([][]float32, error) {
	prefix := e.documentPrefix
	if role == langai.RoleQuery {
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
func (e *Embedder) encode(ctx context.Context, text string) ([]float32, error) {
	text = clampText(text)
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("yzma: embedder: %w", langai.ErrClosed)
	}
	toks := truncateTokens(e.b.tokenize(text), e.b.limit())
	if len(toks) == 0 {
		return nil, errors.New("tokenizer returned no tokens")
	}
	vec, err := e.b.embed(ctx, toks)
	if err != nil {
		return nil, err
	}
	if len(vec) == 0 {
		return nil, errors.New("model returned no vector")
	}
	langai.Normalize(vec)
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

// Package onnx runs image embedding models locally through ONNX Runtime, as
// a langai.ImageEmbedder. The runtime is the shared library the first New or
// Open loads (through github.com/yalue/onnxruntime_go, which is CGO) from
// Config.LibDir; a managed directory gets the pinned build downloaded into
// it, and the pinned model is downloaded into Config.ModelsDir, both verified
// by digest (see fetch.go). Later loads are offline. The library loads once
// per process and every Embedder shares it.
//
// New runs one of the pinned models (see Spec), DefaultModel unless
// Config.Model selects another; Open runs any ONNX vision model described by
// a ModelSpec: the tensor names, the input size, how a picture is resized and
// normalized. An Embedder runs its model until Close; two run side by side.
//
// CPU is the default. Config.Provider selects CUDA (the gpu build of the
// runtime, CUDA 12 and cuDNN 9 on the host) or CoreML (the macOS build).
//
// The pinned runtime build (LibVersion) is the one whose C API headers the
// onnxruntime_go version in this module's go.mod was written against; the
// two are bumped together (pins_test.go).
package onnx

import (
	"context"
	"errors"
	"fmt"
	"image"
	"log/slog"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/internal/fetch"
)

// DefaultModel is the logical identity of the model New runs unless
// Config.Model selects another: the vision tower of google/siglip2-base-patch16-224,
// a 768-dimensional image encoder trained with a text tower, so its vectors
// can later be matched by text too. It is the name the identity to store next
// to vectors starts with (see Embedder.Model); the ONNX file behind it is in
// models.go.
const DefaultModel = "google/siglip2-base-patch16-224"

// Provider selects what runs the model.
type Provider string

const (
	// ProviderCPU runs on the CPU with the default build; it is the zero
	// value's meaning.
	ProviderCPU Provider = "cpu"
	// ProviderCUDA runs on an NVIDIA GPU: the runtime's gpu build, CUDA 12
	// and cuDNN 9 installed on the host. A managed LibDir downloads that
	// build on linux/amd64.
	ProviderCUDA Provider = "cuda"
	// ProviderCoreML runs through Apple's CoreML, built into the macOS
	// runtime build.
	ProviderCoreML Provider = "coreml"
)

// providers lists the accepted values, for error messages.
var providers = []Provider{ProviderCPU, ProviderCUDA, ProviderCoreML}

// normalized returns p with "" read as CPU, or an error naming the accepted
// values.
func (p Provider) normalized() (Provider, error) {
	if p == "" {
		return ProviderCPU, nil
	}
	for _, known := range providers {
		if p == known {
			return p, nil
		}
	}
	return "", fmt.Errorf("unknown provider %q: use one of %v", p, providers)
}

// Resize is how a picture of any shape becomes the square the model takes.
type Resize string

const (
	// ResizeSquash scales the whole picture to Side×Side, aspect ratio
	// ignored: what SigLIP's processor does.
	ResizeSquash Resize = "squash"
	// ResizeCrop scales the picture so its shortest edge is ShortestEdge
	// long and takes the central Side×Side: what DINOv2's processor does.
	ResizeCrop Resize = "crop"
)

// Interpolation is the filter the resize uses, named as the processors name
// them: bilinear (PIL's BILINEAR) or bicubic (PIL's BICUBIC).
type Interpolation string

const (
	InterpolationBilinear Interpolation = "bilinear"
	InterpolationBicubic  Interpolation = "bicubic"
)

// Pooling is how the model's output becomes one vector per picture. It is
// a property of the export: a model whose Output is already [n, Dims]
// needs none; one that exposes its token sequence, [n, tokens, Dims], is
// pooled here, the way its own code pools it. The wrong one yields a
// valid-looking but meaningless vector, so it is never guessed.
type Pooling string

const (
	// PoolingNone takes the output as the vectors: SigLIP's pooler_output.
	PoolingNone Pooling = "none"
	// PoolingCLS takes each picture's first token: DINOv2's class token in
	// last_hidden_state.
	PoolingCLS Pooling = "cls"
	// PoolingMean averages each picture's tokens.
	PoolingMean Pooling = "mean"
)

// poolings lists the accepted values, for error messages.
var poolings = []Pooling{PoolingNone, PoolingCLS, PoolingMean}

// Default tensor names of the Optimum exports.
const (
	defaultInput  = "pixel_values"
	defaultOutput = "pooler_output"
)

// defaultBatchSize is how many images one model run takes when
// Config.BatchSize is not set.
const defaultBatchSize = 8

// Config selects the model, where models and libraries live, and how the
// model runs.
type Config struct {
	// ModelsDir is the directory the ONNX model is downloaded to and loaded
	// from. Defaults to "models". Pre-seed it for offline installs.
	ModelsDir string
	// Model selects one of the pinned models by name (see Spec); "" means
	// DefaultModel.
	Model string
	// LibDir holds the ONNX Runtime shared library (libonnxruntime.so.<version>
	// and, for CUDA, the provider libraries next to it). Required.
	LibDir string
	// ManagedLibDir marks LibDir as owned by this package: when the library
	// is missing, or was installed for another Provider, the pinned build is
	// downloaded into it. Give it a directory of its own: only a missing or
	// empty one, or one an earlier install left its record in, is ever filled
	// or replaced, and anything else is an error rather than a deletion.
	// Installs are serialized by a lock file next to it ("<LibDir>.lock"),
	// between goroutines and processes, and the library a process has loaded
	// is never replaced under it: CUDA after CPU in the same process is an
	// error (CPU after CUDA runs on the CUDA build). An operator-supplied
	// LibDir (false) is never written to, only checked.
	ManagedLibDir bool
	// Provider is what runs the model; "" means ProviderCPU.
	Provider Provider
	// DeviceID is the GPU the CUDA provider uses; 0 is the first.
	DeviceID int
	// Threads is the thread count for one model run (ONNX Runtime's
	// intra-op threads); <= 0 picks a default of at most four.
	Threads int
	// BatchSize is how many images go through the model in one run; <= 0
	// means 8. Larger batches help a GPU; on a CPU they mostly cost memory.
	BatchSize int
	// Logger receives this package's progress at Info (each first-use
	// download before it starts, the installed library and the downloaded
	// model, the runtime load); nil discards it. ONNX Runtime's own
	// warnings go to stderr.
	Logger *slog.Logger
}

// ModelSpec describes one ONNX vision model: where its file is (or comes
// from), which tensors to feed and read, and how a picture is prepared for
// it, which must match what the model was trained with. Spec returns the
// pinned ones New selects from.
type ModelSpec struct {
	// Name names the model. It is the first part of the identity to store
	// next to vectors, which Embedder.Model completes.
	Name string
	// File is the ONNX path. When it is missing and URL is set, it is
	// downloaded there first.
	File string
	// URL is where File is downloaded from when it is absent or, with a Size
	// set, not that long (a truncated file). SHA256 is required
	// with it, and Size, when non-zero, is checked too: Open never downloads
	// unverified bytes.
	URL    string
	SHA256 string
	Size   int64
	// Input and Output name the tensors: the batch of pictures goes in as
	// Input ([n, 3, Side, Side] float32) and the vectors are read from
	// Output, [n, Dims] or, with a Pooling, [n, tokens, Dims]. "" means
	// pixel_values and pooler_output, the names of the Optimum exports.
	Input  string
	Output string
	// Pooling reduces a token sequence in Output to one vector; "" means
	// PoolingNone, an Output that is the vectors already.
	Pooling Pooling
	// Side is the width and height of the square the model takes.
	Side int
	// Resize is how a picture becomes that square; "" means ResizeSquash.
	Resize Resize
	// ShortestEdge is, for ResizeCrop, the length the shortest edge is
	// scaled to before the central Side×Side is cut; 0 means Side.
	ShortestEdge int
	// Interpolation is the resize filter; "" means InterpolationBilinear.
	Interpolation Interpolation
	// Mean and Std normalize each channel: (value/255 - Mean) / Std, in RGB
	// order. A zero Std entry means 1.
	Mean [3]float32
	Std  [3]float32
	// Dims is the size of the model's vectors, known before it loads so a
	// caller can size its vector store from the spec. Open refuses a model
	// whose vectors come out another size; 0 skips the check and takes the
	// size the model file declares.
	Dims int
}

// normalized returns spec with its defaults filled in, or an error naming
// what is wrong with it.
func (spec ModelSpec) normalized() (ModelSpec, error) {
	if spec.Name == "" {
		return spec, errors.New("onnx: ModelSpec.Name is required")
	}
	if spec.File == "" {
		return spec, fmt.Errorf("onnx: model %q: File is required", spec.Name)
	}
	if spec.URL != "" && spec.SHA256 == "" {
		return spec, fmt.Errorf("onnx: model %q: URL without SHA256; pin the digest", spec.Name)
	}
	if spec.Side <= 0 {
		return spec, fmt.Errorf("onnx: model %q: Side %d must be positive", spec.Name, spec.Side)
	}
	if spec.Input == "" {
		spec.Input = defaultInput
	}
	if spec.Output == "" {
		spec.Output = defaultOutput
	}
	switch spec.Pooling {
	case "":
		spec.Pooling = PoolingNone
	case PoolingNone, PoolingCLS, PoolingMean:
	default:
		return spec, fmt.Errorf("onnx: model %q: unknown pooling %q: use one of %v", spec.Name, spec.Pooling, poolings)
	}
	switch spec.Resize {
	case "":
		spec.Resize = ResizeSquash
	case ResizeSquash, ResizeCrop:
	default:
		return spec, fmt.Errorf("onnx: model %q: unknown resize %q: use one of [%s %s]", spec.Name, spec.Resize, ResizeSquash, ResizeCrop)
	}
	if spec.ShortestEdge == 0 {
		spec.ShortestEdge = spec.Side
	}
	if spec.ShortestEdge < spec.Side {
		return spec, fmt.Errorf("onnx: model %q: ShortestEdge %d is shorter than Side %d", spec.Name, spec.ShortestEdge, spec.Side)
	}
	switch spec.Interpolation {
	case "":
		spec.Interpolation = InterpolationBilinear
	case InterpolationBilinear, InterpolationBicubic:
	default:
		return spec, fmt.Errorf("onnx: model %q: unknown interpolation %q: use one of [%s %s]", spec.Name, spec.Interpolation, InterpolationBilinear, InterpolationBicubic)
	}
	for i := range spec.Std {
		if spec.Std[i] == 0 {
			spec.Std[i] = 1
		}
	}
	if spec.Dims < 0 {
		return spec, fmt.Errorf("onnx: model %q: Dims %d is negative", spec.Name, spec.Dims)
	}
	return spec, nil
}

// Runtime is what Open needs besides the model: the runtime library and how
// to run the model. The fields mirror Config's.
type Runtime struct {
	LibDir        string
	ManagedLibDir bool
	Provider      Provider
	DeviceID      int
	Threads       int
	BatchSize     int
	Logger        *slog.Logger
}

// logComponent tags this package's log lines the way the other providers
// tag theirs.
var logComponent = slog.String("component", "onnx")

// defaultThreads is the thread count when a Config sets none: up to four,
// enough to keep a server responsive while a batch runs.
func defaultThreads() int {
	return min(4, runtime.NumCPU())
}

// DefaultSpec is Spec(DefaultModel, modelsDir): the pinned SigLIP 2 vision
// tower inside modelsDir ("" means "models"), 224 pixels squashed, mean and
// std 0.5, 768 dimensions.
func DefaultSpec(modelsDir string) ModelSpec {
	spec, _ := Spec(DefaultModel, modelsDir) // DefaultModel is always pinned
	return spec
}

// New prepares the runtime and the model cfg.Model selects (downloading
// whatever is missing, see fetch.go), loads them, and runs the model once to
// check its output. ctx bounds the downloads only; the returned Embedder
// outlives it. It is Open with Spec(cfg.Model, cfg.ModelsDir).
func New(ctx context.Context, cfg Config) (*Embedder, error) {
	spec, err := Spec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		return nil, err
	}
	return Open(ctx, spec, Runtime{
		LibDir: cfg.LibDir, ManagedLibDir: cfg.ManagedLibDir, Provider: cfg.Provider, DeviceID: cfg.DeviceID,
		Threads: cfg.Threads, BatchSize: cfg.BatchSize, Logger: cfg.Logger,
	})
}

// Open prepares the runtime and the model described by spec (downloading
// whatever is missing, see fetch.go), loads them with the spec's tensor
// names and picture preparation, and runs the model once to check that its
// vectors have the expected size. ctx bounds the downloads only, and gets a
// deadline of 30 minutes when it has none of its own; the returned Embedder
// outlives it and is released with Close. A download that was cut short is
// picked up where it stopped by the next Open.
func Open(ctx context.Context, spec ModelSpec, rt Runtime) (*Embedder, error) {
	logger := rt.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	spec, err := spec.normalized()
	if err != nil {
		return nil, err
	}
	provider, err := rt.Provider.normalized()
	if err != nil {
		return nil, fmt.Errorf("onnx: %w", err)
	}
	if rt.LibDir == "" {
		return nil, errors.New("onnx: LibDir is required")
	}
	libDir, err := filepath.Abs(rt.LibDir)
	if err != nil {
		return nil, err
	}
	if rt.Threads <= 0 {
		rt.Threads = defaultThreads()
	}
	if rt.BatchSize <= 0 {
		rt.BatchSize = defaultBatchSize
	}
	ctx, cancel := fetch.WithDefaultTimeout(ctx, downloadTimeout)
	defer cancel()
	if from := libs.loadedDir(); rt.ManagedLibDir && from != "" && from != libDir {
		// before anything is installed into a directory that cannot be loaded
		return nil, fmt.Errorf("onnx: onnxruntime is already loaded from %s; cannot load %s in the same process", from, libDir)
	}
	// Install and load are one step under the directory's lock (see
	// ensureLibs): nothing replaces the library between the check and the
	// load, and a loaded library is never replaced.
	var libPath string
	err = ensureLibs(ctx, libsWant{
		plat: platform(), dir: libDir, managed: rt.ManagedLibDir, variant: variantFor(provider),
		loadedDir: libs.loadedDir,
		use: func() (err error) {
			if libPath, err = libraryPath(libDir); err != nil {
				return err
			}
			return libs.load(api, libPath, logger)
		},
	}, logger)
	if err != nil {
		return nil, err
	}
	if err := ensureModelFile(ctx, spec, logger); err != nil {
		return nil, err
	}
	sess, dims, err := api.open(spec.File, sessionOptions{
		input: spec.Input, output: spec.Output, provider: provider, deviceID: rt.DeviceID, threads: rt.Threads,
	})
	if err != nil {
		return nil, fmt.Errorf("onnx: open model %q: %w", spec.Name, err)
	}
	logger.Info("onnxruntime loaded", logComponent, slog.String("lib", libPath), slog.String("version", libs.version),
		slog.String("provider", string(provider)), slog.Int("threads", rt.Threads), slog.String("model", spec.Name))
	e, err := newEmbedder(sess, spec, dims, rt.BatchSize)
	if err != nil {
		// the probe failed after the session was created: give it back
		sess.close()
		return nil, err
	}
	return e, nil
}

// Embedder implements langai.ImageEmbedder over a session.
type Embedder struct {
	// a session is run one batch at a time: ONNX Runtime parallelizes
	// inside a run, and the pre-bound tensors are not shared
	mu      sync.Mutex
	s       session
	closed  bool // set by Close and read by encode, both under mu
	dims    int
	batch   int
	pooling Pooling
	pre     preprocessor
	model   string // the identity of its vectors, see Model
}

// verify the interface is actually satisfied
var _ langai.ImageEmbedder = (*Embedder)(nil)

// preparation numbers the way this package turns a picture into the model's
// input, beyond what a ModelSpec says. It is part of Model, and is raised
// whenever the same picture and spec would give other pixels:
//
//	1: transparent pixels scaled as premultiplied colour, which made them black
//	2: transparency composed on white (imageutil.ScaleOnWhite)
const preparation = 2

// modelID builds the identity Model documents from a normalized spec.
func modelID(spec ModelSpec) string {
	triple := func(v [3]float32) string {
		return fmt.Sprintf("%g,%g,%g", v[0], v[1], v[2])
	}
	return fmt.Sprintf("%s;output=%s;pooling=%s;side=%d;resize=%s;edge=%d;filter=%s;mean=%s;std=%s;prep=%d",
		spec.Name, spec.Output, spec.Pooling, spec.Side, spec.Resize, spec.ShortestEdge, spec.Interpolation,
		triple(spec.Mean), triple(spec.Std), preparation)
}

// newEmbedder wraps a session and runs it once on a blank picture, which
// catches a provider that was accepted but cannot run, and settles the
// vector size: the model file's declaration (dims, 0 when dynamic), checked
// against the spec's and against what comes out. Split from Open so the
// probe and its failure modes stay testable with a fake session.
func newEmbedder(s session, spec ModelSpec, dims, batch int) (*Embedder, error) {
	if spec.Dims != 0 && dims != 0 && dims != spec.Dims {
		return nil, fmt.Errorf("onnx: model %q declares %d-dimensional vectors, its spec says %d", spec.Name, dims, spec.Dims)
	}
	if dims == 0 {
		dims = spec.Dims
	}
	e := &Embedder{s: s, dims: dims, batch: batch, pooling: spec.Pooling, model: modelID(spec), pre: preprocessor{
		side: spec.Side, shortestEdge: spec.ShortestEdge, resize: spec.Resize, interpolation: spec.Interpolation,
		mean: spec.Mean, std: spec.Std,
	}}
	probe, err := e.encode([]image.Image{image.NewRGBA(image.Rect(0, 0, spec.Side, spec.Side))})
	if err != nil {
		return nil, fmt.Errorf("onnx: probe model %q: %w", spec.Name, err)
	}
	if e.dims == 0 {
		// neither the file nor the spec said: the probe did
		e.dims = len(probe[0])
	}
	if e.dims == 0 {
		return nil, fmt.Errorf("onnx: model %q returned an empty vector", spec.Name)
	}
	return e, nil
}

// Dimensions implements langai.ImageEmbedder.
func (e *Embedder) Dimensions() int { return e.dims }

// Model identifies the vectors this Embedder produces: the string to store
// next to them, the same on every run and different whenever the vector of
// a picture would be. It is the spec's name followed by what else decides
// a vector, the tensor that is read and how a picture is prepared:
//
//	<ModelSpec.Name>;output=<tensor>;pooling=<none|cls|mean>;side=<n>;resize=<squash|crop>;edge=<n>;filter=<bilinear|bicubic>;mean=<r>,<g>,<b>;std=<r>,<g>,<b>;prep=<n>
//
// with the spec's defaults filled in (edge is ShortestEdge) and prep the
// number of this package's own picture preparation, which goes up when a
// release changes the pixels a picture becomes. The default model is
// "google/siglip2-base-patch16-224;output=pooler_output;pooling=none;side=224;resize=squash;edge=224;filter=bilinear;mean=0.5,0.5,0.5;std=0.5,0.5,0.5;prep=2".
// The provider, threads and batch size are not part of it, and neither is
// the ONNX file, so two exports of one model need two names.
func (e *Embedder) Model() string { return e.model }

// EmbedImages implements langai.ImageEmbedder: the pictures go through the
// model BatchSize at a time, and each comes back as a unit-length vector.
func (e *Embedder) EmbedImages(ctx context.Context, images []image.Image) ([][]float32, error) {
	out := make([][]float32, 0, len(images))
	for start := 0; start < len(images); start += e.batch {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+e.batch, len(images))
		vecs, err := e.encode(images[start:end])
		if err != nil {
			return nil, fmt.Errorf("embed images %d to %d: %w", start, end-1, err)
		}
		out = append(out, vecs...)
	}
	return out, nil
}

// Close frees the session; a later EmbedImages fails. Closing twice is a
// no-op.
func (e *Embedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	e.s.close()
	return nil
}

// encode runs one batch through the model and returns one unit-length
// vector per picture. The pictures are prepared outside the lock, which is
// where the Go-side work is; the run itself is serialized.
func (e *Embedder) encode(images []image.Image) ([][]float32, error) {
	pixels, err := e.pre.tensor(images)
	if err != nil {
		return nil, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("onnx: embedder: %w", langai.ErrClosed)
	}
	values, shape, err := e.s.run(pixels, len(images), e.pre.side)
	if err != nil {
		return nil, err
	}
	out, err := pool(e.pooling, values, shape, len(images))
	if err != nil {
		return nil, err
	}
	if e.dims != 0 && len(out[0]) != e.dims {
		return nil, fmt.Errorf("model returned %d-dimensional vectors, expected %d", len(out[0]), e.dims)
	}
	for _, vec := range out {
		langai.Normalize(vec)
	}
	return out, nil
}

// pool turns the model's output, values laid out as shape, into one fresh
// vector per picture: [n, dims] is taken as is, [n, tokens, dims] is
// reduced by the pooling. A shape the pooling does not fit is an error.
func pool(p Pooling, values []float32, shape []int64, n int) ([][]float32, error) {
	rank := len(shape)
	if rank < 2 || rank > 3 || shape[0] != int64(n) || int64(len(values)) != product(shape) {
		return nil, fmt.Errorf("model returned %d values of shape %v for %d images", len(values), shape, n)
	}
	if (rank == 2) != (p == PoolingNone) {
		return nil, fmt.Errorf("model output has shape %v, which pooling %q does not fit", shape, p)
	}
	dims := int(shape[rank-1])
	if dims == 0 {
		return nil, errors.New("model returned empty vectors")
	}
	tokens := 1
	if rank == 3 {
		tokens = int(shape[1])
		if tokens == 0 {
			return nil, errors.New("model returned no tokens")
		}
	}
	out := make([][]float32, n)
	for i := range n {
		seq := values[i*tokens*dims : (i+1)*tokens*dims]
		vec := make([]float32, dims)
		if p == PoolingMean {
			for t := range tokens {
				for j, v := range seq[t*dims : (t+1)*dims] {
					vec[j] += v
				}
			}
			for j := range vec {
				vec[j] /= float32(tokens)
			}
		} else {
			// none: the vector itself; cls: the first token
			copy(vec, seq[:dims])
		}
		out[i] = vec
	}
	return out, nil
}

// product is the number of values a tensor of the shape holds.
func product(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

package yzma

import (
	"context"
	"errors"
	"fmt"
	"image"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	xdraw "golang.org/x/image/draw"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/internal/fetch"
	"github.com/go-bumbu/langai/internal/imageutil"
	"github.com/go-bumbu/langai/media"
)

// DefaultVisionModel is the model NewVision runs unless Config.Model
// selects another: Qwen/Qwen3-VL-Embedding-2B (Apache-2.0), which embeds
// text, images and videos into one 2048-dimensional space. The GGUF files
// behind it are pinned in pinnedVisions.
const DefaultVisionModel = "Qwen/Qwen3-VL-Embedding-2B"

// Defaults for the zero values of VisionOptions.
const (
	// DefaultInstruction is what the model's authors wrap every input in.
	DefaultInstruction = "Represent the user's input."
	// DefaultVisionContextTokens is the context the model gets: room for a
	// segment of DefaultSegmentLength at DefaultFPS, with margin.
	DefaultVisionContextTokens = 16384
	// DefaultSegmentLength is how much of a video goes through the model as
	// one input; at DefaultFPS that is about 3,800 tokens.
	DefaultSegmentLength = 20 * time.Second
	// DefaultFPS is the frame rate a video is sampled at, the one the model
	// was trained with.
	DefaultFPS = 1.0
	// DefaultMaxSegments is the most segments one video is embedded as
	// unless VisionOptions.MaxSegments says otherwise: four hours at
	// DefaultSegmentLength. Every segment is a run of the model, and how
	// many a video asks for comes from its header, so there is always a cap.
	DefaultMaxSegments = 720
	// maxImageSide caps a picture before it reaches the projector, which
	// scales it down to its token budget anyway; it bounds the bitmap copy.
	maxImageSide = 1536
	// visionBatch is the most tokens one decode takes.
	visionBatch = 2048
)

// VisionOptions are how a Vision runs its model, beyond the Runtime that
// every model shares: what NewVision and OpenVision take next to the model
// and the runtime. The zero value runs the model with the defaults its
// authors recommend; negative numbers are an error.
type VisionOptions struct {
	// Instruction is the task the model represents every input for; ""
	// means DefaultInstruction. It goes into the system turn of the prompt,
	// so texts, images and videos embedded under one instruction compare.
	Instruction string
	// ContextTokens is the most tokens one input may take, images and
	// video frames included; 0 means DefaultVisionContextTokens.
	ContextTokens int
	// ImageMinTokens and ImageMaxTokens bound how many tokens a picture or a
	// frame becomes (the projector scales it to fit); 0 keeps the model's
	// defaults. Fewer tokens are faster; the model's authors note that Qwen
	// VL models want at least 1024 for grounding, which embedding is not.
	ImageMinTokens int
	ImageMaxTokens int
	// SegmentLength is how much of a video goes through the model at once;
	// 0 means DefaultSegmentLength. A longer segment sees more context and
	// costs more tokens: about 190 per second at DefaultFPS.
	SegmentLength time.Duration
	// FPS is the frame rate videos are sampled at; 0 means DefaultFPS.
	FPS float64
	// MaxSegments caps the segments per video, so a feature film does not
	// take an hour; 0 means DefaultMaxSegments. What lies past the cap is
	// not embedded. There is no "all of them": set a number as large as the
	// longest video you mean to embed whole.
	MaxSegments int
	// Cutter cuts a video into segments and reads its length; media/ffmpeg
	// has it. Without one EmbedVideo fails.
	Cutter media.Cutter
	// TempDir is where EmbedVideo writes its segment files, in a directory
	// of its own that it removes again; "" means os.TempDir(). One segment
	// is there at a time, each removed once it is embedded. A video without
	// a length is copied there first, up to MaxSegments segments of it, so
	// such a file needs room for that copy: point TempDir at a disk when
	// the default one is kept in memory (a tmpfs).
	TempDir string
}

// VisionSpec describes one vision embedding model: the language model GGUF,
// the projector GGUF (the vision tower), and where both come from.
type VisionSpec struct {
	// Name names the model. It is the first part of the identity to store
	// next to vectors, which Vision.Model completes.
	Name string
	// File is the model GGUF path; when missing and URL is set, it is
	// downloaded there first, verified against SHA256 and Size.
	File   string
	URL    string
	SHA256 string
	Size   int64
	// Projector is the projector GGUF path, likewise.
	Projector       string
	ProjectorURL    string
	ProjectorSHA256 string
	ProjectorSize   int64
	// Dims is the size of the model's vectors; OpenVision refuses a model
	// whose vectors come out another size; 0 skips the check.
	Dims int
}

// validate reports a spec OpenVision cannot run.
func (spec VisionSpec) validate() error {
	switch {
	case spec.Name == "":
		return errors.New("yzma: VisionSpec.Name is required")
	case spec.File == "" || spec.Projector == "":
		return fmt.Errorf("yzma: vision model %q: File and Projector are required", spec.Name)
	case (spec.URL != "" && spec.SHA256 == "") || (spec.ProjectorURL != "" && spec.ProjectorSHA256 == ""):
		return fmt.Errorf("yzma: vision model %q: URL without SHA256; pin the digest (make semeval logs it for an unpinned entry)", spec.Name)
	case spec.Dims < 0 || spec.Size < 0 || spec.ProjectorSize < 0:
		return fmt.Errorf("yzma: vision model %q: Dims, Size and ProjectorSize must not be negative", spec.Name)
	}
	return nil
}

// withDefaults returns o with the defaults its zero values mean filled in:
// the settings a Vision runs with.
func (o VisionOptions) withDefaults() VisionOptions {
	if o.Instruction == "" {
		o.Instruction = DefaultInstruction
	}
	if o.SegmentLength == 0 {
		o.SegmentLength = DefaultSegmentLength
	}
	if o.FPS == 0 {
		o.FPS = DefaultFPS
	}
	if o.MaxSegments == 0 {
		o.MaxSegments = DefaultMaxSegments
	}
	return o
}

// ModelID is the identity of the vectors a Vision opened with spec and
// opts produces, the string its Model returns, computed without loading
// anything: what to hand NewLazyImageEmbedder, or a store, before the model
// loads. See Vision.Model for its form. The identity names the vector
// length, so it needs spec.Dims, which the pinned specs have: with Dims 0 it
// is "", unknown, until the model loads.
func (spec VisionSpec) ModelID(opts VisionOptions) string {
	if spec.Dims <= 0 {
		return ""
	}
	return visionID(spec.Name, spec.Dims, opts.withDefaults())
}

// visionID builds the identity Vision.Model documents from options with
// their defaults filled in.
func visionID(name string, dims int, opts VisionOptions) string {
	return fmt.Sprintf("%s;instruction=%s;dims=%d;image_tokens=%d-%d;fps=%s;segment=%s", name, shortDigest(opts.Instruction),
		dims, opts.ImageMinTokens, opts.ImageMaxTokens, strconv.FormatFloat(opts.FPS, 'g', -1, 64), opts.SegmentLength)
}

// validate reports options no Vision runs with.
func (o VisionOptions) validate() error {
	switch {
	case o.FPS < 0 || o.SegmentLength < 0 || o.MaxSegments < 0 || o.ContextTokens < 0:
		return errors.New("yzma: VisionOptions: FPS, SegmentLength, MaxSegments and ContextTokens must not be negative")
	case o.ImageMinTokens < 0 || o.ImageMaxTokens < 0:
		return errors.New("yzma: VisionOptions: ImageMinTokens and ImageMaxTokens must not be negative")
	case o.ImageMaxTokens != 0 && o.ImageMinTokens > o.ImageMaxTokens:
		return fmt.Errorf("yzma: VisionOptions: ImageMinTokens %d is above ImageMaxTokens %d", o.ImageMinTokens, o.ImageMaxTokens)
	}
	return nil
}

// pinnedVision is one vision model NewVision can run by name.
type pinnedVision struct {
	name      string
	file      string
	asset     asset
	projFile  string
	projAsset asset
	dims      int
}

// pinnedVisions lists the vision models Config.Model selects from in NewVision,
// DefaultVisionModel first. The default is DevQuasar's Q8_0 conversion with
// the f16 projector: its vectors agree with the f16 model's to within 0.005
// of cosine on the sample set and it runs about 15 % faster.
func pinnedVisions() []pinnedVision {
	return []pinnedVision{
		{
			name: DefaultVisionModel,
			file: "Qwen3-VL-Embedding-2B-Q8_0.gguf",
			asset: asset{
				URL:    "https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF/resolve/main/Qwen.Qwen3-VL-Embedding-2B.Q8_0.gguf",
				SHA256: "7552c2f699c546ce46abd6b66b2aa16ae667c88c830efbd352b12224d4613492",
				Size:   1834428000,
			},
			projFile: "mmproj-Qwen3-VL-Embedding-2B-f16.gguf",
			projAsset: asset{
				URL:    "https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF/resolve/main/mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf",
				SHA256: "3f89a7768ffa6606935319f71bf56bb71871249ba549bf1080a0caea7a088613",
				Size:   819395136,
			},
			dims: 2048,
		},
	}
}

// VisionModels lists the names of the pinned vision models NewVision
// selects from, DefaultVisionModel first.
func VisionModels() []string {
	return names(pinnedVisions(), func(m pinnedVision) string { return m.name })
}

// LookupVisionSpec returns the pinned spec of the named vision model, its
// files inside modelsDir ("" means "models"). name "" means
// DefaultVisionModel; any other name must be one of VisionModels, and the
// error for one that is not lists them.
func LookupVisionSpec(name, modelsDir string) (VisionSpec, error) {
	if name == "" {
		name = DefaultVisionModel
	}
	dir := modelsOr(modelsDir)
	for _, m := range pinnedVisions() {
		if m.name == name {
			return VisionSpec{
				Name: m.name, File: filepath.Join(dir, m.file), URL: m.asset.URL, SHA256: m.asset.SHA256, Size: m.asset.Size,
				Projector: filepath.Join(dir, m.projFile), ProjectorURL: m.projAsset.URL, ProjectorSHA256: m.projAsset.SHA256, ProjectorSize: m.projAsset.Size,
				Dims: m.dims,
			}, nil
		}
	}
	return VisionSpec{}, fmt.Errorf("yzma: unsupported vision model %q: use one of %s", name, strings.Join(VisionModels(), ", "))
}

// DefaultVisionSpec is LookupVisionSpec(DefaultVisionModel, modelsDir).
func DefaultVisionSpec(modelsDir string) VisionSpec {
	spec, _ := LookupVisionSpec(DefaultVisionModel, modelsDir) // DefaultVisionModel is always pinned
	return spec
}

// visionDownloadTimeout bounds the first-use downloads of a vision model
// and its projector, 2.7 GB for the pinned one, when the caller's ctx
// carries no deadline of its own.
const visionDownloadTimeout = time.Hour

// NewVision prepares the libraries, the model cfg.Model selects and its
// projector (downloading whatever is missing), loads them and probes the
// model once. ctx bounds the downloads only; the returned Vision outlives
// it. It is OpenVision with LookupVisionSpec(cfg.Model, cfg.ModelsDir),
// cfg.Runtime and opts.
func NewVision(ctx context.Context, cfg Config, opts VisionOptions) (*Vision, error) {
	spec, err := LookupVisionSpec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		return nil, err
	}
	return OpenVision(ctx, spec, cfg.Runtime, opts)
}

// OpenVision prepares the libraries and the model and projector described
// by spec (downloading whatever is missing), loads them on the GPU
// rt.Accel resolves to or the CPU, and probes the model once. The
// libraries load once per process, so a Vision opens next to an Embedder.
// ctx bounds the downloads only, and gets a deadline of an hour when it has
// none of its own; the returned Vision outlives it and is released with
// Close. A download that was cut short is picked up where it stopped by the
// next OpenVision.
func OpenVision(ctx context.Context, spec VisionSpec, rt Runtime, opts VisionOptions) (*Vision, error) {
	if err := spec.validate(); err != nil {
		return nil, err
	}
	if err := opts.validate(); err != nil {
		return nil, err
	}
	if err := rt.validate(); err != nil {
		return nil, err
	}
	ctx, cancel := fetch.WithDefaultTimeout(ctx, visionDownloadTimeout)
	defer cancel()
	libDir, logger, err := rt.prepare(ctx, visionLib, libs)
	if err != nil {
		return nil, err
	}
	for _, m := range []ModelSpec{
		{Name: spec.Name, File: spec.File, URL: spec.URL, SHA256: spec.SHA256, Size: spec.Size},
		{Name: spec.Name + " projector", File: spec.Projector, URL: spec.ProjectorURL, SHA256: spec.ProjectorSHA256, Size: spec.ProjectorSize},
	} {
		if err := ensureModelFile(ctx, m, logger); err != nil {
			return nil, err
		}
	}
	v, err := openVision(visionLib, libs, libDir, spec, opts, rt)
	if err != nil {
		return nil, err
	}
	logger.Info("llama.cpp loaded", logComponent, slog.String("dir", libDir), slog.String("version", libs.version), slog.Int("threads", rt.Threads), slog.String("accel", v.accel), slog.String("model", spec.Name), slog.Int("dims", v.dims), slog.Duration("segment", v.segment), slog.Float64("fps", v.fps))
	return v, nil
}

// visionAPI is the slice of llama.cpp and libmtmd the Vision drives, with
// handles as plain integers so that only vision_ffi.go names the bindings'
// types. ffiAPI is the real implementation; vision_test.go has a fake.
type visionAPI interface {
	runtimeAPI
	loadModel(path string, gpuLayers int32) (model uintptr, err error)
	freeModel(model uintptr)
	nCtxTrain(model uintptr) int32
	// nEmbdOut is the length of the vectors the model puts out (n_embd_out):
	// the size of a pooled embedding, which is not the hidden size (n_embd)
	// for a model that ends in a projection to another width.
	nEmbdOut(model uintptr) int32
	// newVisionContext creates a context of nCtx tokens that decodes batches
	// of up to batch tokens, with embeddings on and last-token pooling.
	// nCtxGot and batchGot are what llama.cpp granted; mem is the KV cache.
	newVisionContext(model uintptr, nCtx, batch int32, threads int) (ctx uintptr, nCtxGot, batchGot int, mem uintptr, err error)
	freeContext(ctx uintptr)
	memoryClear(mem uintptr) error
	// embeddings returns the pooled vector of sequence 0, n floats, as a
	// view the next decode overwrites.
	embeddings(ctx uintptr, n int32) ([]float32, error)
	// chatTemplate is the template the GGUF carries, "" when none.
	chatTemplate(model uintptr) string
	// applyTemplate lays one system and one user message out with the
	// template, ending where the assistant's turn begins.
	applyTemplate(template, system, user string) (string, error)
	// tokenizeWith converts text to tokens. addSpecial adds the model's
	// beginning-of-sequence token, and parseSpecial reads special-token
	// spellings in text as the control tokens they spell.
	tokenizeWith(model uintptr, text string, addSpecial, parseSpecial bool) []int32
	// tokenText is the bytes token spells, without control-token markup.
	tokenText(model uintptr, token int32) []byte
	// loadProjector loads the projector (the vision tower) for model into a
	// libmtmd context, on the GPU when useGPU, with the image token bounds.
	loadProjector(path string, model uintptr, threads int, useGPU bool, imageMinTokens, imageMaxTokens int) (proj uintptr, err error)
	freeProjector(proj uintptr)
	// bitmapRGB wraps w×h RGB bytes; freed with freeBitmap.
	bitmapRGB(w, h int, rgb []byte) (bitmap uintptr, err error)
	// bitmapVideo reads the video at path at fps frames per second as one
	// bitmap; both handles are freed, bitmap with freeBitmap and video with
	// freeVideo, after the chunks made from it are freed.
	bitmapVideo(proj uintptr, path string, fps float64) (bitmap, video uintptr, err error)
	freeBitmap(bitmap uintptr)
	freeVideo(video uintptr)
	// tokenizeParts turns the parts of a prompt, in order, into chunks: the
	// tokens of each text, read for control tokens only where the part says
	// so, and the tokens of each bitmap, nTokens in all. No part is searched
	// for media markers. Freed with freeChunks.
	tokenizeParts(proj uintptr, parts []promptPart) (chunks uintptr, nTokens int, err error)
	freeChunks(chunks uintptr)
	// evalChunks runs the chunks through the projector and the model on
	// sequence 0, batch tokens at a time, and marks the last token as
	// output.
	evalChunks(proj, ctx, chunks uintptr, batch int32) error
	// setAbort makes the decodes of ctx ask stop as they compute and give up
	// when it says so; nil takes the question away again. llama.cpp asks
	// between the steps of a computation on the CPU; a GPU backend, and the
	// projector's own encoding of a picture, run to the end of their batch.
	setAbort(ctx uintptr, stop func() bool)
}

// promptPart is one piece of a prompt in the order the model reads it: a
// bitmap, or a text.
type promptPart struct {
	// text is the piece's text when bitmap is 0.
	text string
	// special reads special-token spellings in text as the control tokens
	// they spell. That is for the chat template's own text; an input from
	// outside stays plain, so a text that spells "<|im_end|>" cannot end its
	// turn.
	special bool
	// bitmap, when not 0, makes the part a picture or a video.
	bitmap uintptr
}

// visionLib is the API OpenVision drives; tests swap in a fake.
var visionLib visionAPI = ffiAPI{}

// maxTruncations is how often a text is cut again when the cut before it
// still left the input longer than the context. One cut is exact for a
// byte-pair vocabulary; the rest is for a vocabulary whose tokens do not add
// up to the text byte for byte.
const maxTruncations = 8

// Vision embeds texts, images and videos into one space with a vision
// language model trained for embedding: the model's chat template lays out
// an instruction and the input, the projector turns pictures and frames
// into tokens, and the vector is the last token's hidden state, unit
// length. It is a langai.Embedder, a langai.ImageEmbedder and a
// langai.VideoEmbedder; search.MediaIndex takes it as all three.
type Vision struct {
	// a llama.cpp context is not safe for concurrent use, so calls are
	// serialized
	mu          sync.Mutex
	api         visionAPI
	model       uintptr
	ctx         uintptr
	mem         uintptr
	proj        uintptr
	template    string
	layout      promptLayout // the template around the instruction and the input
	instruction string
	nCtx        int // tokens the context holds
	batch       int // most tokens one decode takes
	dims        int
	cutter      media.Cutter
	tempDir     string // where segment files go, "" for os.TempDir()
	segment     time.Duration
	fps         float64
	maxSegments int
	accel       string
	id          string // the identity of its vectors, see Model
	closed      bool
}

// verify the interfaces are actually satisfied
var (
	_ langai.Embedder      = (*Vision)(nil)
	_ langai.ImageEmbedder = (*Vision)(nil)
	_ langai.VideoEmbedder = (*Vision)(nil)
)

// openVision loads the libraries (once, through rt), the model and the
// projector, builds the embedding context and probes the model with a
// text, so a broken setup fails here and not at the first call. Whatever
// it created is freed again when a later step fails.
func openVision(api visionAPI, rt *runtimeLoader, libDir string, spec VisionSpec, opts VisionOptions, run Runtime) (*Vision, error) {
	if err := rt.load(api, libDir, run.Logger); err != nil {
		return nil, err
	}
	layers, used, err := resolveAccel(run.Accel, run.GPULayers, rt.devices)
	if err != nil {
		return nil, err
	}
	model, err := api.loadModel(spec.File, layers)
	if err != nil {
		return nil, fmt.Errorf("load model %s: %w", spec.File, err)
	}
	opts = opts.withDefaults()
	v := &Vision{
		api:         api,
		model:       model,
		template:    api.chatTemplate(model),
		instruction: opts.Instruction,
		cutter:      opts.Cutter,
		tempDir:     opts.TempDir,
		segment:     opts.SegmentLength,
		fps:         opts.FPS,
		maxSegments: opts.MaxSegments,
		accel:       used,
	}
	if v.template == "" {
		v.template = "chatml"
	}
	// the template is laid out once, around stand-ins: no instruction and no
	// input ever goes through it (see promptLayout)
	if v.layout, err = layoutTemplate(api.applyTemplate, v.template); err != nil {
		api.freeModel(model)
		return nil, fmt.Errorf("vision model %q: %w", spec.Name, err)
	}
	nCtxTrain := api.nCtxTrain(model)
	if nCtxTrain <= 0 {
		api.freeModel(model)
		return nil, fmt.Errorf("load model %s: no training context length in the GGUF metadata", spec.File)
	}
	want := opts.ContextTokens
	if want == 0 {
		want = DefaultVisionContextTokens
	}
	nCtx := int32(min(int(nCtxTrain), want))
	batch := min(nCtx, visionBatch)
	ctx, nCtxGot, batchGot, mem, err := api.newVisionContext(model, nCtx, batch, run.Threads)
	if err != nil {
		api.freeModel(model)
		return nil, fmt.Errorf("create llama.cpp context: %w", err)
	}
	v.ctx, v.mem, v.nCtx, v.batch = ctx, mem, min(nCtxGot, int(nCtx)), min(batchGot, int(batch))
	if v.nCtx <= 0 || v.batch <= 0 {
		v.release()
		return nil, fmt.Errorf("llama.cpp granted a %d-token context and %d-token batches for %s", v.nCtx, v.batch, spec.File)
	}
	proj, err := api.loadProjector(spec.Projector, model, run.Threads, layers > 0, opts.ImageMinTokens, opts.ImageMaxTokens)
	if err != nil {
		v.release()
		return nil, fmt.Errorf("load projector %s: %w", spec.Projector, err)
	}
	v.proj = proj
	probe, err := v.embedText(context.Background(), "dimension probe")
	if err != nil {
		v.release()
		return nil, fmt.Errorf("probe vision model %q: %w", spec.Name, err)
	}
	v.dims = len(probe)
	if spec.Dims != 0 && v.dims != spec.Dims {
		v.release()
		return nil, fmt.Errorf("yzma: vision model %q produces %d-dimensional vectors, its spec says %d", spec.Name, v.dims, spec.Dims)
	}
	v.id = visionID(spec.Name, v.dims, opts)
	return v, nil
}

// release frees the projector, the context and the model, whichever exist.
func (v *Vision) release() {
	if v.proj != 0 {
		v.api.freeProjector(v.proj)
		v.proj = 0
	}
	if v.ctx != 0 {
		v.api.freeContext(v.ctx)
		v.ctx = 0
	}
	if v.model != 0 {
		v.api.freeModel(v.model)
		v.model = 0
	}
}

// Dimensions implements langai.Embedder, langai.ImageEmbedder and
// langai.VideoEmbedder.
func (v *Vision) Dimensions() int { return v.dims }

// Accel says what runs the model, as the load log line does: "cpu" or the
// GPU device.
func (v *Vision) Accel() string { return v.accel }

// SegmentLength is how much of a video goes through the model at once.
func (v *Vision) SegmentLength() time.Duration { return v.segment }

// Model identifies the vectors this Vision produces: the string to store
// next to them, the same on every run and different whenever the vector of
// some input would be. It is the spec's name followed by what else decides
// a vector:
//
//	<VisionSpec.Name>;instruction=<id>;dims=<n>;image_tokens=<min>-<max>;fps=<f>;segment=<duration>
//
// <id> is the first eight hex digits of the SHA-256 of the instruction,
// which sets the space every input lands in; dims is the vector length;
// image_tokens are ImageMinTokens and ImageMaxTokens, 0 for the model's
// own, which set the resolution a picture is seen at; fps and segment (as
// time.Duration prints it) set what a video segment holds. With the
// defaults that is
// "Qwen/Qwen3-VL-Embedding-2B;instruction=<id>;dims=2048;image_tokens=0-0;fps=1;segment=20s".
// The context size, MaxSegments, the threads and the accelerator are not
// part of it: they do not move the vector of anything that is embedded.
// VisionSpec.ModelID gives the same string before the model loads.
func (v *Vision) Model() string { return v.id }

// Close frees the projector, the model and its context; a later call
// fails. Closing twice is a no-op.
func (v *Vision) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed {
		return nil
	}
	v.closed = true
	v.release()
	return nil
}

// Embed implements langai.Embedder: each text is one input under the
// instruction. The role makes no difference: the model is symmetric and the
// instruction says what the vectors are for. A text is plain text to the
// model whatever it spells (a control token, libmtmd's media marker, a NUL
// byte), a text beyond 32 KiB is cut, and so is one that does not fit the
// context next to the instruction: the model then sees its beginning, as
// with the text Embedder. A ctx that ends stops the evaluation (see
// visionAPI.setAbort) and is the error.
func (v *Vision) Embed(ctx context.Context, _ langai.Role, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vec, err := v.embedText(ctx, clampText(text))
		if err != nil {
			return nil, fmt.Errorf("embed text %d: %w", i, err)
		}
		out[i] = vec
	}
	return out, nil
}

// EmbedImages implements langai.ImageEmbedder: each picture is one input
// under the instruction, scaled down to maxImageSide first when larger.
func (v *Vision) EmbedImages(ctx context.Context, images []image.Image) ([][]float32, error) {
	out := make([][]float32, len(images))
	for i, img := range images {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if img == nil || img.Bounds().Empty() {
			return nil, fmt.Errorf("embed image %d: empty image", i)
		}
		w, h, rgb := rgbBytes(img)
		vec, err := v.embedBitmap(ctx, func() (uintptr, func(), error) {
			b, err := v.api.bitmapRGB(w, h, rgb)
			if err != nil {
				return 0, nil, err
			}
			return b, func() { v.api.freeBitmap(b) }, nil
		})
		if err != nil {
			return nil, fmt.Errorf("embed image %d: %w", i, err)
		}
		out[i] = vec
	}
	return out, nil
}

// EmbedVideo implements langai.VideoEmbedder: the video is cut into segments
// of SegmentLength (the Cutter does it, by stream copy), each sampled at
// FPS and embedded as one input under the instruction. A video no longer
// than a segment is one input, uncut. At most MaxSegments segments are
// embedded (DefaultMaxSegments unless set); what lies past them is left
// out.
//
// The length is what the Cutter reads from the file, and the file may be
// wrong. A video that ends before its header says is embedded up to its
// end: the walk stops at the Cutter's media.ErrPastEnd, and the last
// segment's End is then where that segment was planned to end, not where
// the pictures did. A length that no video has (nothing, less, more than
// media.MaxDuration) is taken for none.
//
// A file without a length (a recording that was never finalized, a stream
// dump) is never taken for a short one. The Cutter first copies it into a
// Matroska file, which has a length and an index: up to MaxSegments
// segments of it, next to the segments in the temp directory. The copy is
// then cut like any other video. A file that has no length even as a copy
// (a raw elementary stream) is embedded as its first segment only.
//
// The segment files are written into a directory of this call's own inside
// VisionOptions.TempDir, one at a time: each is removed once it is
// embedded, and the directory when the call returns. So a video takes the
// room of one segment there, and one without a length that of its copy as
// well, which is at most MaxSegments segments long.
func (v *Vision) EmbedVideo(ctx context.Context, path string) ([]langai.Segment, error) {
	if v.cutter == nil {
		return nil, errors.New("yzma: no Cutter: set VisionOptions.Cutter (media/ffmpeg) to embed videos")
	}
	duration, err := v.cutter.Duration(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("yzma: probe video: %w", err)
	}
	if duration < 0 || duration > media.MaxDuration {
		duration = 0 // not a length: measured below, like a file that states none
	}
	if duration > 0 && duration <= v.segment {
		vec, err := v.embedVideoFile(ctx, path)
		if err != nil {
			return nil, err
		}
		return []langai.Segment{{Start: 0, End: duration, Vector: vec}}, nil
	}
	dir, err := os.MkdirTemp(v.tempDir, "langai-vision-*")
	if err != nil {
		return nil, fmt.Errorf("yzma: a directory for the segments (VisionOptions.TempDir): %w", err)
	}
	defer func() { _ = os.RemoveAll(dir) }()
	if duration <= 0 {
		if path, duration, err = v.measured(ctx, path, dir); err != nil {
			return nil, err
		}
	}
	return v.embedSegments(ctx, path, duration, dir)
}

// measured gives a video whose container states no length one. The Cutter
// copies it, from its start and for at most MaxSegments segments, into a
// Matroska file in dir, and that copy and its length are returned in the
// video's place. Asking the Cutter for stretch after stretch of the
// original until one comes back empty does not work: a cut that starts
// past the end of a stream copy is not empty, it is the last group of
// pictures again. When even the copy has no length, the original comes
// back with one segment's length, so that its first segment is what gets
// embedded: bounded, where the whole file would not be.
func (v *Vision) measured(ctx context.Context, path, dir string) (string, time.Duration, error) {
	remux := filepath.Join(dir, "source"+fallbackVideoExt)
	// as much as the cap lets through, and never more than a video is
	// believed to be long, which also keeps the product inside a Duration
	bound := media.MaxDuration
	if int64(v.maxSegments) < int64(media.MaxDuration/v.segment) {
		bound = time.Duration(v.maxSegments) * v.segment
	}
	if err := v.cutter.Cut(ctx, path, 0, bound, remux); err == nil {
		if d, err := v.cutter.Duration(ctx, remux); err == nil && d > 0 && d <= media.MaxDuration {
			return remux, d, nil
		}
	}
	if err := ctx.Err(); err != nil {
		return "", 0, err
	}
	return path, v.segment, nil
}

// fallbackVideoExt names the container segments are cut into when the
// video's own file name does not name one known to work (see segmentExts):
// Matroska takes nearly any codec.
const fallbackVideoExt = ".mkv"

// segmentExts maps the extension of a video, in lower case, to the
// extension its segments are cut into. The Cutter picks a segment's
// container by that extension, so only the ones are listed whose container
// was seen to take, by stream copy, what files of that name usually hold
// (ffmpeg 8): the MP4 family (.m4v goes into .mp4, because the muxer ffmpeg
// has behind .m4v refuses HEVC), Matroska and WebM, AVI, MPEG transport and
// program streams, and raw H.264 and HEVC streams, which have no timestamps
// to put into a Matroska file. Every other name, including none, one ffmpeg
// knows no container for (.divx, .rmvb, .ogm) and the suffix of a backup or
// a download (.bak, .part), gets fallbackVideoExt.
var segmentExts = map[string]string{
	".mp4": ".mp4", ".mov": ".mov", ".m4v": ".mp4",
	".mkv": ".mkv", ".webm": ".webm",
	".avi": ".avi",
	".ts":  ".ts", ".mpg": ".mpg", ".mpeg": ".mpeg", ".vob": ".vob",
	".h264": ".h264", ".h265": ".h265", ".hevc": ".hevc",
}

// segmentExt is the extension the segments of the video at path are cut
// into (see segmentExts).
func segmentExt(path string) string {
	if ext, ok := segmentExts[strings.ToLower(filepath.Ext(path))]; ok {
		return ext
	}
	return fallbackVideoExt
}

// segmentCount is how many segments of the given length a video of the
// given duration is cut into, the last one shorter, and at most limit. It
// divides instead of rounding up by addition, so no duration overflows it.
func segmentCount(duration, segment time.Duration, limit int) int {
	n := int64(duration / segment)
	if duration%segment > 0 {
		n++
	}
	return int(min(n, int64(limit)))
}

// embedSegments cuts the video of the given length into segments, into dir,
// and embeds each; MaxSegments caps how many. The length is a plan and not
// a fact: the walk ends early where the Cutter finds the video over
// (media.ErrPastEnd), and nothing is sized by the count.
func (v *Vision) embedSegments(ctx context.Context, path string, duration time.Duration, dir string) ([]langai.Segment, error) {
	ext := segmentExt(path)
	var out []langai.Segment
	for i := range segmentCount(duration, v.segment, v.maxSegments) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		start := time.Duration(i) * v.segment
		end := min(start+v.segment, duration)
		cut := filepath.Join(dir, fmt.Sprintf("segment-%03d%s", i, ext))
		err := v.cutter.Cut(ctx, path, start, v.segment, cut)
		if i > 0 && errors.Is(err, media.ErrPastEnd) {
			break // the video is shorter than its header said
		}
		if err != nil {
			return nil, fmt.Errorf("yzma: cut segment %d: %w", i, err)
		}
		vec, err := v.embedVideoFile(ctx, cut)
		// the segment is done with, whatever came of it: one is on disk at a
		// time, not the whole video
		_ = os.Remove(cut)
		if err != nil {
			return nil, fmt.Errorf("yzma: segment %d: %w", i, err)
		}
		out = append(out, langai.Segment{Start: start, End: end, Vector: vec})
	}
	return out, nil
}

// embedVideoFile embeds one video file as one input.
func (v *Vision) embedVideoFile(ctx context.Context, path string) ([]float32, error) {
	return v.embedBitmap(ctx, func() (uintptr, func(), error) {
		b, video, err := v.api.bitmapVideo(v.proj, path, v.fps)
		if err != nil {
			return 0, nil, err
		}
		return b, func() { v.api.freeBitmap(b); v.api.freeVideo(video) }, nil
	})
}

// embedBitmap embeds one bitmap (a picture or a video) under the
// instruction; open makes the bitmap and returns how to free it, which
// happens after the chunks made from it are gone.
func (v *Vision) embedBitmap(ctx context.Context, open func() (uintptr, func(), error)) ([]float32, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed {
		return nil, fmt.Errorf("yzma: vision embedder: %w", langai.ErrClosed)
	}
	bitmap, free, err := open()
	if err != nil {
		return nil, err
	}
	defer free()
	return v.embedLocked(ctx, "", []uintptr{bitmap})
}

// embedText embeds one text under the instruction.
func (v *Vision) embedText(ctx context.Context, text string) ([]float32, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed {
		return nil, fmt.Errorf("yzma: vision embedder: %w", langai.ErrClosed)
	}
	return v.embedLocked(ctx, text, nil)
}

// parts lays one input out the way the chat template places a system and a
// user message: the instruction as the system message, then the bitmaps and
// the text as the user message. Only the template's own pieces are read for
// control tokens.
func (v *Vision) parts(text string, bitmaps []uintptr) []promptPart {
	parts := make([]promptPart, 0, 5+len(bitmaps))
	parts = append(parts,
		promptPart{text: v.layout.before, special: true},
		promptPart{text: v.instruction},
		promptPart{text: v.layout.between, special: true})
	for _, b := range bitmaps {
		parts = append(parts, promptPart{bitmap: b})
	}
	if text != "" {
		parts = append(parts, promptPart{text: text})
	}
	return append(parts, promptPart{text: v.layout.after, special: true})
}

// tokenize turns one input into chunks of at most the context's tokens
// where a text can make it so: a text that pushes the input past the
// context is cut from its end, as the text Embedder truncates, instead of
// failing the batch it came in. Pictures and videos are not cut, so n can
// still come back larger than the context; the caller refuses that. The
// chunks are freed by the caller.
func (v *Vision) tokenize(text string, bitmaps []uintptr) (chunks uintptr, n int, err error) {
	for cuts := 0; ; cuts++ {
		chunks, n, err = v.api.tokenizeParts(v.proj, v.parts(text, bitmaps))
		if err != nil || n <= v.nCtx || text == "" || cuts == maxTruncations {
			return chunks, n, err
		}
		v.api.freeChunks(chunks)
		text = v.shorten(text, n-v.nCtx)
	}
}

// shorten returns the beginning of text that tokenizes to about over tokens
// fewer, "" when it has no more than that. The cut is where the token that
// has to go begins, found by adding up the bytes of the tokens kept, and
// moved back to a character boundary.
func (v *Vision) shorten(text string, over int) string {
	tokens := v.api.tokenizeWith(v.model, text, false, false)
	keep := len(tokens) - over
	if keep <= 0 {
		return ""
	}
	cut := 0
	for _, tok := range tokens[:keep] {
		cut += len(v.api.tokenText(v.model, tok))
	}
	if cut >= len(text) {
		// the tokens do not spell the text byte for byte (another kind of
		// vocabulary): cut in proportion, and let tokenize check
		cut = len(text) * keep / len(tokens)
	}
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut]
}

// embedLocked tokenizes one input (a text, bitmaps, or both) as the parts
// of a prompt, runs it through the model and reads the pooled vector,
// normalized. A ctx that ends while the model computes stops it and is the
// error. The caller holds the mutex.
func (v *Vision) embedLocked(ctx context.Context, text string, bitmaps []uintptr) ([]float32, error) {
	chunks, n, err := v.tokenize(text, bitmaps)
	if err != nil {
		return nil, fmt.Errorf("tokenize: %w", err)
	}
	defer v.api.freeChunks(chunks)
	if n <= 0 {
		return nil, errors.New("tokenizer returned no tokens")
	}
	if n > v.nCtx {
		return nil, fmt.Errorf("input of %d tokens exceeds the %d-token context: raise ContextTokens, lower ImageMaxTokens or shorten SegmentLength", n, v.nCtx)
	}
	if v.mem != 0 {
		if err := v.api.memoryClear(v.mem); err != nil {
			return nil, fmt.Errorf("clear kv cache: %w", err)
		}
	}
	if err := v.eval(ctx, chunks); err != nil {
		return nil, err
	}
	dims := v.dims
	if dims == 0 { // the probe: the model says
		dims = int(v.api.nEmbdOut(v.model))
	}
	view, err := v.api.embeddings(v.ctx, int32(dims)) //nolint:gosec // a model's width, a small int
	if err != nil {
		return nil, fmt.Errorf("read embeddings: %w", err)
	}
	if len(view) == 0 {
		return nil, errors.New("model returned no vector")
	}
	out := make([]float32, len(view))
	copy(out, view)
	langai.Normalize(out)
	return out, nil
}

// eval runs the chunks through the projector and the model. While it does,
// llama.cpp asks whether ctx has ended and gives up when it has; that is
// reported as ctx's error, whatever the evaluation itself returned.
func (v *Vision) eval(ctx context.Context, chunks uintptr) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if ctx.Done() != nil {
		v.api.setAbort(v.ctx, func() bool { return ctx.Err() != nil })
		defer v.api.setAbort(v.ctx, nil)
	}
	if err := v.api.evalChunks(v.proj, v.ctx, chunks, int32(v.batch)); err != nil { //nolint:gosec // batch is at most visionBatch
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return fmt.Errorf("evaluate: %w", err)
	}
	return nil
}

// rgbBytes returns img as packed RGB, scaled down so that its longer side
// is at most maxImageSide (the projector shrinks it to its token budget
// anyway) and composed on white where it is transparent (see
// imageutil.ScaleOnWhite, which provider/onnx prepares pictures with too).
func rgbBytes(img image.Image) (w, h int, rgb []byte) {
	b := img.Bounds()
	w, h = b.Dx(), b.Dy()
	if long := max(w, h); long > maxImageSide {
		w, h = max(1, w*maxImageSide/long), max(1, h*maxImageSide/long)
	}
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	imageutil.ScaleOnWhite(rgba, rgba.Bounds(), img, xdraw.CatmullRom)
	rgb = make([]byte, 0, w*h*3)
	for i := 0; i < len(rgba.Pix); i += 4 {
		rgb = append(rgb, rgba.Pix[i], rgba.Pix[i+1], rgba.Pix[i+2])
	}
	return w, h, rgb
}

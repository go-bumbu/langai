package yzma

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	xdraw "golang.org/x/image/draw"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/internal/fetch"
	"github.com/go-bumbu/langai/media"
)

// DefaultVisionModel is the model NewVision runs unless VisionConfig.Model
// selects another: Qwen/Qwen3-VL-Embedding-2B (Apache-2.0), which embeds
// text, images and videos into one 2048-dimensional space. The GGUF files
// behind it are pinned in pinnedVisions.
const DefaultVisionModel = "Qwen/Qwen3-VL-Embedding-2B"

// Defaults for the zero values of VisionConfig.
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
	// maxImageSide caps a picture before it reaches the projector, which
	// scales it down to its token budget anyway; it bounds the bitmap copy.
	maxImageSide = 1536
	// visionBatch is the most tokens one decode takes.
	visionBatch = 2048
)

// VisionConfig selects the vision model and how it runs. The embedded Config
// names the model (VisionConfig.Model, "" meaning DefaultVisionModel), the
// directories, the threads and the acceleration, as for New.
type VisionConfig struct {
	Config
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
	// take an hour; 0 means all of them.
	MaxSegments int
	// Cutter cuts a video into segments and reads its length; media/ffmpeg
	// has it. Without one EmbedVideo fails.
	Cutter media.Cutter
}

// VisionSpec describes one vision embedding model: the language model GGUF,
// the projector GGUF (the vision tower), and where both come from.
type VisionSpec struct {
	// Name is the logical identity cached alongside vectors.
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

// pinnedVision is one vision model NewVision can run by name.
type pinnedVision struct {
	name      string
	file      string
	asset     asset
	projFile  string
	projAsset asset
	dims      int
}

// pinnedVisions lists the vision models VisionConfig.Model selects from,
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

// LookupVisionSpec returns the pinned spec of the named vision model, its
// files inside modelsDir ("" means "models"). name "" means
// DefaultVisionModel; any other name must be one of the pinned vision
// models, and the error for one that is not lists them.
func LookupVisionSpec(name, modelsDir string) (VisionSpec, error) {
	if name == "" {
		name = DefaultVisionModel
	}
	if modelsDir == "" {
		modelsDir = "models"
	}
	models := pinnedVisions()
	names := make([]string, 0, len(models))
	for _, m := range models {
		if m.name == name {
			return VisionSpec{
				Name: m.name, File: filepath.Join(modelsDir, m.file), URL: m.asset.URL, SHA256: m.asset.SHA256, Size: m.asset.Size,
				Projector: filepath.Join(modelsDir, m.projFile), ProjectorURL: m.projAsset.URL, ProjectorSHA256: m.projAsset.SHA256, ProjectorSize: m.projAsset.Size,
				Dims: m.dims,
			}, nil
		}
		names = append(names, m.name)
	}
	return VisionSpec{}, fmt.Errorf("unsupported vision model %q: use one of %s", name, strings.Join(names, ", "))
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
// it. It is OpenVision with LookupVisionSpec(cfg.Model, cfg.ModelsDir).
func NewVision(ctx context.Context, cfg VisionConfig) (*Vision, error) {
	spec, err := LookupVisionSpec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		return nil, err
	}
	return OpenVision(ctx, spec, cfg)
}

// OpenVision prepares the libraries and the model and projector described
// by spec (downloading whatever is missing), loads them on the GPU
// cfg.Accel resolves to or the CPU, and probes the model once. The
// libraries load once per process, so a Vision opens next to an Embedder.
// ctx bounds the downloads only, and gets a deadline of an hour when it has
// none of its own; the returned Vision outlives it and is released with
// Close. A download that was cut short is picked up where it stopped by the
// next OpenVision.
func OpenVision(ctx context.Context, spec VisionSpec, cfg VisionConfig) (*Vision, error) {
	if spec.Name == "" {
		return nil, errors.New("yzma: VisionSpec.Name is required")
	}
	if spec.File == "" || spec.Projector == "" {
		return nil, fmt.Errorf("yzma: vision model %q: File and Projector are required", spec.Name)
	}
	if (spec.URL != "" && spec.SHA256 == "") || (spec.ProjectorURL != "" && spec.ProjectorSHA256 == "") {
		return nil, fmt.Errorf("yzma: vision model %q: URL without SHA256; pin the digest (make semeval logs it for an unpinned entry)", spec.Name)
	}
	if cfg.FPS < 0 || cfg.SegmentLength < 0 || cfg.MaxSegments < 0 || cfg.ContextTokens < 0 {
		return nil, fmt.Errorf("yzma: vision model %q: FPS, SegmentLength, MaxSegments and ContextTokens must not be negative", spec.Name)
	}
	ctx, cancel := fetch.WithDefaultTimeout(ctx, visionDownloadTimeout)
	defer cancel()
	rt := cfg.runtime()
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
	v, err := openVision(visionLib, libs, libDir, spec, cfg, rt)
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
	// loadProjector loads the projector (the vision tower) for model into a
	// libmtmd context, on the GPU when useGPU, with the image token bounds.
	loadProjector(path string, model uintptr, threads int, useGPU bool, imageMinTokens, imageMaxTokens int) (proj uintptr, err error)
	freeProjector(proj uintptr)
	// marker is the text that stands for a bitmap in a prompt.
	marker() string
	// bitmapRGB wraps w×h RGB bytes; freed with freeBitmap.
	bitmapRGB(w, h int, rgb []byte) (bitmap uintptr, err error)
	// bitmapVideo reads the video at path at fps frames per second as one
	// bitmap; both handles are freed, bitmap with freeBitmap and video with
	// freeVideo, after the chunks made from it are freed.
	bitmapVideo(proj uintptr, path string, fps float64) (bitmap, video uintptr, err error)
	freeBitmap(bitmap uintptr)
	freeVideo(video uintptr)
	// tokenizeMedia turns the prompt, with one marker per bitmap, into chunks:
	// text tokens and the bitmaps' tokens, nTokens in all. Freed with
	// freeChunks.
	tokenizeMedia(proj uintptr, prompt string, bitmaps []uintptr) (chunks uintptr, nTokens int, err error)
	freeChunks(chunks uintptr)
	// evalChunks runs the chunks through the projector and the model on
	// sequence 0, batch tokens at a time, and marks the last token as
	// output.
	evalChunks(proj, ctx, chunks uintptr, batch int32) error
}

// visionLib is the API OpenVision drives; tests swap in a fake.
var visionLib visionAPI = ffiAPI{}

// Vision embeds texts, images and videos into one space with a vision
// language model trained for embedding: the model's chat template lays out
// an instruction and the input, the projector turns pictures and frames
// into tokens, and the vector is the last token's hidden state, unit
// length. It is a langai.Embedder, a langai.ImageEmbedder and a
// media.VideoEmbedder; search.MediaIndex takes it as all three.
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
	instruction string
	marker      string
	nCtx        int // tokens the context holds
	batch       int // most tokens one decode takes
	dims        int
	cutter      media.Cutter
	segment     time.Duration
	fps         float64
	maxSegments int
	accel       string
	closed      bool
}

// verify the interfaces are actually satisfied
var (
	_ langai.Embedder      = (*Vision)(nil)
	_ langai.ImageEmbedder = (*Vision)(nil)
	_ media.VideoEmbedder  = (*Vision)(nil)
)

// openVision loads the libraries (once, through rt), the model and the
// projector, builds the embedding context and probes the model with a
// text, so a broken setup fails here and not at the first call. Whatever
// it created is freed again when a later step fails.
func openVision(api visionAPI, rt *runtimeLoader, libDir string, spec VisionSpec, cfg VisionConfig, run Runtime) (*Vision, error) {
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
	v := &Vision{
		api:         api,
		model:       model,
		template:    api.chatTemplate(model),
		instruction: cfg.Instruction,
		cutter:      cfg.Cutter,
		segment:     cfg.SegmentLength,
		fps:         cfg.FPS,
		maxSegments: cfg.MaxSegments,
		accel:       used,
	}
	if v.template == "" {
		v.template = "chatml"
	}
	if v.instruction == "" {
		v.instruction = DefaultInstruction
	}
	if v.segment == 0 {
		v.segment = DefaultSegmentLength
	}
	if v.fps == 0 {
		v.fps = DefaultFPS
	}
	nCtxTrain := api.nCtxTrain(model)
	if nCtxTrain <= 0 {
		api.freeModel(model)
		return nil, fmt.Errorf("load model %s: no training context length in the GGUF metadata", spec.File)
	}
	want := cfg.ContextTokens
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
	proj, err := api.loadProjector(spec.Projector, model, run.Threads, layers > 0, cfg.ImageMinTokens, cfg.ImageMaxTokens)
	if err != nil {
		v.release()
		return nil, fmt.Errorf("load projector %s: %w", spec.Projector, err)
	}
	v.proj = proj
	v.marker = api.marker() // libmtmd is loaded now
	probe, err := v.embedPrompt(context.Background(), "dimension probe", nil)
	if err != nil {
		v.release()
		return nil, fmt.Errorf("probe vision model %q: %w", spec.Name, err)
	}
	v.dims = len(probe)
	if spec.Dims != 0 && v.dims != spec.Dims {
		v.release()
		return nil, fmt.Errorf("yzma: vision model %q produces %d-dimensional vectors, its spec says %d", spec.Name, v.dims, spec.Dims)
	}
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
// media.VideoEmbedder.
func (v *Vision) Dimensions() int { return v.dims }

// Accel says what runs the model, as the load log line does: "cpu" or the
// GPU device.
func (v *Vision) Accel() string { return v.accel }

// SegmentLength is how much of a video goes through the model at once.
func (v *Vision) SegmentLength() time.Duration { return v.segment }

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
// instruction says what the vectors are for.
func (v *Vision) Embed(ctx context.Context, _ langai.Role, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vec, err := v.embedPrompt(ctx, clampText(text), nil)
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

// EmbedVideo implements media.VideoEmbedder: the video is cut into segments
// of SegmentLength (the Cutter does it, by stream copy), each sampled at
// FPS and embedded as one input under the instruction. A video no longer
// than a segment is one input, uncut.
func (v *Vision) EmbedVideo(ctx context.Context, path string) ([]media.Segment, error) {
	if v.cutter == nil {
		return nil, errors.New("yzma: no Cutter: set VisionConfig.Cutter (media/ffmpeg) to embed videos")
	}
	duration, err := v.cutter.Duration(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("yzma: probe video: %w", err)
	}
	if duration <= v.segment {
		vec, err := v.embedVideoFile(ctx, path)
		if err != nil {
			return nil, err
		}
		return []media.Segment{{Start: 0, End: duration, Vector: vec}}, nil
	}
	n := int((duration + v.segment - 1) / v.segment)
	if v.maxSegments > 0 {
		n = min(n, v.maxSegments)
	}
	dir, err := os.MkdirTemp("", "langai-vision-*")
	if err != nil {
		return nil, err
	}
	defer func() { _ = os.RemoveAll(dir) }()
	out := make([]media.Segment, 0, n)
	for i := range n {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		start := time.Duration(i) * v.segment
		end := min(start+v.segment, duration)
		cut := filepath.Join(dir, fmt.Sprintf("segment-%03d%s", i, filepath.Ext(path)))
		if err := v.cutter.Cut(ctx, path, start, v.segment, cut); err != nil {
			return nil, fmt.Errorf("yzma: cut segment %d: %w", i, err)
		}
		vec, err := v.embedVideoFile(ctx, cut)
		if err != nil {
			return nil, fmt.Errorf("yzma: segment %d: %w", i, err)
		}
		out = append(out, media.Segment{Start: start, End: end, Vector: vec})
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
	return v.embedLocked(ctx, v.marker, []uintptr{bitmap})
}

// embedPrompt embeds one input: the user text, with one marker per bitmap
// in front of it.
func (v *Vision) embedPrompt(ctx context.Context, text string, bitmaps []uintptr) ([]float32, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed {
		return nil, fmt.Errorf("yzma: vision embedder: %w", langai.ErrClosed)
	}
	return v.embedLocked(ctx, strings.Repeat(v.marker, len(bitmaps))+text, bitmaps)
}

// embedLocked lays the prompt out with the chat template, tokenizes it with
// the bitmaps, runs it through the model and reads the pooled vector,
// normalized. The caller holds the mutex.
func (v *Vision) embedLocked(_ context.Context, user string, bitmaps []uintptr) ([]float32, error) {
	prompt, err := v.api.applyTemplate(v.template, v.instruction, user)
	if err != nil {
		return nil, fmt.Errorf("lay out the prompt: %w", err)
	}
	chunks, n, err := v.api.tokenizeMedia(v.proj, prompt, bitmaps)
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
	if err := v.api.evalChunks(v.proj, v.ctx, chunks, int32(v.batch)); err != nil { //nolint:gosec // batch is at most visionBatch
		return nil, fmt.Errorf("evaluate: %w", err)
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

// rgbBytes returns img as packed RGB, scaled down so that its longer side
// is at most maxImageSide (the projector shrinks it to its token budget
// anyway) and composed on white where it is transparent.
func rgbBytes(img image.Image) (w, h int, rgb []byte) {
	b := img.Bounds()
	w, h = b.Dx(), b.Dy()
	if long := max(w, h); long > maxImageSide {
		w, h = max(1, w*maxImageSide/long), max(1, h*maxImageSide/long)
	}
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(rgba, rgba.Bounds(), image.White, image.Point{}, draw.Src)
	if w == b.Dx() && h == b.Dy() {
		draw.Draw(rgba, rgba.Bounds(), img, b.Min, draw.Over)
	} else {
		xdraw.CatmullRom.Scale(rgba, rgba.Bounds(), img, b, draw.Over, nil)
	}
	rgb = make([]byte, 0, w*h*3)
	for i := 0; i < len(rgba.Pix); i += 4 {
		rgb = append(rgb, rgba.Pix[i], rgba.Pix[i+1], rgba.Pix[i+2])
	}
	return w, h, rgb
}

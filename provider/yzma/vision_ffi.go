package yzma

import (
	"errors"
	"fmt"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

// mtmdLoader loads libmtmd once per process, from the directory llama.cpp
// was loaded from, the first time a projector is needed: a text-only
// operator directory without it still serves the Embedder and the
// Completer.
var mtmdLoader struct {
	once sync.Once
	err  error
}

func loadMTMD() error {
	mtmdLoader.once.Do(func() {
		if err := mtmd.Load(llama.LibPath()); err != nil {
			mtmdLoader.err = fmt.Errorf("load libmtmd from %s: %w (the pinned llama.cpp build ships it; a managed LibDir installs it)", llama.LibPath(), err)
			return
		}
		// libmtmd narrates every picture it encodes; route it like llama.cpp's
		// own lines, to the logger load installed (warnings and errors only)
		mtmd.LogSet(llamaLogCallback)
	})
	return mtmdLoader.err
}

func (ffiAPI) newVisionContext(model uintptr, nCtx, batch int32, threads int) (uintptr, int, int, uintptr, error) {
	n, b := uint32(nCtx), uint32(batch) //nolint:gosec // both are small positive sizes openVision derived from the model's window
	p := llama.ContextDefaultParams()
	p.NCtx = n
	p.NBatch = b
	p.NUbatch = b
	p.NSeqMax = 1
	p.PoolingType = llama.PoolingTypeLast
	p.Embeddings = 1
	if threads > 0 {
		t := int32(threads) //nolint:gosec // threads is a small positive config value
		p.NThreads = t
		p.NThreadsBatch = t
	}
	ctx, err := llama.InitFromModel(llama.Model(model), p)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	mem, _ := llama.GetMemory(ctx)
	return uintptr(ctx), int(llama.NCtx(ctx)), min(int(llama.NBatch(ctx)), int(llama.NUBatch(ctx))), uintptr(mem), nil
}

func (ffiAPI) chatTemplate(model uintptr) string {
	return llama.ModelChatTemplate(llama.Model(model), "")
}

func (ffiAPI) loadProjector(path string, model uintptr, threads int, useGPU bool, imageMinTokens, imageMaxTokens int) (uintptr, error) {
	if err := loadMTMD(); err != nil {
		return 0, err
	}
	p := mtmd.ContextParamsDefault()
	p.UseGPU = useGPU
	if threads > 0 {
		p.Threads = int32(threads) //nolint:gosec // threads is a small positive config value
	}
	if imageMinTokens > 0 {
		p.ImageMinTokens = int32(imageMinTokens) //nolint:gosec // a token budget, a small int
	}
	if imageMaxTokens > 0 {
		p.ImageMaxTokens = int32(imageMaxTokens) //nolint:gosec // a token budget, a small int
	}
	ctx, err := mtmd.InitFromFile(path, llama.Model(model), p)
	if err != nil {
		return 0, err
	}
	if !mtmd.SupportVision(ctx) {
		_ = mtmd.Free(ctx)
		return 0, errors.New("the projector has no vision support")
	}
	return uintptr(ctx), nil
}

func (ffiAPI) freeProjector(proj uintptr) { _ = mtmd.Free(mtmd.Context(proj)) }

func (ffiAPI) marker() string { return mtmd.DefaultMarker() }

func (ffiAPI) bitmapRGB(w, h int, rgb []byte) (uintptr, error) {
	b := mtmd.BitmapInit(uint32(w), uint32(h), rgb) //nolint:gosec // picture sides, small positive ints
	if b == 0 {
		return 0, fmt.Errorf("libmtmd refused a %dx%d bitmap", w, h)
	}
	return uintptr(b), nil
}

func (ffiAPI) bitmapVideo(proj uintptr, path string, fps float64) (uintptr, uintptr, error) {
	opt := mtmd.InitOptDefault()
	opt.VideoParams.FPSTarget = float32(fps)
	opt.VideoParams.TimestampIntervalMs = 0 // the embedding wants the pictures, not timestamps in the token stream
	w := mtmd.BitmapInitFromFile(mtmd.Context(proj), path, false, opt)
	if w.Bitmap == 0 {
		return 0, 0, fmt.Errorf("libmtmd could not read %s as a video (ffmpeg and ffprobe must be on PATH)", path)
	}
	return uintptr(w.Bitmap), uintptr(w.VideoContext), nil
}

func (ffiAPI) freeBitmap(bitmap uintptr) { mtmd.BitmapFree(mtmd.Bitmap(bitmap)) }

func (ffiAPI) freeVideo(video uintptr) { mtmd.VideoFree(mtmd.VideoContext(video)) }

func (ffiAPI) tokenizeMedia(proj uintptr, prompt string, bitmaps []uintptr) (uintptr, int, error) {
	chunks := mtmd.InputChunksInit()
	bms := make([]mtmd.Bitmap, len(bitmaps))
	for i, b := range bitmaps {
		bms[i] = mtmd.Bitmap(b)
	}
	// addSpecial adds the beginning-of-sequence token; parseSpecial reads
	// the template's control tokens as such
	text := mtmd.NewInputText(prompt, true, true)
	if ret := mtmd.Tokenize(mtmd.Context(proj), chunks, text, bms); ret != 0 {
		mtmd.InputChunksFree(chunks)
		switch ret {
		case 1:
			return 0, 0, fmt.Errorf("%d bitmaps for the markers in the prompt", len(bitmaps))
		case 2:
			return 0, 0, errors.New("media preprocessing failed")
		}
		return 0, 0, fmt.Errorf("mtmd_tokenize returned %d", ret)
	}
	n := 0
	for i := range mtmd.InputChunksSize(chunks) {
		n += int(mtmd.InputChunkGetNTokens(mtmd.InputChunksGet(chunks, i))) //nolint:gosec // a token count, a small int
	}
	return uintptr(chunks), n, nil
}

func (ffiAPI) freeChunks(chunks uintptr) { mtmd.InputChunksFree(mtmd.InputChunks(chunks)) }

func (ffiAPI) evalChunks(proj, ctx, chunks uintptr, batch int32) error {
	var n llama.Pos
	if ret := mtmd.HelperEvalChunks(mtmd.Context(proj), llama.Context(ctx), mtmd.InputChunks(chunks), 0, 0, batch, true, &n); ret != 0 {
		return fmt.Errorf("mtmd_helper_eval_chunks returned %d", ret)
	}
	return nil
}

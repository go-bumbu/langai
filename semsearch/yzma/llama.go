package yzma

import (
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/ebitengine/purego"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/utils"
)

// api is the llama.cpp surface Open drives; tests swap in a fake.
var api llamaAPI = ffiAPI{}

// ffiAPI implements llamaAPI over yzma's bindings. This is the only file that
// imports them: everything that decides something (load once, version lock,
// sizes, guards, cleanup) lives in backend.go, where a fake covers it, and
// what remains here are thin calls that need the native libraries to run.
type ffiAPI struct{}

func (ffiAPI) load(dir string, logger *slog.Logger) error {
	if err := llama.Load(dir); err != nil {
		return fmt.Errorf("load llama.cpp libraries from %s: %w (they need libgomp1 and libstdc++6 installed; libffi8 is required by the Go bindings)", dir, err)
	}
	llama.LogSet(logCallback(logger))
	if err := llama.Init(); err != nil {
		return fmt.Errorf("init llama.cpp backends from %s: %w", dir, err)
	}
	return nil
}

func (ffiAPI) version() string { return llama.Version() }

func (ffiAPI) loadModel(path string) (uintptr, error) {
	model, err := llama.ModelLoadFromFile(path, llama.ModelDefaultParams())
	if err != nil {
		return 0, err
	}
	if model == 0 {
		return 0, errors.New("llama.cpp returned no model (see its error log lines)")
	}
	return uintptr(model), nil
}

func (ffiAPI) freeModel(model uintptr) { _ = llama.ModelFree(llama.Model(model)) }

func (ffiAPI) nCtxTrain(model uintptr) int32 { return llama.ModelNCtxTrain(llama.Model(model)) }

func (ffiAPI) nEmbd(model uintptr) int32 { return llama.ModelNEmbd(llama.Model(model)) }

// poolingTypes maps this package's pooling names onto llama.cpp's; openLlama
// validates the name first, so a lookup never misses.
var poolingTypes = map[Pooling]llama.PoolingType{
	PoolingMean: llama.PoolingTypeMean,
	PoolingCLS:  llama.PoolingTypeCLS,
	PoolingLast: llama.PoolingTypeLast,
}

func (ffiAPI) newContext(model uintptr, nCtx int32, threads int, pooling Pooling) (uintptr, int, uintptr, error) {
	n := uint32(nCtx) //nolint:gosec // nCtx is a small positive model constant
	p := llama.ContextDefaultParams()
	p.NCtx = n
	p.NBatch = n
	p.NUbatch = n
	p.NSeqMax = 1
	p.PoolingType = poolingTypes[pooling]
	p.Embeddings = 1
	if threads > 0 {
		t := int32(threads) //nolint:gosec // threads is a small positive config value
		p.NThreads = t
		p.NThreadsBatch = t
	}
	ctx, err := llama.InitFromModel(llama.Model(model), p)
	if err != nil {
		return 0, 0, 0, err
	}
	// llama.cpp may adjust the sizes it was asked for: read them back
	limit := min(int(llama.NCtx(ctx)), int(llama.NBatch(ctx)), int(llama.NUBatch(ctx)))
	mem, _ := llama.GetMemory(ctx) // 0 when the model has no memory (BERT)
	return uintptr(ctx), limit, uintptr(mem), nil
}

func (ffiAPI) freeContext(ctx uintptr) { _ = llama.Free(llama.Context(ctx)) }

func (ffiAPI) tokenize(model uintptr, text string) []int32 {
	// addSpecial adds [CLS]/[SEP]; parseSpecial stays false so special-token
	// spellings inside untrusted text are ordinary text, not control tokens.
	toks := llama.Tokenize(llama.ModelGetVocab(llama.Model(model)), text, true, false)
	out := make([]int32, len(toks))
	for i, t := range toks {
		out[i] = int32(t)
	}
	return out
}

func (ffiAPI) memoryClear(mem uintptr) error { return llama.MemoryClear(llama.Memory(mem), true) }

func (ffiAPI) decode(ctx uintptr, tokens []int32) (int32, error) {
	// Mean pooling reads every token's output, so each one is marked as an
	// output, as llama.cpp's embedding example does: a batch that leaves the
	// flags unset (llama_batch_get_one) makes llama.cpp override them and log
	// a warning on every decode. BatchInit's arrays are C-owned and Add copies
	// into them, so the batch borrows no Go memory.
	n := int32(len(tokens)) //nolint:gosec // embed caps len(tokens) at the context limit, an int32 llama.cpp granted
	batch := llama.BatchInit(n, 0, 1)
	defer func() { _ = llama.BatchFree(batch) }()
	seq := []llama.SeqId{0}
	for i := int32(0); i < n; i++ {
		if err := batch.Add(llama.Token(tokens[i]), llama.Pos(i), seq, true); err != nil {
			return 0, fmt.Errorf("fill batch: %w", err)
		}
	}
	return llama.Decode(llama.Context(ctx), batch)
}

func (ffiAPI) embeddings(ctx uintptr, n int32) ([]float32, error) {
	return llama.GetEmbeddingsSeq(llama.Context(ctx), 0, n)
}

// logCallback forwards llama.cpp warnings and errors to logger. Everything
// else (progress, model metadata dumps) is dropped: llama.cpp is chatty, and
// the lines an operator needs are the ones explaining a failed load, which
// the bindings' own "failed to load model" error does not carry.
func logCallback(logger *slog.Logger) uintptr {
	if logger == nil {
		return llama.LogSilent()
	}
	return purego.NewCallback(func(level int32, text *byte, _ uintptr) uintptr {
		if text == nil {
			return 0
		}
		msg := strings.TrimSpace(utils.BytePtrToString(text))
		if msg == "" {
			return 0
		}
		switch llama.LogLevel(level) {
		case llama.LogLevelWarn:
			logger.Warn(msg, slog.String("component", "llama.cpp"))
		case llama.LogLevelError:
			logger.Error(msg, slog.String("component", "llama.cpp"))
		}
		return 0
	})
}

// defaultThreads keeps an embedder working in the background (a tagger, say)
// from monopolizing the host: at most four threads, fewer on small machines.
func defaultThreads() int {
	return min(4, max(1, int(llama.Threads())))
}

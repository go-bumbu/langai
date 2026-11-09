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

// api is the llama.cpp surface Open drives, and chatLib the one OpenCompleter
// drives; tests swap in fakes.
var (
	api     llamaAPI = ffiAPI{}
	chatLib chatAPI  = ffiAPI{}
)

// ffiAPI implements llamaAPI over yzma's bindings. This is the only file that
// imports them: everything that decides something (load once, version lock,
// sizes, guards, cleanup) lives in backend.go, where a fake covers it, and
// what remains here are thin calls that need the native libraries to run.
type ffiAPI struct{}

func (ffiAPI) load(dir string) error {
	if err := llama.Load(dir); err != nil {
		return fmt.Errorf("load llama.cpp libraries from %s: %w (they need libgomp1 and libstdc++6 installed; libffi8 is required by the Go bindings)", dir, err)
	}
	llamaLogCallback = logCallback()
	llama.LogSet(llamaLogCallback)
	if err := llama.Init(); err != nil {
		return fmt.Errorf("init llama.cpp backends from %s: %w", dir, err)
	}
	return nil
}

func (ffiAPI) version() string { return llama.Version() }

func (ffiAPI) devices() []device {
	n := llama.GGMLBackendDeviceCount()
	out := make([]device, 0, n)
	for i := range n {
		dev := llama.GGMLBackendDeviceGet(i)
		if dev == 0 {
			continue
		}
		t := llama.GGMLBackendDevType(dev)
		out = append(out, device{
			name:        llama.GGMLBackendDeviceName(dev),
			backend:     llama.GGMLBackendRegName(llama.GGMLBackendDeviceBackendReg(dev)),
			description: llama.GGMLBackendDeviceDescription(dev),
			gpu:         t == llama.GGMLBackendDeviceTypeGPU || t == llama.GGMLBackendDeviceTypeIGPU,
		})
	}
	return out
}

func (ffiAPI) loadModel(path string, gpuLayers int32) (uintptr, error) {
	p := llama.ModelDefaultParams()
	p.NGpuLayers = gpuLayers
	model, err := llama.ModelLoadFromFile(path, p)
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

func (ffiAPI) nEmbdOut(model uintptr) int32 { return llama.ModelNEmbdOut(llama.Model(model)) }

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

func (f ffiAPI) tokenize(model uintptr, text string) []int32 {
	// addSpecial adds [CLS]/[SEP]; parseSpecial stays false so special-token
	// spellings inside untrusted text are ordinary text, not control tokens.
	return f.tokenizeWith(model, text, true, false)
}

func (ffiAPI) tokenizeWith(model uintptr, text string, addSpecial, parseSpecial bool) []int32 {
	toks := llama.Tokenize(llama.ModelGetVocab(llama.Model(model)), text, addSpecial, parseSpecial)
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

func (ffiAPI) newChatContext(model uintptr, nCtx, batch int32, threads int) (uintptr, int, int, uintptr, error) {
	n, b := uint32(nCtx), uint32(batch) //nolint:gosec // both are small positive sizes the Completer derived from the model's window
	p := llama.ContextDefaultParams()
	p.NCtx = n
	p.NBatch = b
	p.NUbatch = b
	p.NSeqMax = 1
	if threads > 0 {
		t := int32(threads) //nolint:gosec // threads is a small positive config value
		p.NThreads = t
		p.NThreadsBatch = t
	}
	ctx, err := llama.InitFromModel(llama.Model(model), p)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	// llama.cpp may adjust the sizes it was asked for: read them back
	mem, _ := llama.GetMemory(ctx) // 0 when the model has no memory
	return uintptr(ctx), int(llama.NCtx(ctx)), min(int(llama.NBatch(ctx)), int(llama.NUBatch(ctx))), uintptr(mem), nil
}

func (ffiAPI) applyTemplate(template, system, user string) (string, error) {
	msgs := []llama.ChatMessage{llama.NewChatMessage("system", system), llama.NewChatMessage("user", user)}
	// a template adds a few dozen bytes of role markup around the messages; if
	// llama.cpp wants more room than this, it reports how much
	buf := make([]byte, 2*(len(system)+len(user))+512)
	n := llama.ChatApplyTemplate(template, msgs, true, buf)
	if int(n) > len(buf) {
		buf = make([]byte, n)
		n = llama.ChatApplyTemplate(template, msgs, true, buf)
	}
	if n < 0 {
		return "", fmt.Errorf("llama.cpp cannot apply the template (code %d)", n)
	}
	return string(buf[:n]), nil
}

func (ffiAPI) newSampler() uintptr {
	chain := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(chain, llama.SamplerInitGreedy())
	return uintptr(chain)
}

func (ffiAPI) freeSampler(sampler uintptr) { llama.SamplerFree(llama.Sampler(sampler)) }

func (ffiAPI) decodeAt(ctx uintptr, tokens []int32, pos int32) (int32, error) {
	// BatchInit's arrays are C-owned and Add copies into them, so the batch
	// borrows no Go memory. Only the last token's logits are read.
	n := int32(len(tokens)) //nolint:gosec // Completer feeds at most one batch of tokens, a size llama.cpp granted
	batch := llama.BatchInit(n, 0, 1)
	defer func() { _ = llama.BatchFree(batch) }()
	seq := []llama.SeqId{0}
	for i := int32(0); i < n; i++ {
		if err := batch.Add(llama.Token(tokens[i]), llama.Pos(pos+i), seq, i == n-1); err != nil {
			return 0, fmt.Errorf("fill batch: %w", err)
		}
	}
	return llama.Decode(llama.Context(ctx), batch)
}

func (ffiAPI) sample(sampler, ctx uintptr) int32 {
	return int32(llama.SamplerSample(llama.Sampler(sampler), llama.Context(ctx), -1))
}

func (ffiAPI) isEOG(model uintptr, token int32) bool {
	return llama.VocabIsEOG(llama.ModelGetVocab(llama.Model(model)), llama.Token(token))
}

func (ffiAPI) tokenText(model uintptr, token int32) []byte {
	vocab := llama.ModelGetVocab(llama.Model(model))
	// special is false: a control token a model writes is markup, not reply
	buf := make([]byte, 64)
	n := llama.TokenToPiece(vocab, llama.Token(token), buf, 0, false)
	if n < 0 { // the piece needs more room than this, and says how much
		buf = make([]byte, -n)
		n = llama.TokenToPiece(vocab, llama.Token(token), buf, 0, false)
	}
	if n <= 0 {
		return nil
	}
	return buf[:n]
}

// llamaLogCallback is the callback load installed for llama.cpp's log, which
// libmtmd gets too when it is loaded later.
var llamaLogCallback uintptr

// expectedWarning reports whether a llama.cpp warning describes what this
// package does on purpose, so it is dropped rather than logged once per
// input. The Vision embedder feeds each input through libmtmd's helper,
// which marks only the last token as an output, and the embedding context
// then marks every token itself and says so; llama.cpp's own server logs
// the same line on every embedding request.
func expectedWarning(msg string) bool {
	return strings.Contains(msg, "embeddings required but some input tokens were not marked as outputs")
}

// logCallback is the one log callback the libraries get for the life of the
// process. It hands every line to forwardLog, which looks up where lines go
// at that moment (llamaLog), so the destination can change after the
// libraries loaded; while there is none the line is not even copied.
func logCallback() uintptr {
	return purego.NewCallback(func(level int32, text *byte, _ uintptr) uintptr {
		if text != nil && llamaLog.Load() != nil {
			forwardLog(level, utils.BytePtrToString(text))
		}
		return 0
	})
}

// forwardLog forwards a llama.cpp warning or error to the logger llamaLog
// points at. Everything else (progress, model metadata dumps) is dropped:
// llama.cpp is chatty, and the lines an operator needs are the ones
// explaining a failed load, which the bindings' own "failed to load model"
// error does not carry.
func forwardLog(level int32, text string) {
	logger := llamaLog.Load()
	if logger == nil {
		return
	}
	msg := strings.TrimSpace(text)
	if msg == "" || expectedWarning(msg) {
		return
	}
	switch llama.LogLevel(level) {
	case llama.LogLevelWarn:
		logger.Warn(msg, slog.String("component", "llama.cpp"))
	case llama.LogLevelError:
		logger.Error(msg, slog.String("component", "llama.cpp"))
	}
}

// defaultThreads keeps an embedder working in the background (a tagger, say)
// from monopolizing the host: at most four threads, fewer on small machines.
func defaultThreads() int {
	return min(4, max(1, int(llama.Threads())))
}

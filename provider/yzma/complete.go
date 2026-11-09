package yzma

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/internal/fetch"
)

// DefaultCompleterModel is the logical identity of the chat model NewCompleter
// runs unless Config.Model selects another: Qwen/Qwen3-4B-Instruct-2507,
// Alibaba's 4-billion-parameter instruction-tuned model under Apache-2.0. It
// answers without a thinking phase, speaks the many languages Qwen3 was
// trained on, and runs on a CPU. The GGUF behind it is defaultCompleterAsset in
// fetch.go.
const DefaultCompleterModel = "Qwen/Qwen3-4B-Instruct-2507"

const (
	// defaultContextTokens is the context a CompleterSpec asks for when it
	// sets none. A tagging.Generator sends a long text in windows of about
	// 6 KB, roughly 1,500 to 2,500 tokens of prose depending on the language,
	// and such a window, the system prompt and a short reply fit in it with
	// room to spare, where 2048 tokens would not always hold them. The cost is
	// memory: the KV cache roughly doubles, a few hundred MB for the default
	// model. A window whose prompt does not fit the context fails the
	// Generate call.
	defaultContextTokens = 4096
	// completerBatch is the most tokens one decode takes while the prompt is
	// fed in. llama.cpp aborts the process on a larger batch, so a longer
	// prompt goes in as several.
	completerBatch = 512
	// defaultMaxNewTokens caps a reply when a CompleterSpec sets no cap: a
	// dozen short tags are far below it.
	defaultMaxNewTokens = 128
	// systemMarker and userMarker stand in for the two messages while llama.cpp
	// lays the chat template out; promptTokens then puts the messages where
	// they stood. Neither can occur in a template.
	systemMarker = "@@system@@"
	userMarker   = "@@user@@"
)

// CompleterSpec describes one GGUF instruction-tuned chat model: where its
// file is (or comes from) and how it must be run. LookupCompleterSpec returns
// the pinned ones NewCompleter selects from.
type CompleterSpec struct {
	// Name is the logical identity of the model.
	Name string
	// File is the GGUF path. When it is missing and URL is set, it is
	// downloaded there first.
	File string
	// URL is where File is downloaded from when it is absent or, with a Size
	// set, not that long (a truncated file). SHA256 is required
	// with it, and Size, when non-zero, is checked too: OpenCompleter never
	// downloads unverified bytes.
	URL    string
	SHA256 string
	Size   int64
	// Template names the llama.cpp chat template that lays messages out the
	// way the model was trained on them ("chatml", "llama3", ...; see
	// llama.ChatBuiltinTemplates). Required: a model given the wrong template
	// still answers, only worse, so it is never guessed.
	Template string
	// ContextTokens is the context window, the prompt and the reply together,
	// capped at the model's training window; 0 means 4096 (it was 2048). The
	// default costs a few hundred MB more for the default model; set 2048 to
	// keep the old memory if you never tag long texts. A tagging.Generator
	// sends windows of about 6 KB, up to 2,500 tokens, so with a context under
	// 4096 keep GeneratorConfig.MaxBytes at or under ChunkBytes, or a window
	// that does not fit fails the whole Generate call.
	ContextTokens int
	// MaxNewTokens caps one reply; 0 means 128.
	MaxNewTokens int
}

// validate reports a spec OpenCompleter cannot run.
func (s CompleterSpec) validate() error {
	switch {
	case s.Name == "":
		return errors.New("yzma: CompleterSpec.Name is required")
	case s.File == "":
		return fmt.Errorf("yzma: model %q: File is required", s.Name)
	case s.URL != "" && s.SHA256 == "":
		return fmt.Errorf("yzma: model %q: URL without SHA256; pin the digest (make semeval logs it for an unpinned entry)", s.Name)
	case s.Template == "":
		return fmt.Errorf("yzma: model %q: Template is required", s.Name)
	case s.ContextTokens < 0 || s.MaxNewTokens < 0:
		return fmt.Errorf("yzma: model %q: ContextTokens and MaxNewTokens must not be negative", s.Name)
	}
	return nil
}

// NewCompleter prepares the libraries and the chat model cfg.Model selects
// (downloading whatever is missing, see fetch.go) and loads them. cfg is New's
// Config, read for the chat model: Model names one of the pinned chat models
// (see LookupCompleterSpec), "" meaning DefaultCompleterModel, so a Config
// that leaves it empty can build the Embedder and the Completer both. ctx
// bounds the downloads only; the returned Completer outlives it. It is
// OpenCompleter with LookupCompleterSpec(cfg.Model, cfg.ModelsDir).
func NewCompleter(ctx context.Context, cfg Config) (*Completer, error) {
	spec, err := LookupCompleterSpec(cfg.Model, cfg.ModelsDir)
	if err != nil {
		return nil, err
	}
	return OpenCompleter(ctx, spec, cfg.runtime())
}

// OpenCompleter prepares the libraries and the model described by spec
// (downloading whatever is missing, see fetch.go) and loads them. The libraries
// load once per process, so a Completer opens next to an Embedder. ctx bounds
// the downloads only, and gets a deadline of an hour when it has none of its
// own; the returned Completer outlives it and is released with Close. A
// download that was cut short is picked up where it stopped by the next
// OpenCompleter.
func OpenCompleter(ctx context.Context, spec CompleterSpec, rt Runtime) (*Completer, error) {
	if err := spec.validate(); err != nil {
		return nil, err
	}
	ctx, cancel := fetch.WithDefaultTimeout(ctx, completerDownloadTimeout)
	defer cancel()
	libDir, logger, err := rt.prepare(ctx, chatLib, libs)
	if err != nil {
		return nil, err
	}
	asModel := ModelSpec{Name: spec.Name, File: spec.File, URL: spec.URL, SHA256: spec.SHA256, Size: spec.Size}
	if err := ensureModelFile(ctx, asModel, logger); err != nil {
		return nil, err
	}
	c, err := openCompleter(chatLib, libs, libDir, spec.File, spec, rt.Threads, rt.Accel, rt.GPULayers, rt.Logger)
	if err != nil {
		return nil, err
	}
	logger.Info("llama.cpp loaded", logComponent, slog.String("dir", libDir), slog.String("version", libs.version), slog.Int("threads", rt.Threads), slog.String("accel", c.accel), slog.String("model", spec.Name))
	return c, nil
}

// chatAPI is the slice of llama.cpp the Completer drives, with handles as
// plain integers so that only llama.go names the bindings' types. ffiAPI
// (llama.go) is the real implementation; complete_test.go has a fake.
type chatAPI interface {
	runtimeAPI
	// loadModel loads a GGUF file with gpuLayers layers on the GPU (0: all
	// on the CPU); a zero handle is an error.
	loadModel(path string, gpuLayers int32) (model uintptr, err error)
	freeModel(model uintptr)
	nCtxTrain(model uintptr) int32
	// newChatContext creates a generation context of nCtx tokens that takes
	// batches of up to batch tokens. nCtxGot and batchGot are the sizes
	// llama.cpp actually granted, which may be smaller; mem is 0 when the
	// model has no KV cache.
	newChatContext(model uintptr, nCtx, batch int32, threads int) (ctx uintptr, nCtxGot, batchGot int, mem uintptr, err error)
	freeContext(ctx uintptr)
	// applyTemplate lays one system and one user message out with the named
	// chat template, ending where the assistant's turn begins.
	applyTemplate(template, system, user string) (string, error)
	// tokenizeWith converts text to tokens. addSpecial adds the model's
	// beginning-of-sequence token, and parseSpecial reads special-token
	// spellings in text as the control tokens they spell.
	tokenizeWith(model uintptr, text string, addSpecial, parseSpecial bool) []int32
	memoryClear(mem uintptr) error
	// newSampler returns a greedy sampler, freed with freeSampler.
	newSampler() uintptr
	freeSampler(sampler uintptr)
	// decodeAt runs tokens through the model as positions pos.., keeping the
	// logits of the last one; ret is llama_decode's return value (0 = ok).
	decodeAt(ctx uintptr, tokens []int32, pos int32) (ret int32, err error)
	// setAbort makes the decodes of ctx ask stop as they compute and give up
	// when it says so; nil takes the question away again.
	setAbort(ctx uintptr, stop func() bool)
	// sample draws the next token from the logits of the last decode.
	sample(sampler, ctx uintptr) int32
	// isEOG reports whether token ends a reply.
	isEOG(model uintptr, token int32) bool
	// tokenText is the bytes token spells, without control-token markup. A
	// character can be spread over several tokens, so they are not always valid
	// UTF-8 by themselves.
	tokenText(model uintptr, token int32) []byte
}

// Completer implements langai.Completer over a loaded chat model: it answers
// a system message and a user message with the text the model writes.
type Completer struct {
	// a llama.cpp context is not safe for concurrent use, so calls are
	// serialized; generating is CPU-bound anyway
	mu       sync.Mutex
	api      chatAPI
	model    uintptr
	ctx      uintptr
	mem      uintptr // 0 for models without a KV cache
	template string
	nCtx     int // tokens the context holds, the prompt and the reply together
	batch    int // most tokens one decode takes
	maxNew   int
	accel    string // what runs the model, for the load log line
	closed   bool   // set by Close and read by Complete, both under mu
}

// openCompleter loads the libraries (once, through rt) and the model, on the
// GPU the accelerator resolves to (see resolveAccel) or the CPU, builds a
// generation context for it and tries the chat template once, so that a name
// llama.cpp does not know fails here and not at the first call. Whatever it
// created is freed again when a later step fails.
func openCompleter(api chatAPI, rt *runtimeLoader, libDir, modelPath string, spec CompleterSpec, threads int, accel Accel, gpuLayers int, logger *slog.Logger) (*Completer, error) {
	if err := rt.load(api, libDir, logger); err != nil {
		return nil, err
	}
	layers, used, err := resolveAccel(accel, gpuLayers, rt.devices)
	if err != nil {
		return nil, err
	}
	model, err := api.loadModel(modelPath, layers)
	if err != nil {
		return nil, fmt.Errorf("load model %s: %w", modelPath, err)
	}
	nCtxTrain := api.nCtxTrain(model)
	if nCtxTrain <= 0 {
		api.freeModel(model)
		return nil, fmt.Errorf("load model %s: no training context length in the GGUF metadata", modelPath)
	}
	want := cmp.Or(spec.ContextTokens, defaultContextTokens)
	nCtx := int32(min(int(nCtxTrain), want)) //nolint:gosec // no larger than nCtxTrain, an int32
	batch := min(nCtx, completerBatch)
	ctx, nCtxGot, batchGot, mem, err := api.newChatContext(model, nCtx, batch, threads)
	if err != nil {
		api.freeModel(model)
		return nil, fmt.Errorf("create llama.cpp context: %w", err)
	}
	c := &Completer{
		api:      api,
		model:    model,
		ctx:      ctx,
		mem:      mem,
		template: spec.Template,
		nCtx:     min(nCtxGot, int(nCtx)),
		batch:    min(batchGot, int(batch)),
		maxNew:   cmp.Or(spec.MaxNewTokens, defaultMaxNewTokens),
		accel:    used,
	}
	// a context or a batch of no tokens would hold no prompt, and feed would
	// never advance
	if c.nCtx <= 0 || c.batch <= 0 {
		c.release()
		return nil, fmt.Errorf("llama.cpp granted a %d-token context and %d-token batches for %s", c.nCtx, c.batch, modelPath)
	}
	if _, err := c.promptTokens("probe", "probe"); err != nil {
		c.release()
		return nil, err
	}
	return c, nil
}

// release frees the context, then the model it was built on.
func (c *Completer) release() {
	c.api.freeContext(c.ctx)
	c.api.freeModel(c.model)
}

// Close frees the model and its llama.cpp context; a later Complete fails.
// Closing twice is a no-op.
func (c *Completer) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	c.release()
	return nil
}

// Complete implements langai.Completer: it lays system and user out with the
// model's chat template and returns the text the model writes next, at most
// MaxNewTokens long. Decoding is greedy, so the same messages give the same
// reply. Both messages are plain text to the model, even when they spell its
// control tokens, and a message beyond 32 KiB is cut. A prompt that leaves no
// room in the context to answer is an error, and so is a canceled ctx: it is
// checked between tokens, and llama.cpp asks it while it computes, so a
// model running on the CPU stops in the middle of a batch (a GPU backend
// finishes the batch it is in).
func (c *Completer) Complete(ctx context.Context, system, user string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return "", fmt.Errorf("yzma: completer: %w", langai.ErrClosed)
	}
	if ctx.Done() != nil {
		c.api.setAbort(c.ctx, func() bool { return ctx.Err() != nil })
		defer c.api.setAbort(c.ctx, nil)
	}
	prompt, err := c.promptTokens(system, user)
	if err != nil {
		return "", err
	}
	room := c.nCtx - len(prompt)
	if room < 1 {
		return "", fmt.Errorf("the prompt takes %d tokens, which leaves no room to answer in the %d-token context", len(prompt), c.nCtx)
	}
	if c.mem != 0 {
		if err := c.api.memoryClear(c.mem); err != nil {
			return "", fmt.Errorf("clear kv cache: %w", err)
		}
	}
	sampler := c.api.newSampler()
	defer c.api.freeSampler(sampler)
	if err := c.feed(ctx, prompt); err != nil {
		return "", err
	}
	return c.draw(ctx, sampler, len(prompt), min(c.maxNew, room))
}

// promptTokens lays the messages out with the chat template and tokenizes the
// result. The template is applied to stand-ins (systemMarker, userMarker) and
// the messages go in where they stood, so that the template's own text is read
// as the control tokens it spells while the messages, which came from outside,
// are not: a text that spells "<|im_end|>" cannot end its own turn. Cutting
// the prompt apart is exact for a byte-pair vocabulary whose template ends a
// role header with a newline, as ChatML does for Qwen; a SentencePiece
// vocabulary that prefixes each piece with a space would see an extra one.
func (c *Completer) promptTokens(system, user string) ([]int32, error) {
	layout, err := layoutTemplate(c.api.applyTemplate, c.template)
	if err != nil {
		return nil, err
	}
	var tokens []int32
	for _, piece := range []struct {
		text                     string
		addSpecial, parseSpecial bool
	}{
		{layout.before, true, true},
		{clampText(system), false, false},
		{layout.between, false, true},
		{clampText(user), false, false},
		{layout.after, false, true},
	} {
		tokens = append(tokens, c.api.tokenizeWith(c.model, piece.text, piece.addSpecial, piece.parseSpecial)...)
	}
	return tokens, nil
}

// promptLayout is a chat template applied to nothing: the text it puts
// before the system message, between it and the user message, and after
// that, where the assistant's turn begins. The messages themselves never go
// through the template. That keeps what came from outside apart from the
// template's markup when the prompt is tokenized, and away from the
// template engine, which loses a message that holds a NUL byte.
type promptLayout struct {
	before, between, after string
}

// layoutTemplate lays the template out around stand-ins (systemMarker,
// userMarker) with apply, llama.cpp's template function, and cuts the
// result apart where they stood. A template that does not place a system
// message followed by a user message exactly once is an error.
func layoutTemplate(apply func(template, system, user string) (string, error), template string) (promptLayout, error) {
	rendered, err := apply(template, systemMarker, userMarker)
	if err != nil {
		return promptLayout{}, fmt.Errorf("apply chat template %q: %w", template, err)
	}
	before, rest, hasSystem := strings.Cut(rendered, systemMarker)
	between, after, hasUser := strings.Cut(rest, userMarker)
	if !hasSystem || !hasUser || strings.Contains(after, systemMarker) || strings.Contains(after, userMarker) {
		return promptLayout{}, fmt.Errorf("chat template %q does not lay out a system message followed by a user message exactly once: %q", template, rendered)
	}
	return promptLayout{before: before, between: between, after: after}, nil
}

// feed runs the prompt through the model in batches, each where the last
// ended.
func (c *Completer) feed(ctx context.Context, prompt []int32) error {
	for start := 0; start < len(prompt); start += c.batch {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := c.decode(ctx, prompt[start:min(start+c.batch, len(prompt))], start); err != nil {
			return err
		}
	}
	return nil
}

// draw writes the reply after a prompt of promptLen tokens: it draws up to
// limit tokens from the sampler, stopping early at an end-of-generation token.
// The bytes of the tokens are joined and read as UTF-8 once, since a character
// can be spread over several of them.
func (c *Completer) draw(ctx context.Context, sampler uintptr, promptLen, limit int) (string, error) {
	var reply []byte
	for n := range limit {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		tok := c.api.sample(sampler, c.ctx)
		if c.api.isEOG(c.model, tok) {
			break
		}
		reply = append(reply, c.api.tokenText(c.model, tok)...)
		if n+1 == limit {
			break // nothing follows the last token, so it is not decoded
		}
		if err := c.decode(ctx, []int32{tok}, promptLen+n); err != nil {
			return "", err
		}
	}
	return strings.ToValidUTF8(string(reply), "�"), nil
}

// decode runs tokens through the model as positions pos... A decode that
// failed because ctx ended (see setAbort) reports ctx's error.
func (c *Completer) decode(ctx context.Context, tokens []int32, pos int) error {
	// pos stays below the context size, an int32 llama.cpp granted
	ret, err := c.api.decodeAt(c.ctx, tokens, int32(pos)) //nolint:gosec // see above
	if err == nil && ret == 0 {
		return nil
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	if err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	return fmt.Errorf("decode returned %d", ret)
}

package yzma

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/go-bumbu/langai"
)

// the Completer must be what a tagging.Generator asks for
var _ langai.Completer = (*Completer)(nil)

// Handles and tokens the fake hands out.
const (
	fakeSampler uintptr = 0x3000
	fakeMem     uintptr = 0x4000
	// bos is the token the fake adds where llama.cpp would add special tokens,
	// and endOfReply the one it reports as end of generation. Every other
	// token is the rune it was made from, so a test can read a prompt back.
	bos        int32 = 1
	endOfReply int32 = 1 << 20

	testChatModelPath = "/opt/models/chat.gguf"
)

// callLog records the calls a fake saw.
type callLog struct{ calls []fakeCall }

func (l *callLog) record(name string, args ...any) {
	l.calls = append(l.calls, fakeCall{name: name, args: args})
}

func (l *callLog) names() []string {
	out := make([]string, len(l.calls))
	for i, c := range l.calls {
		out[i] = c.name
	}
	return out
}

func (l *callLog) count(name string) int {
	n := 0
	for _, c := range l.calls {
		if c.name == name {
			n++
		}
	}
	return n
}

// only returns the arguments of the single call to the named method and fails
// the test unless there was exactly one.
func (l *callLog) only(t *testing.T, name string) []any {
	t.Helper()
	if n := l.count(name); n != 1 {
		t.Fatalf("%s was called %d times, want once; calls: %v", name, n, l.names())
	}
	for _, c := range l.calls {
		if c.name == name {
			return c.args
		}
	}
	return nil
}

// decodeCall is one batch the fake was asked to decode.
type decodeCall struct {
	tokens []int32
	pos    int32
}

// tokenizeCall is one piece of the prompt the fake was asked to tokenize.
type tokenizeCall struct {
	text                     string
	addSpecial, parseSpecial bool
}

// fakeChat is a chatAPI without native code. Its template lays a prompt out as
// "<chat>" system "<mid>" user "<end>", its tokens are the runes of the text,
// and its sampler plays a script, so Completer runs its real code against it.
type fakeChat struct {
	callLog
	loadErr       error
	loadModelErr  error
	trainCtx      int32
	grantCtx      int  // what newChatContext grants; 0 = what it was asked for
	grantBatch    int  // likewise for the batch size
	grantNothing  bool // newChatContext grants a context of no tokens
	mem           uintptr
	newContextErr error
	templateErr   error
	render        func(system, user string) string
	script        []int32          // the tokens the sampler returns, then endOfReply
	pieces        map[int32][]byte // the bytes of a token; its rune when absent
	decodeRet     int32
	decodeErr     error
	failDecode    int       // fail the nth decode (1-based); 0 = never
	onSample      func(int) // called with the number of the sample about to be drawn

	decodes   []decodeCall
	tokenized []tokenizeCall
	samples   int
}

var _ chatAPI = (*fakeChat)(nil)

func newFakeChat() *fakeChat { return &fakeChat{trainCtx: 4096, mem: fakeMem} }

func (f *fakeChat) load(dir string, _ *slog.Logger) error {
	f.record("load", dir)
	return f.loadErr
}

func (f *fakeChat) version() string { return libVersion }

func (f *fakeChat) loadModel(path string) (uintptr, error) {
	f.record("loadModel", path)
	if f.loadModelErr != nil {
		return 0, f.loadModelErr
	}
	return fakeModel, nil
}

func (f *fakeChat) freeModel(model uintptr) { f.record("freeModel", model) }

func (f *fakeChat) nCtxTrain(model uintptr) int32 {
	f.record("nCtxTrain", model)
	return f.trainCtx
}

func (f *fakeChat) newChatContext(model uintptr, nCtx, batch int32, threads int) (uintptr, int, int, uintptr, error) {
	f.record("newChatContext", model, nCtx, batch, threads)
	if f.newContextErr != nil {
		return 0, 0, 0, 0, f.newContextErr
	}
	if f.grantNothing {
		return fakeCtx, 0, 0, f.mem, nil
	}
	gotCtx, gotBatch := int(nCtx), int(batch)
	if f.grantCtx != 0 {
		gotCtx = f.grantCtx
	}
	if f.grantBatch != 0 {
		gotBatch = f.grantBatch
	}
	return fakeCtx, gotCtx, gotBatch, f.mem, nil
}

func (f *fakeChat) freeContext(ctx uintptr) { f.record("freeContext", ctx) }

func (f *fakeChat) applyTemplate(template, system, user string) (string, error) {
	f.record("applyTemplate", template, system, user)
	if f.templateErr != nil {
		return "", f.templateErr
	}
	if f.render != nil {
		return f.render(system, user), nil
	}
	return "<chat>" + system + "<mid>" + user + "<end>", nil
}

func (f *fakeChat) tokenizeWith(model uintptr, text string, addSpecial, parseSpecial bool) []int32 {
	f.record("tokenizeWith", model, text, addSpecial, parseSpecial)
	f.tokenized = append(f.tokenized, tokenizeCall{text, addSpecial, parseSpecial})
	var out []int32
	if addSpecial {
		out = append(out, bos)
	}
	for _, r := range text {
		out = append(out, int32(r))
	}
	return out
}

func (f *fakeChat) memoryClear(mem uintptr) error {
	f.record("memoryClear", mem)
	return nil
}

func (f *fakeChat) newSampler() uintptr {
	f.record("newSampler")
	return fakeSampler
}

func (f *fakeChat) freeSampler(sampler uintptr) { f.record("freeSampler", sampler) }

func (f *fakeChat) decodeAt(ctx uintptr, tokens []int32, pos int32) (int32, error) {
	f.record("decodeAt", ctx, len(tokens), pos)
	f.decodes = append(f.decodes, decodeCall{slices.Clone(tokens), pos})
	if f.failDecode != 0 && len(f.decodes) == f.failDecode {
		return 0, errors.New("decode failed")
	}
	return f.decodeRet, f.decodeErr
}

func (f *fakeChat) sample(sampler, ctx uintptr) int32 {
	f.record("sample", sampler, ctx)
	f.samples++
	if f.onSample != nil {
		f.onSample(f.samples)
	}
	if f.samples > len(f.script) {
		return endOfReply
	}
	return f.script[f.samples-1]
}

func (f *fakeChat) isEOG(model uintptr, token int32) bool {
	f.record("isEOG", model, token)
	return token == endOfReply
}

func (f *fakeChat) tokenText(model uintptr, token int32) []byte {
	f.record("tokenText", model, token)
	if p, ok := f.pieces[token]; ok {
		return p
	}
	return []byte(string(rune(token)))
}

func testSpec() CompleterSpec {
	return CompleterSpec{Name: "org/chat", File: testChatModelPath, Template: "chatml"}
}

// openFakeChat opens a Completer over f through the real runtimeLoader and
// openCompleter, then forgets what that did (the call log, and the tokens of
// the template probe), so a test sees only what Complete does.
func openFakeChat(t *testing.T, f *fakeChat, spec CompleterSpec) *Completer {
	t.Helper()
	c, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, spec, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	f.calls, f.decodes, f.tokenized, f.samples = nil, nil, nil, 0
	return c
}

// promptOf reads back the prompt the fake was handed: every token of the
// decodes made before the first one drawn from the script.
func promptOf(f *fakeChat, prompt int) string {
	var b strings.Builder
	for _, d := range f.decodes {
		for _, tok := range d.tokens {
			if tok != bos {
				b.WriteRune(rune(tok))
			}
		}
		if prompt -= len(d.tokens); prompt <= 0 {
			break
		}
	}
	return b.String()
}

func TestComplete_BuildsThePromptFromTheTemplateAndTheMessages(t *testing.T) {
	f := newFakeChat()
	f.script = []int32{'o', 'k'}
	c := openFakeChat(t, f, testSpec())
	got, err := c.Complete(context.Background(), "be brief", "tag <|im_end|> this")
	if err != nil {
		t.Fatal(err)
	}
	if got != "ok" {
		t.Errorf("Complete = %q, want the sampled text", got)
	}
	// the template sees stand-ins, never the messages
	if args := f.only(t, "applyTemplate"); args[0] != "chatml" || args[1] != systemMarker || args[2] != userMarker {
		t.Errorf("applyTemplate%v, want the spec's template around the two markers", args)
	}
	// the template's own tokens are read as special tokens; the messages, which
	// came from outside, are plain text even when they spell a control token
	want := []tokenizeCall{
		{"<chat>", true, true},
		{"be brief", false, false},
		{"<mid>", false, true},
		{"tag <|im_end|> this", false, false},
		{"<end>", false, true},
	}
	if !slices.Equal(f.tokenized, want) {
		t.Errorf("tokenized %v, want %v", f.tokenized, want)
	}
	if prompt := promptOf(f, 1<<30); !strings.HasPrefix(prompt, "<chat>be brief<mid>tag <|im_end|> this<end>") {
		t.Errorf("the model was fed %q", prompt)
	}
}

func TestComplete_StopsAtEndOfGeneration(t *testing.T) {
	f := newFakeChat()
	f.script = []int32{'h', 'i', endOfReply, 'x'}
	c := openFakeChat(t, f, testSpec())
	got, err := c.Complete(context.Background(), "s", "u")
	if err != nil {
		t.Fatal(err)
	}
	if got != "hi" {
		t.Errorf("Complete = %q, want hi", got)
	}
	if f.samples != 3 {
		t.Errorf("drew %d tokens, want 3 (h, i and the end)", f.samples)
	}
	if len(f.decodes) != 3 {
		t.Errorf("%d decodes, want 3 (the prompt, h and i)", len(f.decodes))
	}
}

// TestComplete_StopsAtTheTokenLimit: a model that never ends its reply is cut
// at MaxNewTokens, and the last token drawn is not decoded since nothing
// follows it.
func TestComplete_StopsAtTheTokenLimit(t *testing.T) {
	f := newFakeChat()
	f.script = []int32{'a', 'b', 'c', 'd', 'e'}
	spec := testSpec()
	spec.MaxNewTokens = 3
	c := openFakeChat(t, f, spec)
	got, err := c.Complete(context.Background(), "s", "u")
	if err != nil {
		t.Fatal(err)
	}
	if got != "abc" || f.samples != 3 {
		t.Errorf("Complete = %q after %d draws, want abc after 3", got, f.samples)
	}
	if len(f.decodes) != 3 {
		t.Errorf("%d decodes, want 3 (the prompt, a and b)", len(f.decodes))
	}
}

func TestComplete_DecodesAtGrowingPositions(t *testing.T) {
	f := newFakeChat()
	f.script = []int32{'h', 'i'}
	c := openFakeChat(t, f, testSpec())
	if _, err := c.Complete(context.Background(), "s", "u"); err != nil {
		t.Fatal(err)
	}
	prompt := int32(len(f.decodes[0].tokens))
	if f.decodes[0].pos != 0 || f.decodes[1].pos != prompt || f.decodes[2].pos != prompt+1 {
		t.Errorf("positions %d, %d, %d, want 0, %d, %d", f.decodes[0].pos, f.decodes[1].pos, f.decodes[2].pos, prompt, prompt+1)
	}
}

// TestComplete_FeedsALongPromptInBatches: llama.cpp aborts the process on a
// batch larger than the context's, so a prompt longer than one is cut into
// batches, each decoded where the last ended.
func TestComplete_FeedsALongPromptInBatches(t *testing.T) {
	f := newFakeChat()
	f.grantBatch = 4
	c := openFakeChat(t, f, testSpec())
	if _, err := c.Complete(context.Background(), "be brief", "tag this"); err != nil {
		t.Fatal(err)
	}
	var pos int32
	for i, d := range f.decodes {
		if len(d.tokens) == 0 || len(d.tokens) > 4 || d.pos != pos {
			t.Errorf("decode %d: %d tokens at %d, want 1..4 tokens at %d", i, len(d.tokens), d.pos, pos)
		}
		pos += int32(len(d.tokens))
	}
	if len(f.decodes) < 3 {
		t.Fatalf("%d decodes: the prompt was too short to need batches", len(f.decodes))
	}
	if got := promptOf(f, int(pos)); got != "<chat>be brief<mid>tag this<end>" {
		t.Errorf("the model was fed %q", got)
	}
}

func TestComplete_RefusesAPromptThatLeavesNoRoom(t *testing.T) {
	f := newFakeChat()
	f.grantCtx = 10
	c := openFakeChat(t, f, testSpec())
	_, err := c.Complete(context.Background(), "be brief", "tag this")
	if err == nil || !strings.Contains(err.Error(), "10-token") {
		t.Fatalf("err = %v, want a refusal naming the 10-token context", err)
	}
	if f.count("decodeAt") != 0 || f.count("newSampler") != 0 {
		t.Errorf("a prompt that does not fit reached llama.cpp: %v", f.names())
	}
}

// TestComplete_ClampsTheReplyToTheRoomLeft: the context holds the prompt and
// the reply together, so a reply may not outgrow what the prompt left.
func TestComplete_ClampsTheReplyToTheRoomLeft(t *testing.T) {
	probe := newFakeChat()
	if _, err := openFakeChat(t, probe, testSpec()).Complete(context.Background(), "s", "u"); err != nil {
		t.Fatal(err)
	}
	prompt := len(probe.decodes[0].tokens)

	f := newFakeChat()
	f.script = []int32{'a', 'b', 'c', 'd', 'e'}
	spec := testSpec()
	spec.ContextTokens = prompt + 2
	c := openFakeChat(t, f, spec)
	got, err := c.Complete(context.Background(), "s", "u")
	if err != nil {
		t.Fatal(err)
	}
	if got != "ab" {
		t.Errorf("Complete = %q, want the 2 tokens that fit", got)
	}
}

// TestComplete_ClearsTheMemoryAndFreesTheSamplerEveryCall: calls are
// independent, so each starts from an empty KV cache with a sampler of its own.
func TestComplete_ClearsTheMemoryAndFreesTheSamplerEveryCall(t *testing.T) {
	f := newFakeChat()
	c := openFakeChat(t, f, testSpec())
	for range 2 {
		if _, err := c.Complete(context.Background(), "s", "u"); err != nil {
			t.Fatal(err)
		}
	}
	if f.count("memoryClear") != 2 || f.count("newSampler") != 2 || f.count("freeSampler") != 2 {
		t.Errorf("memoryClear %d, newSampler %d, freeSampler %d times, want twice each", f.count("memoryClear"), f.count("newSampler"), f.count("freeSampler"))
	}
	for _, call := range f.calls {
		if call.name == "memoryClear" && call.args[0] != fakeMem {
			t.Errorf("memoryClear(%v), want the context's memory %v", call.args[0], fakeMem)
		}
	}
}

func TestComplete_SkipsTheMemoryClearForAModelWithoutMemory(t *testing.T) {
	f := newFakeChat()
	f.mem = 0
	c := openFakeChat(t, f, testSpec())
	if _, err := c.Complete(context.Background(), "s", "u"); err != nil {
		t.Fatal(err)
	}
	if n := f.count("memoryClear"); n != 0 {
		t.Errorf("memoryClear was called %d times without a memory handle", n)
	}
}

func TestComplete_ReportsDecodeFailuresAndFreesTheSampler(t *testing.T) {
	for _, tc := range []struct {
		name  string
		setup func(*fakeChat)
	}{
		{"error on the prompt", func(f *fakeChat) { f.decodeErr = errors.New("out of memory") }},
		{"nonzero return on the prompt", func(f *fakeChat) { f.decodeRet = 1 }},
		{"error on a generated token", func(f *fakeChat) { f.script = []int32{'a', 'b'}; f.failDecode = 2 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeChat()
			tc.setup(f)
			c := openFakeChat(t, f, testSpec())
			if _, err := c.Complete(context.Background(), "s", "u"); err == nil {
				t.Fatal("a failed decode was not reported")
			}
			if f.count("freeSampler") != 1 {
				t.Errorf("freeSampler called %d times, want once", f.count("freeSampler"))
			}
		})
	}
}

func TestComplete_StopsWhenTheContextIsCanceled(t *testing.T) {
	f := newFakeChat()
	f.script = []int32{'a', 'b', 'c'}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	f.onSample = func(n int) {
		if n == 1 {
			cancel()
		}
	}
	c := openFakeChat(t, f, testSpec())
	if _, err := c.Complete(ctx, "s", "u"); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if f.samples != 1 {
		t.Errorf("drew %d tokens after the cancel, want the loop to stop", f.samples)
	}
	if f.count("freeSampler") != 1 {
		t.Errorf("freeSampler called %d times, want once", f.count("freeSampler"))
	}
}

func TestComplete_StopsBeforeStartingOnACanceledContext(t *testing.T) {
	f := newFakeChat()
	c := openFakeChat(t, f, testSpec())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Complete(ctx, "s", "u"); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(f.calls) != 0 {
		t.Errorf("a canceled call reached llama.cpp: %v", f.names())
	}
}

// TestComplete_JoinsTheBytesOfTokensBeforeReadingUTF8: a character can be
// spread over several tokens, so the reply is assembled as bytes and read as
// UTF-8 once, and a character cut off at the end is replaced, not kept broken.
func TestComplete_JoinsTheBytesOfTokensBeforeReadingUTF8(t *testing.T) {
	pieces := map[int32][]byte{1000: {0xE2, 0x82}, 1001: {0xAC}}
	for _, tc := range []struct {
		script []int32
		want   string
	}{
		{[]int32{1000, 1001}, "€"},
		{[]int32{'a', 1000}, "a\uFFFD"},
	} {
		f := newFakeChat()
		f.pieces = pieces
		f.script = tc.script
		c := openFakeChat(t, f, testSpec())
		got, err := c.Complete(context.Background(), "s", "u")
		if err != nil {
			t.Fatal(err)
		}
		if got != tc.want || !utf8.ValidString(got) {
			t.Errorf("script %v: Complete = %q, want %q", tc.script, got, tc.want)
		}
	}
}

// TestOpenCompleter_RejectsATemplateThatDoesNotKeepBothMessages: some
// templates drop a system message or reorder the turns. The prompt cannot be
// built around the messages then, and a prompt without them is worse than an
// error, which open raises by trying the template once.
func TestOpenCompleter_RejectsATemplateThatDoesNotKeepBothMessages(t *testing.T) {
	for name, render := range map[string]func(system, user string) string{
		"no user message":   func(system, _ string) string { return "<chat>" + system },
		"no system message": func(_, user string) string { return "<chat>" + user },
		"turns reordered":   func(system, user string) string { return user + system },
	} {
		t.Run(name, func(t *testing.T) {
			f := newFakeChat()
			f.render = render
			_, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, testSpec(), 2, nil)
			if err == nil || !strings.Contains(err.Error(), "chatml") {
				t.Fatalf("err = %v, want a refusal naming the template", err)
			}
			if names := strings.Join(f.names(), ","); !strings.HasSuffix(names, "freeContext,freeModel") {
				t.Fatalf("calls %s: the context must be freed before its model", names)
			}
		})
	}
}

func TestComplete_CutsAHugeMessage(t *testing.T) {
	f := newFakeChat()
	f.trainCtx = 1 << 16
	spec := testSpec()
	spec.ContextTokens = 1 << 16
	c := openFakeChat(t, f, spec)
	if _, err := c.Complete(context.Background(), "s", strings.Repeat("a", maxInputBytes+10)); err != nil {
		t.Fatal(err)
	}
	var user string
	for _, tok := range f.tokenized {
		if len(tok.text) > len(user) {
			user = tok.text
		}
	}
	if len(user) != maxInputBytes {
		t.Errorf("the model was given %d bytes of the message, want it cut to %d", len(user), maxInputBytes)
	}
}

func TestComplete_ReplacesInvalidUTF8InAMessage(t *testing.T) {
	f := newFakeChat()
	c := openFakeChat(t, f, testSpec())
	if _, err := c.Complete(context.Background(), "s", "bad \xff byte"); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(f.tokenized, tokenizeCall{"bad \uFFFD byte", false, false}) {
		t.Errorf("tokenized %v, want the invalid byte replaced", f.tokenized)
	}
}

func TestCompleter_CloseFreesTheContextThenTheModel(t *testing.T) {
	f := newFakeChat()
	c := openFakeChat(t, f, testSpec())
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if names := strings.Join(f.names(), ","); names != "freeContext,freeModel" {
		t.Fatalf("calls %s: want the context freed before its model, once", names)
	}
	f.calls = nil
	if _, err := c.Complete(context.Background(), "s", "u"); err == nil {
		t.Fatal("a closed Completer answered")
	}
	if len(f.calls) != 0 {
		t.Errorf("a closed Completer reached llama.cpp: %v", f.names())
	}
}

func TestOpenCompleter_StopsAtAFailedLoad(t *testing.T) {
	boom := errors.New("boom")
	t.Run("libraries", func(t *testing.T) {
		f := newFakeChat()
		f.loadErr = boom
		if _, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, testSpec(), 2, nil); !errors.Is(err, boom) {
			t.Fatalf("err = %v, want the load error", err)
		}
		if got := strings.Join(f.names(), ","); got != "load" {
			t.Fatalf("calls after a failed library load: %s, want only load", got)
		}
	})
	t.Run("model", func(t *testing.T) {
		f := newFakeChat()
		f.loadModelErr = boom
		_, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, testSpec(), 2, nil)
		if !errors.Is(err, boom) || !strings.Contains(err.Error(), testChatModelPath) {
			t.Fatalf("err = %v, want the load error naming %s", err, testChatModelPath)
		}
		if f.count("freeModel") != 0 || f.count("newChatContext") != 0 {
			t.Fatalf("calls after a failed model load: %v", f.names())
		}
	})
}

func TestOpenCompleter_RejectsModelWithoutTrainingContext(t *testing.T) {
	f := newFakeChat()
	f.trainCtx = 0
	if _, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, testSpec(), 2, nil); err == nil {
		t.Fatal("a model without a training context length was accepted")
	}
	if args := f.only(t, "freeModel"); args[0] != fakeModel {
		t.Fatalf("freeModel(%v), want the loaded model %v", args[0], fakeModel)
	}
	if n := f.count("newChatContext"); n != 0 {
		t.Fatalf("newChatContext was called %d times for a model without a training context", n)
	}
}

func TestOpenCompleter_FreesTheModelWhenTheContextFails(t *testing.T) {
	boom := errors.New("cannot allocate the KV cache")
	f := newFakeChat()
	f.newContextErr = boom
	if _, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, testSpec(), 2, nil); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want it to wrap the context error", err)
	}
	if f.only(t, "freeModel")[0] != fakeModel || f.count("freeContext") != 0 {
		t.Fatalf("calls %v: want the model freed and no context", f.names())
	}
}

// TestOpenCompleter_FreesTheContextThenTheModelWhenTheTemplateFails: the
// template is tried once at open, so a name llama.cpp does not know is an
// error there and not at the first call, and nothing leaks.
func TestOpenCompleter_FreesTheContextThenTheModelWhenTheTemplateFails(t *testing.T) {
	f := newFakeChat()
	f.templateErr = errors.New("unknown template")
	_, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, testSpec(), 2, nil)
	if err == nil || !strings.Contains(err.Error(), "chatml") {
		t.Fatalf("err = %v, want a refusal naming the template", err)
	}
	if names := strings.Join(f.names(), ","); !strings.HasSuffix(names, "freeContext,freeModel") {
		t.Fatalf("calls %s: the context must be freed before its model", names)
	}
}

func TestOpenCompleter_RejectsAContextOfNoTokens(t *testing.T) {
	f := newFakeChat()
	f.grantNothing = true
	if _, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, testSpec(), 2, nil); err == nil {
		t.Fatal("a context granting no tokens was accepted")
	}
	if names := strings.Join(f.names(), ","); !strings.HasSuffix(names, "freeContext,freeModel") {
		t.Fatalf("calls %s: the context must be freed before its model", names)
	}
}

// TestOpenCompleter_SizesTheContext: the context is the spec's size (2048 by
// default) within the model's training window, batches are 512 tokens within
// the context, and what llama.cpp grants below that wins.
func TestOpenCompleter_SizesTheContext(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		train                  int32
		spec                   int
		grantCtx, grantBatch   int
		askCtx, askBatch       int32
		wantCtx, wantBatchSize int
	}{
		{"defaults", 32768, 0, 0, 0, 2048, 512, 2048, 512},
		{"the spec's size", 32768, 1024, 0, 0, 1024, 512, 1024, 512},
		{"a small context shrinks the batch", 32768, 300, 0, 0, 300, 300, 300, 300},
		{"the training window caps it", 1000, 4096, 0, 0, 1000, 512, 1000, 512},
		{"granted less", 32768, 0, 700, 100, 2048, 512, 700, 100},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeChat()
			f.trainCtx = tc.train
			f.grantCtx, f.grantBatch = tc.grantCtx, tc.grantBatch
			spec := testSpec()
			spec.ContextTokens = tc.spec
			c, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, spec, 3, nil)
			if err != nil {
				t.Fatal(err)
			}
			if args := f.only(t, "newChatContext"); args[1] != tc.askCtx || args[2] != tc.askBatch || args[3] != 3 {
				t.Errorf("newChatContext%v, want nCtx %d, batch %d, 3 threads", args, tc.askCtx, tc.askBatch)
			}
			if c.nCtx != tc.wantCtx || c.batch != tc.wantBatchSize {
				t.Errorf("context %d tokens, batches %d, want %d and %d", c.nCtx, c.batch, tc.wantCtx, tc.wantBatchSize)
			}
		})
	}
}

// fakeChatRuntime does what fakeRuntime does and routes the Completer's
// llama.cpp calls to f too; both fakes restore the globals when the test ends.
func fakeChatRuntime(t *testing.T, f *fakeChat, tmp string) Runtime {
	t.Helper()
	rt := fakeRuntime(t, newFakeAPI(), tmp)
	orig := chatLib
	chatLib = f
	t.Cleanup(func() { chatLib = orig })
	return rt
}

func TestOpenCompleter_ValidatesTheSpecBeforeTouchingAnything(t *testing.T) {
	good := testSpec()
	for _, tc := range []struct {
		name string
		edit func(*CompleterSpec, *Runtime)
		want string
	}{
		{"no name", func(s *CompleterSpec, _ *Runtime) { s.Name = "" }, "Name"},
		{"no file", func(s *CompleterSpec, _ *Runtime) { s.File = "" }, "File"},
		{"a URL without a digest", func(s *CompleterSpec, _ *Runtime) { s.URL = "https://example.com/m.gguf" }, "SHA256"},
		{"no template", func(s *CompleterSpec, _ *Runtime) { s.Template = "" }, "Template"},
		{"negative context", func(s *CompleterSpec, _ *Runtime) { s.ContextTokens = -1 }, "negative"},
		{"negative reply cap", func(s *CompleterSpec, _ *Runtime) { s.MaxNewTokens = -1 }, "negative"},
		{"no library directory", func(_ *CompleterSpec, rt *Runtime) { rt.LibDir = "" }, "LibDir"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeChat()
			rt := fakeChatRuntime(t, f, t.TempDir())
			spec := good
			tc.edit(&spec, &rt)
			if _, err := OpenCompleter(context.Background(), spec, rt); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want a refusal naming %s", err, tc.want)
			}
			if len(f.calls) != 0 {
				t.Errorf("an invalid spec reached llama.cpp: %v", f.names())
			}
		})
	}
}

// TestOpenCompleter_DownloadsAndLoads runs the whole OpenCompleter path
// (pinned downloads, runtime load, model open) without network or native code,
// logging each step as Open does.
func TestOpenCompleter_DownloadsAndLoads(t *testing.T) {
	tmp := t.TempDir()
	f := newFakeChat()
	f.script = []int32{'o', 'k'}
	rt := fakeChatRuntime(t, f, tmp)
	offline(t)
	_, modelAsset := serve(t, []byte("GGUF fake chat model"))
	modelPath := filepath.Join(tmp, "models", "chat.gguf")
	var logs bytes.Buffer
	rt.Logger = slog.New(slog.NewTextHandler(&logs, nil))
	spec := CompleterSpec{Name: "org/chat", File: modelPath, URL: modelAsset.URL, SHA256: modelAsset.SHA256, Size: modelAsset.Size, Template: "chatml"}
	c, err := OpenCompleter(context.Background(), spec, rt)
	if err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "loadModel"); args[0] != modelPath {
		t.Errorf("loadModel(%v), want the model downloaded into %s", args[0], modelPath)
	}
	if args := f.only(t, "newChatContext"); args[3] != 2 {
		t.Errorf("newChatContext ran with %v threads, want the configured 2", args[3])
	}
	checkLog(t, logs.String(), []logLine{
		{"downloading llama.cpp " + LibTag, nil},
		{"installed llama.cpp " + LibTag, nil},
		{"downloading model org/chat", []string{"url=" + modelAsset.URL, "sha256=" + modelAsset.SHA256, "dest=" + modelPath}},
		{"downloaded model", []string{"path=" + modelPath}},
		{"llama.cpp loaded", []string{"version=" + libVersion, "threads=2", "model=org/chat"}},
	})
	got, err := c.Complete(context.Background(), "s", "u")
	if err != nil || got != "ok" {
		t.Fatalf("Complete = %q, %v", got, err)
	}
}

// TestNewCompleter_RunsThePinnedModel: Config.Model selects one of the pinned
// chat models, downloaded into ModelsDir under a fixed file name, and the
// Completer frames messages with that model's template.
func TestNewCompleter_RunsThePinnedModel(t *testing.T) {
	tmp := t.TempDir()
	f := newFakeChat()
	rt := fakeChatRuntime(t, f, tmp)
	offline(t)
	_, a := serve(t, []byte("GGUF fake chat model"))
	orig := defaultCompleterAsset
	defaultCompleterAsset = a
	t.Cleanup(func() { defaultCompleterAsset = orig })
	modelsDir := filepath.Join(tmp, "models")
	c, err := NewCompleter(context.Background(), Config{ModelsDir: modelsDir, LibDir: rt.LibDir, ManagedLibDir: true, Threads: 2})
	if err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "loadModel"); args[0] != filepath.Join(modelsDir, defaultCompleterFile) {
		t.Errorf("loadModel(%v), want the pinned file inside %s", args[0], modelsDir)
	}
	f.calls = nil
	if _, err := c.Complete(context.Background(), "s", "u"); err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "applyTemplate"); args[0] != "chatml" {
		t.Errorf("applyTemplate(%v), want the pinned model's chatml", args[0])
	}
}

func TestNewCompleter_UnknownModelListsTheAvailableOnes(t *testing.T) {
	_, err := NewCompleter(context.Background(), Config{Model: "BAAI/bge-m3", LibDir: t.TempDir()})
	if err == nil || !strings.Contains(err.Error(), DefaultCompleterModel) {
		t.Fatalf("err = %v, want a refusal listing the chat models", err)
	}
}

// TestOpenCompleter_SharesTheLibrariesWithTheEmbedder: an application that
// embeds and generates runs both in one process, and the libraries bind into
// process globals, so they load once whichever opens first.
func TestOpenCompleter_SharesTheLibrariesWithTheEmbedder(t *testing.T) {
	tmp := t.TempDir()
	fa := newFakeAPI()
	rt := fakeRuntime(t, fa, tmp)
	fc := newFakeChat()
	orig := chatLib
	chatLib = fc
	t.Cleanup(func() { chatLib = orig })
	for _, name := range []string{"embedder.gguf", "chat.gguf"} {
		if err := os.WriteFile(filepath.Join(tmp, name), []byte("GGUF fake model"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := Open(context.Background(), ModelSpec{Name: "org/embed", File: filepath.Join(tmp, "embedder.gguf")}, rt); err != nil {
		t.Fatal(err)
	}
	spec := CompleterSpec{Name: "org/chat", File: filepath.Join(tmp, "chat.gguf"), Template: "chatml"}
	if _, err := OpenCompleter(context.Background(), spec, rt); err != nil {
		t.Fatalf("a Completer next to an Embedder: %v", err)
	}
	if n := fa.count("load") + fc.count("load"); n != 1 {
		t.Fatalf("the libraries were loaded %d times, want once", n)
	}
}

func TestLookupCompleterSpec(t *testing.T) {
	spec, err := LookupCompleterSpec("", "")
	if err != nil {
		t.Fatal(err)
	}
	if spec.Name != DefaultCompleterModel || spec.File != filepath.Join("models", defaultCompleterFile) || spec.Template != "chatml" {
		t.Fatalf("spec = %+v, want the default chat model from the models directory", spec)
	}
	if spec.URL == "" || spec.SHA256 == "" || spec.Size == 0 {
		t.Fatalf("spec = %+v, want a pinned download", spec)
	}
	named, err := LookupCompleterSpec(DefaultCompleterModel, "/data/models")
	if err != nil || filepath.Dir(named.File) != "/data/models" {
		t.Fatalf("spec = %+v, %v, want a file inside /data/models", named, err)
	}
	if def := DefaultCompleterSpec("/data/models"); def != named {
		t.Errorf("DefaultCompleterSpec = %+v, want %+v", def, named)
	}
	if _, err := LookupCompleterSpec("org/unknown", ""); err == nil || !strings.Contains(err.Error(), DefaultCompleterModel) {
		t.Fatalf("err = %v, want a refusal listing the available models", err)
	}
}

// TestPinnedCompletersAreWellFormed: a pin with a typo in its digest or a
// template name llama.cpp does not know would only show up on the first
// download, so the shape is checked here.
func TestPinnedCompletersAreWellFormed(t *testing.T) {
	for _, m := range pinnedCompleters() {
		if !strings.HasPrefix(m.asset.URL, "https://") || len(m.asset.SHA256) != 64 || m.asset.Size <= 0 {
			t.Errorf("%s: asset %+v is not a pinned https download", m.name, m.asset)
		}
		if m.file == "" || m.file != filepath.Base(m.file) || !strings.HasSuffix(m.file, ".gguf") {
			t.Errorf("%s: file %q is not a bare .gguf name", m.name, m.file)
		}
		if m.template == "" {
			t.Errorf("%s: no chat template", m.name)
		}
	}
}

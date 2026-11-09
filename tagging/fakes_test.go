package tagging

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
)

// tallyEmbedder wraps an Embedder (FakeEmbedder when inner is nil) and counts
// the texts it is asked to embed, per role and per text, so a test can assert
// what the engine embedded and what it found in the cache. It counts the
// texts of a call that fails too. It can fail every call of one role, and
// return a 3-dimensional vector for chosen texts. It is safe for concurrent
// use.
type tallyEmbedder struct {
	inner langai.Embedder
	// noDims has Dimensions report 0, so a vector of another length passes
	// the embedder's own check and reaches the engine's
	noDims bool

	mu    sync.Mutex
	texts map[langai.Role]map[string]int
	calls map[langai.Role]int
	fail  map[langai.Role]error
	short map[string]bool
	// failOn, when not empty, fails every call with a text containing it
	failOn    string
	failOnErr error
	// fixed holds the vector Embed returns for a text instead of the inner
	// embedder's
	fixed map[string][]float32
	// hook, when set, runs at the start of every Embed with the number of the
	// call in any role (1 for the first); an error from it fails the call
	hook   func(call int) error
	nCalls int
	// largest is the most texts one Embed was asked for
	largest int
	// hang holds the roles whose Embed never answers: it blocks until its ctx
	// ends, as a model that hangs does, and gave is how many such calls gave
	// up when their ctx ended
	hang map[langai.Role]bool
	gave int
}

// hangOn has every Embed of role block until its ctx ends; false clears it.
func (e *tallyEmbedder) hangOn(role langai.Role, on bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.hang == nil {
		e.hang = map[langai.Role]bool{}
	}
	e.hang[role] = on
}

// gaveUp returns how many hanging Embed calls returned because their ctx
// ended.
func (e *tallyEmbedder) gaveUp() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.gave
}

// largestBatch returns the most texts one Embed call was asked for.
func (e *tallyEmbedder) largestBatch() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.largest
}

func (e *tallyEmbedder) embedder() langai.Embedder {
	if e.inner == nil {
		return langaitest.FakeEmbedder{}
	}
	return e.inner
}

func (e *tallyEmbedder) Dimensions() int {
	if e.noDims {
		return 0
	}
	return e.embedder().Dimensions()
}

func (e *tallyEmbedder) Embed(ctx context.Context, role langai.Role, texts []string) ([][]float32, error) {
	e.mu.Lock()
	if e.texts == nil {
		e.texts, e.calls = map[langai.Role]map[string]int{}, map[langai.Role]int{}
	}
	if e.texts[role] == nil {
		e.texts[role] = map[string]int{}
	}
	for _, t := range texts {
		e.texts[role][t]++
	}
	e.calls[role]++
	e.nCalls++
	e.largest = max(e.largest, len(texts))
	call, hook, hangs := e.nCalls, e.hook, e.hang[role]
	err := e.fail[role]
	for _, t := range texts {
		if e.failOn != "" && strings.Contains(t, e.failOn) {
			err = e.failOnErr
		}
	}
	short, fixed := maps.Clone(e.short), maps.Clone(e.fixed)
	e.mu.Unlock()
	if hook != nil {
		if err := hook(call); err != nil {
			return nil, err
		}
	}
	if err != nil {
		return nil, err
	}
	if hangs {
		<-ctx.Done()
		e.mu.Lock()
		e.gave++
		e.mu.Unlock()
		return nil, ctx.Err()
	}
	vs, err := e.embedder().Embed(ctx, role, texts)
	if err != nil {
		return nil, err
	}
	for i, t := range texts {
		if short[t] {
			vs[i] = []float32{1, 0, 0}
		}
		if v, ok := fixed[t]; ok {
			vs[i] = slices.Clone(v)
		}
	}
	return vs, nil
}

// vecFor has Embed return v for text.
func (e *tallyEmbedder) vecFor(text string, v []float32) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.fixed == nil {
		e.fixed = map[string][]float32{}
	}
	e.fixed[text] = v
}

// atCall has hook run at the start of every Embed.
func (e *tallyEmbedder) atCall(hook func(call int) error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.hook = hook
}

// logRecorder is a slog.Handler that keeps every record, so a test can assert
// what the engine reported. It is safe for concurrent use.
type logRecorder struct {
	mu      sync.Mutex
	records []slog.Record
}

func (l *logRecorder) Enabled(context.Context, slog.Level) bool { return true }

func (l *logRecorder) Handle(_ context.Context, r slog.Record) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.records = append(l.records, r.Clone())
	return nil
}

func (l *logRecorder) WithAttrs([]slog.Attr) slog.Handler { return l }

func (l *logRecorder) WithGroup(string) slog.Handler { return l }

// lines returns each record as "LEVEL message key=value ...", in order.
func (l *logRecorder) lines() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]string, len(l.records))
	for i, r := range l.records {
		var b strings.Builder
		b.WriteString(r.Level.String() + " " + r.Message)
		r.Attrs(func(a slog.Attr) bool {
			b.WriteString(" " + a.Key + "=" + a.Value.String())
			return true
		})
		out[i] = b.String()
	}
	return out
}

// matching returns the lines that contain every one of parts.
func (l *logRecorder) matching(parts ...string) []string {
	var out []string
	for _, line := range l.lines() {
		ok := true
		for _, p := range parts {
			ok = ok && strings.Contains(line, p)
		}
		if ok {
			out = append(out, line)
		}
	}
	return out
}

// faultyCache is an EmbeddingCache whose writes can fail: Put with putErr, and
// Get, after reading, with getErr next to the vectors it read.
type faultyCache struct {
	EmbeddingCache
	mu             sync.Mutex
	putErr, getErr error
}

func (c *faultyCache) Get(ctx context.Context, index string, keys []CacheKey) (map[CacheKey][]float32, error) {
	got, err := c.EmbeddingCache.Get(ctx, index, keys)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return got, c.getErr
}

func (c *faultyCache) Put(ctx context.Context, index string, vecs map[CacheKey][]float32) error {
	c.mu.Lock()
	err := c.putErr
	c.mu.Unlock()
	if err != nil {
		return err
	}
	return c.EmbeddingCache.Put(ctx, index, vecs)
}

// failRole makes every Embed of role fail with err; a nil err clears it.
func (e *tallyEmbedder) failRole(role langai.Role, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.fail == nil {
		e.fail = map[langai.Role]error{}
	}
	if err == nil {
		delete(e.fail, role)
		return
	}
	e.fail[role] = err
}

// failContaining makes every Embed with a text containing word fail with err;
// a nil err clears it.
func (e *tallyEmbedder) failContaining(word string, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if err == nil {
		word = ""
	}
	e.failOn, e.failOnErr = word, err
}

// shortFor has Embed return a 3-dimensional vector for text.
func (e *tallyEmbedder) shortFor(text string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.short == nil {
		e.short = map[string]bool{}
	}
	e.short[text] = true
}

// count returns how many texts were embedded in role.
func (e *tallyEmbedder) count(role langai.Role) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	n := 0
	for _, c := range e.texts[role] {
		n += c
	}
	return n
}

// total returns how many texts were embedded in any role.
func (e *tallyEmbedder) total() int { return e.count(langai.RoleQuery) + e.count(langai.RoleDocument) }

// times returns how often text was embedded in role.
func (e *tallyEmbedder) times(role langai.Role, text string) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.texts[role][text]
}

// callsOf returns how many Embed calls were made in role.
func (e *tallyEmbedder) callsOf(role langai.Role) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.calls[role]
}

// stubEmbedder maps texts to fixed vectors. Unknown text is an error, so
// nothing is scored by accident.
type stubEmbedder struct {
	dims int
	vecs map[string][]float32
}

func (s stubEmbedder) Dimensions() int { return s.dims }

func (s stubEmbedder) Embed(_ context.Context, _ langai.Role, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		v, ok := s.vecs[text]
		if !ok {
			return nil, fmt.Errorf("stubEmbedder: no vector for %q", text)
		}
		out[i] = v
	}
	return out, nil
}

// vectorlessEmbedder breaks the Embedder contract on purpose: every call
// succeeds without returning a single vector.
type vectorlessEmbedder struct{}

func (vectorlessEmbedder) Dimensions() int { return 64 }

func (vectorlessEmbedder) Embed(context.Context, langai.Role, []string) ([][]float32, error) {
	return nil, nil
}

// brokenEmbedder breaks the Embedder contract on purpose: it says its vectors
// have dims entries, and returns doc for every text embedded as a document (a
// tag's name) and query for every one embedded as a query (an item's text).
type brokenEmbedder struct {
	dims       int
	doc, query []float32
}

func (b brokenEmbedder) Dimensions() int { return b.dims }

func (b brokenEmbedder) Embed(_ context.Context, role langai.Role, texts []string) ([][]float32, error) {
	v := b.query
	if role == langai.RoleDocument {
		v = b.doc
	}
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = slices.Clone(v)
	}
	return out, nil
}

// fakeCatalog is a map-backed Catalog. It ignores the index and partition, so
// one catalog can stand behind several spaces. Members returns the members of
// the ids asked for, plus every entry of extra, asked or not. It counts its
// calls and records the ids of each Members call. It can fail with a set
// error, and Tags can block until release is closed, or its ctx ends; entered
// receives a value (without blocking) each time Tags starts. onMembers, when
// set, runs once at the next Members call: inside a rebuild, between the
// engine's read of the index and its Apply. It is safe for concurrent use.
type fakeCatalog struct {
	mu        sync.Mutex
	tags      []CatalogTag
	members   map[string][]string
	extra     map[string][]string
	err       error
	release   chan struct{}
	entered   chan struct{}
	onMembers func()
	tagsCalls int
	memCalls  int
	memIDs    [][]string
	// panicWith, when not nil, is what Tags panics with
	panicWith any
	// inTags is how many Tags calls are running, maxInTags the most that
	// ever ran together
	inTags, maxInTags int
	// delay is how long every Tags call takes, unless its ctx ends first
	delay time.Duration
}

// setPanic has every later Tags call panic with v; nil clears it.
func (c *fakeCatalog) setPanic(v any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.panicWith = v
}

// overlap returns the most Tags calls that ever ran together.
func (c *fakeCatalog) overlap() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.maxInTags
}

func (c *fakeCatalog) Tags(ctx context.Context, _, _ string) ([]CatalogTag, error) {
	c.mu.Lock()
	c.tagsCalls++
	c.inTags++
	c.maxInTags = max(c.maxInTags, c.inTags)
	release, entered, panicWith, delay := c.release, c.entered, c.panicWith, c.delay
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		c.inTags--
		c.mu.Unlock()
	}()
	if panicWith != nil {
		panic(panicWith)
	}
	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if entered != nil {
		select {
		case entered <- struct{}{}:
		default:
		}
	}
	if release != nil {
		select {
		case <-release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err != nil {
		return nil, c.err
	}
	return slices.Clone(c.tags), nil
}

func (c *fakeCatalog) Members(_ context.Context, _, _ string, tagIDs []string) (map[string][]string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if hook := c.onMembers; hook != nil {
		c.onMembers = nil
		hook()
	}
	c.memCalls++
	c.memIDs = append(c.memIDs, slices.Clone(tagIDs))
	if c.err != nil {
		return nil, c.err
	}
	out := map[string][]string{}
	for _, id := range tagIDs {
		if m, ok := c.members[id]; ok {
			out[id] = slices.Clone(m)
		}
	}
	for id, m := range c.extra {
		out[id] = slices.Clone(m)
	}
	return out, nil
}

// set replaces the tags and members.
func (c *fakeCatalog) set(tags []CatalogTag, members map[string][]string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tags, c.members = tags, members
}

// runAtMembers has the next Members call run hook first.
func (c *fakeCatalog) runAtMembers(hook func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onMembers = hook
}

// setErr has every later call fail with err; nil clears it.
func (c *fakeCatalog) setErr(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.err = err
}

// counts returns the number of Tags and Members calls so far.
func (c *fakeCatalog) counts() (tags, members int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.tagsCalls, c.memCalls
}

// lastMemberIDs returns the ids of the last Members call, or nil.
func (c *fakeCatalog) lastMemberIDs() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.memIDs) == 0 {
		return nil
	}
	return c.memIDs[len(c.memIDs)-1]
}

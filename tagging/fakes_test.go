package tagging

import (
	"context"
	"maps"
	"slices"
	"sync"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
)

// tallyEmbedder wraps an Embedder (FakeEmbedder when inner is nil) and counts
// the texts it is asked to embed, per role and per text, so a test can assert
// what the engine embedded and what it found in the cache. It counts the
// texts of a call that fails too. It can fail every call of one role, and
// return a 3-dimensional vector for chosen texts. Named apart from the older
// countingEmbedder of tagger_test.go, which counts calls. It is safe for
// concurrent use.
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
	err := e.fail[role]
	short := maps.Clone(e.short)
	e.mu.Unlock()
	if err != nil {
		return nil, err
	}
	vs, err := e.embedder().Embed(ctx, role, texts)
	if err != nil {
		return nil, err
	}
	for i, t := range texts {
		if short[t] {
			vs[i] = []float32{1, 0, 0}
		}
	}
	return vs, nil
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

// fakeCatalog is a map-backed Catalog. It ignores the index and partition, so
// one catalog can stand behind several spaces. Members returns the members of
// the ids asked for, plus every entry of extra, asked or not. It counts its
// calls and records the ids of each Members call. It can fail with a set
// error, and Tags can block until release is closed, or its ctx ends; entered
// receives a value (without blocking) each time Tags starts. It is safe for
// concurrent use.
type fakeCatalog struct {
	mu        sync.Mutex
	tags      []CatalogTag
	members   map[string][]string
	extra     map[string][]string
	err       error
	release   chan struct{}
	entered   chan struct{}
	tagsCalls int
	memCalls  int
	memIDs    [][]string
}

func (c *fakeCatalog) Tags(ctx context.Context, _, _ string) ([]CatalogTag, error) {
	c.mu.Lock()
	c.tagsCalls++
	release, entered := c.release, c.entered
	c.mu.Unlock()
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

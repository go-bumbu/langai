// Package status is the example service's home page: what is loaded and how.
// It lists every model with the state of its lazy loader (not started,
// loading, loaded, failed or closed), its identity and dimensions and what
// runs it, and loads one on request; every collection of the vector stores
// with the model its vectors were recorded under (vector.Store.Model), how
// many documents it holds, and a reset that empties it; and the Tagger's
// embedding cache with the result of its last prune. Handler serves it over
// HTTP. It knows the library's ports only, so its tests run on fakes.
package status

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/vector"
)

// Errors the handler maps to statuses.
var (
	// ErrNotFound is returned for a model or collection key the Service
	// does not know.
	ErrNotFound = errors.New("status: no such model or collection")
	// ErrNotLazy is returned by Load for a model that has no loader: one
	// loaded at startup, or one that is no model at all.
	ErrNotLazy = errors.New("status: the model has no lazy loader")
	// ErrNoAction is returned for an action the Service was not given: a
	// collection without a reset, a cache without a prune.
	ErrNoAction = errors.New("status: not available")
)

// Recovery is what the page says next to a load that failed: the lazy
// loaders keep a failure and never retry on their own.
const Recovery = "A failed load is kept: every call returns this error and nothing loads again. " +
	"Fix the cause and restart the service. In an application that must stay up, build a new lazy value " +
	"with the same constructor (langai.NewLazyEmbedder and its kin), use it in place of this one and Close this one."

// Loader is the lazy loader of a model. langai.LazyEmbedder,
// langai.LazyImageEmbedder and langai.LazyCompleter all are one.
type Loader interface {
	State() (langai.LazyState, error)
	Load(ctx context.Context) error
}

// Model describes one model of the service.
type Model struct {
	// Key names the model in the API's paths.
	Key string
	// Name says what the model is for, UsedBy which pages call it.
	Name   string
	UsedBy string
	// ID is the model's identity, the string the stores record with its
	// vectors; for a chat model, its name.
	ID string
	// Dimensions is the size of its vectors; 0 for a model that makes none.
	Dimensions int
	// Runtime is the provider that runs it.
	Runtime string
	// RunsOn says what the model runs on (a CPU, a GPU device); it is
	// asked at every view, since a lazy model only knows once it is loaded.
	// Nil leaves it out.
	RunsOn func() string
	// Lazy is the model's lazy loader; nil for a model that has none, whose
	// state is then Fixed.
	Lazy Loader
	// Fixed is the state shown for a model without a loader, such as
	// "loaded at startup" or "no model".
	Fixed string
	// Why says why the model is, or is not, behind a lazy loader.
	Why string
}

// Collection describes one collection of a vector store, as a page's index
// uses it.
type Collection struct {
	// Key names the collection in the API's paths.
	Key string
	// Name says what it holds, Page is the path of the page that fills it.
	Name string
	Page string
	// Store is the store it lives in and Coord its place there; Backend
	// describes the store ("sqlite-vec · data/index-x.sqlite").
	Store   vector.Store
	Coord   vector.Coord
	Backend string
	// Model is the identity of the model the page's index writes with now.
	// When the collection records another one, the index refuses to write
	// or search until the collection is reset.
	Model string
	// Reset empties the collection (search.Index.Reset or
	// search.MediaIndex.Reset) together with what the page keeps about it;
	// nil leaves the action out.
	Reset func(ctx context.Context) error
}

// Prune is the outcome of one run of tagging.Tagger.Prune.
type Prune struct {
	At      time.Time
	Removed int64
	// UnusedFor is the age the run pruned at: embeddings not used for that
	// long were removed.
	UnusedFor time.Duration
	Err       error
}

// Cache describes the Tagger's embedding cache.
type Cache struct {
	// Name and Backend say what it is and where it lives.
	Name    string
	Backend string
	// Size returns the bytes of the database that holds the cache; nil
	// leaves the size out.
	Size func(ctx context.Context) (int64, error)
	// LastPrune returns the outcome of the last prune; the zero Prune
	// means none ran yet. Nil leaves it out.
	LastPrune func() Prune
	// PruneNow prunes the cache as the service's schedule does; nil leaves
	// the action out.
	PruneNow func(ctx context.Context) (Prune, error)
}

// Page is one feature of the service, a page of its own, for the list of
// links: Shows says in plain words what it does, Uses names the parts of the
// library it exercises, Choices the models it can run, and Try lists things
// to do on it.
type Page struct {
	Path    string   `json:"path"`
	Name    string   `json:"name"`
	Shows   string   `json:"shows"`
	Uses    []string `json:"uses,omitempty"`
	Choices []Choice `json:"choices,omitempty"`
	Try     []string `json:"try,omitempty"`
}

// Choice is one model a feature needs, as a role ("media model"), with
// every model that can fill it: the one the service runs and the ones it
// could. Note says how to switch, or why there is nothing to switch.
type Choice struct {
	Role    string   `json:"role"`
	Note    string   `json:"note,omitempty"`
	Options []Option `json:"options"`
}

// Option is one model a Choice can run.
type Option struct {
	Name string `json:"name"`
	// Engine names what runs it: llama.cpp, ONNX Runtime, pure Go.
	Engine string `json:"engine"`
	// Dims is the size of its vectors; 0 for a model that makes none.
	Dims int `json:"dims,omitempty"`
	// Bytes is its download size; 0 when the pin does not record one.
	Bytes int64 `json:"bytes,omitempty"`
	// Does says what it embeds or does, and how it differs from the others.
	Does string `json:"does,omitempty"`
	// Select is how to run it, e.g. "MODEL=facebook/dinov2-base"; empty when
	// the example cannot switch to it.
	Select string `json:"select,omitempty"`
	// Key is, for the option the service runs now, the key of that model
	// (Model.Key), so the page shows its state; empty for the others.
	Key string `json:"key,omitempty"`
}

// Fact is one line of the setup: a setting and its value.
type Fact struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// Config wires a Service.
type Config struct {
	Pages  []Page
	Models []Model
	// Collections lists the collections as they are now: a page may open
	// one only when it is first asked for it.
	Collections func() []Collection
	// Cache is the Tagger's embedding cache; nil leaves the section out.
	Cache *Cache
	// Setup is shown as is: the directories, the store backend.
	Setup []Fact
}

// Service answers the status page.
type Service struct {
	cfg     Config
	started time.Time

	mu sync.Mutex
	// loads holds how long each load this Service asked for took.
	loads map[string]time.Duration
}

// New returns a Service. Model keys must be unique and not empty.
func New(cfg Config) (*Service, error) {
	seen := map[string]bool{}
	for _, m := range cfg.Models {
		if m.Key == "" || seen[m.Key] {
			return nil, fmt.Errorf("status: model key %q is empty or used twice", m.Key)
		}
		seen[m.Key] = true
	}
	for _, p := range cfg.Pages {
		for _, c := range p.Choices {
			for _, o := range c.Options {
				if o.Key != "" && !seen[o.Key] {
					return nil, fmt.Errorf("status: page %q runs model %q, which is not among the models", p.Name, o.Key)
				}
			}
		}
	}
	return &Service{cfg: cfg, started: time.Now(), loads: map[string]time.Duration{}}, nil
}

// ModelStatus is a model as the page shows it.
type ModelStatus struct {
	Key        string `json:"key"`
	Name       string `json:"name"`
	UsedBy     string `json:"usedBy"`
	ID         string `json:"id"`
	Dimensions int    `json:"dimensions"`
	Runtime    string `json:"runtime"`
	RunsOn     string `json:"runsOn"`
	// Lazy says the model has a lazy loader, so State moves and Load works.
	Lazy bool `json:"lazy"`
	// State is the lazy loader's state in words (langai.LazyState), or the
	// fixed state of a model without one.
	State string `json:"state"`
	// Error is the kept failure of a load that failed, or what calls return
	// once the loader is closed; Recovery says what to do about a failure.
	Error    string `json:"error,omitempty"`
	Recovery string `json:"recovery,omitempty"`
	// LoadMs is how long the load took when this page asked for it; 0 when
	// a page's first use loaded the model, or nothing has yet.
	LoadMs int64  `json:"loadMs"`
	Why    string `json:"why"`
}

// CollectionStatus is a collection as the page shows it.
type CollectionStatus struct {
	Key        string `json:"key"`
	Name       string `json:"name"`
	Page       string `json:"page"`
	Backend    string `json:"backend"`
	Index      string `json:"index"`
	Partition  string `json:"partition"`
	Collection string `json:"collection"`
	// Model is what the page's index writes with, Recorded what the store
	// holds the collection's vectors under ("" for an empty collection).
	Model    string `json:"model"`
	Recorded string `json:"recorded"`
	// Mismatch says the two differ: the index refuses to write or search
	// until the collection is reset.
	Mismatch bool `json:"mismatch"`
	// Legacy says the store holds the index in the tables of the schema
	// before this version (vector.ErrLegacySchema): the reset drops them.
	Legacy   bool `json:"legacy,omitempty"`
	Docs     int  `json:"docs"`
	CanReset bool `json:"canReset"`
	// Error is why the store could not be asked.
	Error string `json:"error,omitempty"`
}

// PruneStatus is a prune as the page shows it.
type PruneStatus struct {
	At          time.Time `json:"at"`
	Removed     int64     `json:"removed"`
	UnusedForMs int64     `json:"unusedForMs"`
	Error       string    `json:"error,omitempty"`
}

// CacheStatus is the embedding cache as the page shows it.
type CacheStatus struct {
	Name    string `json:"name"`
	Backend string `json:"backend"`
	// Bytes is the size of the database that holds the cache; -1 when it is
	// not known.
	Bytes     int64  `json:"bytes"`
	SizeError string `json:"sizeError,omitempty"`
	// LastPrune is nil until a prune ran.
	LastPrune *PruneStatus `json:"lastPrune,omitempty"`
	CanPrune  bool         `json:"canPrune"`
}

// Snapshot is everything the page shows.
type Snapshot struct {
	Started     time.Time          `json:"started"`
	UptimeMs    int64              `json:"uptimeMs"`
	Pages       []Page             `json:"pages"`
	Setup       []Fact             `json:"setup"`
	Models      []ModelStatus      `json:"models"`
	Collections []CollectionStatus `json:"collections"`
	Cache       *CacheStatus       `json:"cache,omitempty"`
}

// Snapshot asks every model for its state and every store about its
// collections. It loads nothing: a lazy model's identity and dimensions come
// from its spec.
func (s *Service) Snapshot(ctx context.Context) Snapshot {
	snap := Snapshot{
		Started: s.started, UptimeMs: time.Since(s.started).Milliseconds(),
		Pages: s.cfg.Pages, Setup: s.cfg.Setup,
		Models: make([]ModelStatus, 0, len(s.cfg.Models)), Collections: []CollectionStatus{},
	}
	for _, m := range s.cfg.Models {
		snap.Models = append(snap.Models, s.modelStatus(m))
	}
	for _, c := range s.collections() {
		snap.Collections = append(snap.Collections, collectionStatus(ctx, c))
	}
	if s.cfg.Cache != nil {
		cs := cacheStatus(ctx, *s.cfg.Cache)
		snap.Cache = &cs
	}
	return snap
}

func (s *Service) collections() []Collection {
	if s.cfg.Collections == nil {
		return nil
	}
	return s.cfg.Collections()
}

func (s *Service) modelStatus(m Model) ModelStatus {
	st := ModelStatus{
		Key: m.Key, Name: m.Name, UsedBy: m.UsedBy, ID: m.ID, Dimensions: m.Dimensions,
		Runtime: m.Runtime, State: m.Fixed, Why: m.Why,
	}
	if m.RunsOn != nil {
		st.RunsOn = m.RunsOn()
	}
	if m.Lazy == nil {
		return st
	}
	state, err := m.Lazy.State()
	st.Lazy, st.State = true, state.String()
	if err != nil {
		st.Error = err.Error()
	}
	if state == langai.LazyFailed {
		st.Recovery = Recovery
	}
	s.mu.Lock()
	st.LoadMs = s.loads[m.Key].Milliseconds()
	s.mu.Unlock()
	return st
}

// Load loads the model of the given key now and returns its status after:
// loaded, or failed with the error the loader keeps. A model that is loaded
// already answers at once. ctx bounds the wait only: the load goes on when
// it ends, and the next view shows where it stands.
func (s *Service) Load(ctx context.Context, key string) (ModelStatus, error) {
	for _, m := range s.cfg.Models {
		if m.Key != key {
			continue
		}
		if m.Lazy == nil {
			return ModelStatus{}, ErrNotLazy
		}
		before, _ := m.Lazy.State()
		start := time.Now()
		err := m.Lazy.Load(ctx)
		if before == langai.LazyNotStarted && ctx.Err() == nil {
			// this call started the load and saw it end
			s.mu.Lock()
			s.loads[key] = time.Since(start)
			s.mu.Unlock()
		}
		st := s.modelStatus(m)
		if err != nil && st.Error == "" {
			// the wait ended before the load did
			st.Error = err.Error()
		}
		return st, nil
	}
	return ModelStatus{}, ErrNotFound
}

// Reset empties the collection of the given key and returns its status
// after.
func (s *Service) Reset(ctx context.Context, key string) (CollectionStatus, error) {
	for _, c := range s.collections() {
		if c.Key != key {
			continue
		}
		if c.Reset == nil {
			return CollectionStatus{}, ErrNoAction
		}
		if err := c.Reset(ctx); err != nil {
			return CollectionStatus{}, fmt.Errorf("status: reset %s: %w", key, err)
		}
		return collectionStatus(ctx, c), nil
	}
	return CollectionStatus{}, ErrNotFound
}

// Prune prunes the embedding cache now and returns its status after.
func (s *Service) Prune(ctx context.Context) (CacheStatus, error) {
	if s.cfg.Cache == nil || s.cfg.Cache.PruneNow == nil {
		return CacheStatus{}, ErrNoAction
	}
	if _, err := s.cfg.Cache.PruneNow(ctx); err != nil {
		return CacheStatus{}, fmt.Errorf("status: prune: %w", err)
	}
	return cacheStatus(ctx, *s.cfg.Cache), nil
}

// docsPage is how many documents one Docs call of a count reads.
const docsPage = 1000

func collectionStatus(ctx context.Context, c Collection) CollectionStatus {
	st := CollectionStatus{
		Key: c.Key, Name: c.Name, Page: c.Page, Backend: c.Backend,
		Index: c.Coord.Index, Partition: c.Coord.Partition, Collection: c.Coord.Collection,
		Model: c.Model, CanReset: c.Reset != nil,
	}
	if c.Store == nil {
		st.Error = "no store"
		return st
	}
	recorded, err := c.Store.Model(ctx, c.Coord)
	if err != nil {
		st.Error, st.Legacy = err.Error(), errors.Is(err, vector.ErrLegacySchema)
		return st
	}
	st.Recorded = recorded
	st.Mismatch = recorded != "" && c.Model != "" && recorded != c.Model
	// the number of documents is the length of the full list, read a page
	// at a time
	after := ""
	for {
		docs, err := c.Store.Docs(ctx, c.Coord, after, docsPage)
		if err != nil {
			st.Error = err.Error()
			return st
		}
		st.Docs += len(docs)
		if len(docs) < docsPage {
			return st
		}
		after = docs[len(docs)-1].ID
	}
}

func cacheStatus(ctx context.Context, c Cache) CacheStatus {
	st := CacheStatus{Name: c.Name, Backend: c.Backend, Bytes: -1, CanPrune: c.PruneNow != nil}
	if c.Size != nil {
		n, err := c.Size(ctx)
		if err != nil {
			st.SizeError = err.Error()
		} else {
			st.Bytes = n
		}
	}
	if c.LastPrune != nil {
		if p := c.LastPrune(); !p.At.IsZero() {
			st.LastPrune = &PruneStatus{At: p.At, Removed: p.Removed, UnusedForMs: p.UnusedFor.Milliseconds()}
			if p.Err != nil {
				st.LastPrune.Error = p.Err.Error()
			}
		}
	}
	return st
}

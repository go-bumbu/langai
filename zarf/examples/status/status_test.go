package status

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/search"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/vector"
)

// fixture is a Service over the library's own lazy loaders, each loading a
// fake: a text embedder that loads, an image embedder whose load fails, a
// chat model, and a hash "model" that has no loader. Its one collection is
// a search.Index over the text embedder on a MemStore.
type fixture struct {
	svc    *Service
	text   *sense.LazyEmbedder
	store  *vector.MemStore
	index  *search.Index
	coord  vector.Coord
	pruned int
	resets int
	// pageModel is the model the collection's row says the page runs.
	pageModel string
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{store: vector.NewMemStore(), coord: vector.Coord{Index: "examples", Partition: "local", Collection: "docs"}, pageModel: sensetest.FakeEmbedderModel}
	f.text = sense.NewLazyEmbedder(sensetest.FakeEmbedderModel, 64, func(context.Context) (sense.Embedder, error) {
		return sensetest.FakeEmbedder{}, nil
	})
	image := sense.NewLazyImageEmbedder(sensetest.FakeImageEmbedderModel, 48, func(context.Context) (sense.ImageEmbedder, error) {
		return nil, errors.New("the model file is missing")
	})
	chat := sense.NewLazyCompleter(func(context.Context) (sense.Completer, error) {
		return &sensetest.FakeCompleter{}, nil
	})
	t.Cleanup(func() { _ = f.text.Close(); _ = image.Close(); _ = chat.Close() })
	ix, err := search.New(f.text, f.store, f.coord.Index, f.coord.Partition, f.coord.Collection)
	if err != nil {
		t.Fatal(err)
	}
	f.index = ix
	last := Prune{}
	svc, err := New(Config{
		Pages: []Page{{Path: "textsearch/", Name: "Text search", Shows: "documents"}},
		Setup: []Fact{{Name: "store", Value: "memory"}},
		Models: []Model{
			{Key: "text", Name: "text embedding model", UsedBy: "text search", ID: f.text.Model(), Dimensions: f.text.Dimensions(), Runtime: "fake", RunsOn: func() string { return "cpu" }, Lazy: f.text, Why: "loads on first use"},
			{Key: "image", Name: "image model", ID: image.Model(), Dimensions: image.Dimensions(), Lazy: image},
			{Key: "chat", Name: "chat model", ID: "fake-chat", Lazy: chat},
			{Key: "hash", Name: "perceptual hash", ID: "phash-64", Dimensions: 64, Fixed: "no model"},
		},
		Collections: func() []Collection {
			return []Collection{
				{Key: "docs", Name: "documents", Page: "textsearch/", Store: f.store, Coord: f.coord, Backend: "memory", Model: f.pageModel,
					Reset: func(ctx context.Context) error { f.resets++; return ix.Reset(ctx) }},
				{Key: "fixed", Name: "read only", Store: f.store, Coord: vector.Coord{Index: "examples", Partition: "local", Collection: "other"}},
			}
		},
		Cache: &Cache{
			Name: "embedding cache", Backend: "memory",
			Size:      func(context.Context) (int64, error) { return 4096, nil },
			LastPrune: func() Prune { return last },
			PruneNow: func(context.Context) (Prune, error) {
				f.pruned++
				last = Prune{At: time.Now(), Removed: 3, UnusedFor: 30 * 24 * time.Hour}
				return last, nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	f.svc = svc
	return f
}

func model(t *testing.T, snap Snapshot, key string) ModelStatus {
	t.Helper()
	for _, m := range snap.Models {
		if m.Key == key {
			return m
		}
	}
	t.Fatalf("no model %q in the snapshot", key)
	return ModelStatus{}
}

func collection(t *testing.T, snap Snapshot, key string) CollectionStatus {
	t.Helper()
	for _, c := range snap.Collections {
		if c.Key == key {
			return c
		}
	}
	t.Fatalf("no collection %q in the snapshot", key)
	return CollectionStatus{}
}

func TestNew_RefusesEmptyAndRepeatedKeys(t *testing.T) {
	if _, err := New(Config{Models: []Model{{Key: ""}}}); err == nil {
		t.Error("an empty model key was accepted")
	}
	if _, err := New(Config{Models: []Model{{Key: "a"}, {Key: "a"}}}); err == nil {
		t.Error("a repeated model key was accepted")
	}
	// a page names the model it runs by key: an unknown one is a wiring
	// mistake, and a known one passes through to the page
	runs := func(key string) []Page {
		return []Page{{Name: "p", Choices: []Choice{{Role: "r", Options: []Option{{Name: "m", Key: key}, {Name: "other"}}}}}}
	}
	if _, err := New(Config{Models: []Model{{Key: "a"}}, Pages: runs("b")}); err == nil {
		t.Error("a page running an unknown model was accepted")
	}
	known, err := New(Config{Models: []Model{{Key: "a"}}, Pages: runs("a")})
	if err != nil {
		t.Fatal(err)
	}
	if snap := known.Snapshot(context.Background()); len(snap.Pages) != 1 || len(snap.Pages[0].Choices) != 1 || snap.Pages[0].Choices[0].Options[0].Key != "a" {
		t.Errorf("the page's choices: %+v", snap.Pages)
	}
	svc, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	if snap := svc.Snapshot(context.Background()); len(snap.Models) != 0 || len(snap.Collections) != 0 || snap.Cache != nil {
		t.Errorf("an empty service shows %+v", snap)
	}
}

func TestSnapshot_ShowsLazyStatesWithoutLoading(t *testing.T) {
	f := newFixture(t)
	snap := f.svc.Snapshot(context.Background())
	text := model(t, snap, "text")
	if !text.Lazy || text.State != "not started" || text.ID != sensetest.FakeEmbedderModel || text.Dimensions != 64 || text.RunsOn != "cpu" || text.Error != "" {
		t.Errorf("text model before any use: %+v", text)
	}
	if hash := model(t, snap, "hash"); hash.Lazy || hash.State != "no model" {
		t.Errorf("a model without a loader: %+v", hash)
	}
	if state, _ := f.text.State(); state != sense.LazyNotStarted {
		t.Errorf("the snapshot loaded the model: %v", state)
	}
	if len(snap.Pages) != 1 || len(snap.Setup) != 1 || snap.Started.IsZero() {
		t.Errorf("pages %v, setup %v, started %v", snap.Pages, snap.Setup, snap.Started)
	}
}

func TestLoad_LoadsOnceAndKeepsAFailure(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	st, err := f.svc.Load(ctx, "text")
	if err != nil || st.State != "loaded" || st.Error != "" || st.Recovery != "" {
		t.Fatalf("load: %+v, %v", st, err)
	}
	if st, err := f.svc.Load(ctx, "text"); err != nil || st.State != "loaded" {
		t.Errorf("a second load: %+v, %v", st, err)
	}
	if st, err := f.svc.Load(ctx, "chat"); err != nil || st.State != "loaded" {
		t.Errorf("chat load: %+v, %v", st, err)
	}

	// the failure is the state, not the call's error, and it is kept
	st, err = f.svc.Load(ctx, "image")
	if err != nil || st.State != "failed" || !strings.Contains(st.Error, "model file is missing") || st.Recovery != Recovery {
		t.Fatalf("failed load: %+v, %v", st, err)
	}
	again := model(t, f.svc.Snapshot(ctx), "image")
	if again.State != "failed" || again.Error != st.Error || again.Recovery == "" {
		t.Errorf("the failure was not kept: %+v", again)
	}

	if _, err := f.svc.Load(ctx, "hash"); !errors.Is(err, ErrNotLazy) {
		t.Errorf("load of a model without a loader: %v", err)
	}
	if _, err := f.svc.Load(ctx, "nope"); !errors.Is(err, ErrNotFound) {
		t.Errorf("load of an unknown model: %v", err)
	}
}

func TestLoad_AWaitThatEndsLeavesTheLoadRunning(t *testing.T) {
	release := make(chan struct{})
	slow := sense.NewLazyEmbedder(sensetest.FakeEmbedderModel, 64, func(context.Context) (sense.Embedder, error) {
		<-release
		return sensetest.FakeEmbedder{}, nil
	})
	t.Cleanup(func() { _ = slow.Close() })
	svc, err := New(Config{Models: []Model{{Key: "slow", Lazy: slow}}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	st, err := svc.Load(ctx, "slow")
	if err != nil || st.State != "loading" || !strings.Contains(st.Error, "deadline") || st.LoadMs != 0 {
		t.Fatalf("a load whose wait timed out: %+v, %v", st, err)
	}
	close(release)
	if st, err := svc.Load(context.Background(), "slow"); err != nil || st.State != "loaded" {
		t.Errorf("the load did not go on: %+v, %v", st, err)
	}
}

func TestCollections_RecordedModelCountAndReset(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	c := collection(t, f.svc.Snapshot(ctx), "docs")
	if c.Docs != 0 || c.Recorded != "" || c.Mismatch || !c.CanReset || c.Index != "examples" || c.Collection != "docs" {
		t.Errorf("an empty collection: %+v", c)
	}
	docs := make([]search.Doc, 3)
	for i := range docs {
		docs[i] = search.Doc{ID: fmt.Sprintf("doc-%d", i), Text: fmt.Sprintf("text number %d about bread", i)}
	}
	if _, err := f.index.IndexBatch(ctx, docs); err != nil {
		t.Fatal(err)
	}
	c = collection(t, f.svc.Snapshot(ctx), "docs")
	if c.Docs != 3 || c.Recorded != sensetest.FakeEmbedderModel || c.Mismatch {
		t.Errorf("a filled collection: %+v", c)
	}
	// the page now runs another model than the store recorded
	f.pageModel = "another-model"
	if c = collection(t, f.svc.Snapshot(ctx), "docs"); !c.Mismatch {
		t.Errorf("no mismatch shown: %+v", c)
	}

	st, err := f.svc.Reset(ctx, "docs")
	if err != nil || st.Docs != 0 || st.Recorded != "" || st.Mismatch || f.resets != 1 {
		t.Errorf("reset: %+v, %v, %d resets", st, err, f.resets)
	}
	if _, err := f.svc.Reset(ctx, "fixed"); !errors.Is(err, ErrNoAction) {
		t.Errorf("reset of a collection without one: %v", err)
	}
	if _, err := f.svc.Reset(ctx, "nope"); !errors.Is(err, ErrNotFound) {
		t.Errorf("reset of an unknown collection: %v", err)
	}
}

func TestCollections_CountPagesThroughTheStore(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	docs := make([]search.Doc, docsPage+5)
	for i := range docs {
		docs[i] = search.Doc{ID: fmt.Sprintf("doc-%05d", i), Text: "bread"}
	}
	if _, err := f.index.IndexBatch(ctx, docs); err != nil {
		t.Fatal(err)
	}
	if c := collection(t, f.svc.Snapshot(ctx), "docs"); c.Docs != docsPage+5 {
		t.Errorf("counted %d documents, want %d", c.Docs, docsPage+5)
	}
}

// brokenStore fails every read.
type brokenStore struct{ vector.Store }

func (brokenStore) Model(context.Context, vector.Coord) (string, error) {
	return "", errors.New("the database is gone")
}

type docsBroken struct{ vector.Store }

func (docsBroken) Docs(context.Context, vector.Coord, string, int) ([]vector.DocInfo, error) {
	return nil, errors.New("the listing failed")
}

func TestCollections_AStoreErrorIsShown(t *testing.T) {
	coord := vector.Coord{Index: "examples"}
	svc, err := New(Config{Collections: func() []Collection {
		return []Collection{
			{Key: "broken", Store: brokenStore{vector.NewMemStore()}, Coord: coord},
			{Key: "listing", Store: docsBroken{vector.NewMemStore()}, Coord: coord},
			{Key: "none", Coord: coord},
			{Key: "failing", Store: vector.NewMemStore(), Coord: coord, Reset: func(context.Context) error { return errors.New("locked") }},
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	snap := svc.Snapshot(context.Background())
	for key, want := range map[string]string{"broken": "database is gone", "listing": "listing failed", "none": "no store"} {
		if c := collection(t, snap, key); !strings.Contains(c.Error, want) {
			t.Errorf("%s: error %q, want %q in it", key, c.Error, want)
		}
	}
	if _, err := svc.Reset(context.Background(), "failing"); err == nil || !strings.Contains(err.Error(), "locked") {
		t.Errorf("a failing reset: %v", err)
	}
}

func TestCache_SizeAndPrune(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	c := f.svc.Snapshot(ctx).Cache
	if c == nil || c.Bytes != 4096 || c.LastPrune != nil || !c.CanPrune {
		t.Fatalf("cache before a prune: %+v", c)
	}
	st, err := f.svc.Prune(ctx)
	if err != nil || st.LastPrune == nil || st.LastPrune.Removed != 3 || st.LastPrune.UnusedForMs != (30*24*time.Hour).Milliseconds() || f.pruned != 1 {
		t.Errorf("prune: %+v, %v", st, err)
	}
}

func TestCache_FailuresAndMissingActions(t *testing.T) {
	ctx := context.Background()
	// a cache whose size and prune fail
	failing, err := New(Config{Cache: &Cache{
		Size:      func(context.Context) (int64, error) { return 0, errors.New("no such file") },
		LastPrune: func() Prune { return Prune{At: time.Now(), Err: errors.New("cutoff too recent")} },
		PruneNow:  func(context.Context) (Prune, error) { return Prune{}, errors.New("closed") },
	}})
	if err != nil {
		t.Fatal(err)
	}
	c := failing.Snapshot(ctx).Cache
	if c.Bytes != -1 || c.SizeError != "no such file" || c.LastPrune == nil || c.LastPrune.Error != "cutoff too recent" {
		t.Errorf("a failing cache: %+v", c)
	}
	if _, err := failing.Prune(ctx); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Errorf("a failing prune: %v", err)
	}
	// and one that was given neither
	bare, err := New(Config{Cache: &Cache{Name: "cache"}})
	if err != nil {
		t.Fatal(err)
	}
	if c := bare.Snapshot(ctx).Cache; c.Bytes != -1 || c.CanPrune || c.LastPrune != nil {
		t.Errorf("a bare cache: %+v", c)
	}
	if _, err := bare.Prune(ctx); !errors.Is(err, ErrNoAction) {
		t.Errorf("prune without one: %v", err)
	}
}

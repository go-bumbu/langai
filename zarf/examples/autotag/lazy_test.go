package autotag

import (
	"context"
	"errors"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/tagging"
)

// lazyNotebook is a notebook over the library's lazy loaders, each loading a
// fake, with the states the page shows wired as the service wires them.
func lazyNotebook(t *testing.T, cache tagging.EmbeddingCache, file string, loads *atomic.Int64, loadErr error) (*Notebook, *sense.LazyEmbedder) {
	t.Helper()
	emb := sense.NewLazyEmbedder(sensetest.FakeEmbedderModel, 64, func(context.Context) (sense.Embedder, error) {
		loads.Add(1)
		if loadErr != nil {
			return nil, loadErr
		}
		return sensetest.FakeEmbedder{}, nil
	})
	chat := sense.NewLazyCompleter(func(context.Context) (sense.Completer, error) {
		return &sensetest.FakeCompleter{Replies: []string{flatReply}}, nil
	})
	state := func(l interface {
		State() (sense.LazyState, error)
	}) func() string {
		return func() string { s, _ := l.State(); return s.String() }
	}
	nb, err := New(context.Background(), Config{
		Cache:         cache,
		Tagging:       tagging.TaggerConfig{Model: emb.Model(), NameMinScore: 0.2, MemberMinScore: 0.2},
		Chat:          chat,
		Embedder:      emb,
		File:          file,
		ChatState:     state(chat),
		EmbedderState: state(emb),
		RunsOn:        func() string { return "a fake" },
		Info:          Info{MinScore: 0.3, Accel: "configured"},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = nb.Close(); _ = emb.Close(); _ = chat.Close() })
	return nb, emb
}

func TestNotebook_ALazyEmbedderLoadsOnTheFirstNewText(t *testing.T) {
	ctx := context.Background()
	cache := tagging.NewMemCache()
	file := filepath.Join(t.TempDir(), "notebook.json")
	var loads atomic.Int64

	// an empty notebook embeds nothing: the model stays unloaded
	nb, _ := lazyNotebook(t, cache, file, &loads, nil)
	if loads.Load() != 0 || nb.EmbedderState() != "not started" || nb.ChatState() != "not started" || nb.RunsOn() != "a fake" {
		t.Fatalf("after opening an empty notebook: %d loads, embedder %q, chat %q", loads.Load(), nb.EmbedderState(), nb.ChatState())
	}
	if _, err := nb.AddTag(ctx, "cooking / bread", false); err != nil {
		t.Fatal(err)
	}
	note, err := nb.AddNote(ctx, "Sourdough", "Baking bread with a sourdough starter")
	if err != nil {
		t.Fatal(err)
	}
	if v := noteView(t, nb, note.ID); v.Error != "" || len(v.Auto) != 1 {
		t.Errorf("view after the first note: %+v", v)
	}
	if loads.Load() != 1 || nb.EmbedderState() != "loaded" {
		t.Errorf("after the first text: %d loads, embedder %q", loads.Load(), nb.EmbedderState())
	}
	// the chat model loads on the first recommendation
	if _, err := nb.Recommend(ctx, note.ID, RecommendOptions{}); err != nil || nb.ChatState() != "loaded" {
		t.Errorf("recommend: %v, chat %q", err, nb.ChatState())
	}

	// a restart over the same cache finds every embedding: nothing loads
	var again atomic.Int64
	nb2, _ := lazyNotebook(t, cache, file, &again, nil)
	if v := noteView(t, nb2, note.ID); v.Error != "" || len(v.Auto) != 1 {
		t.Errorf("view after the restart: %+v", v)
	}
	if again.Load() != 0 || nb2.EmbedderState() != "not started" {
		t.Errorf("the restart loaded the model: %d loads, embedder %q", again.Load(), nb2.EmbedderState())
	}
}

func TestNotebook_OpensWithAnEmbedderThatCannotLoad(t *testing.T) {
	ctx := context.Background()
	file := filepath.Join(t.TempDir(), "notebook.json")
	var loads atomic.Int64
	seed, _ := lazyNotebook(t, tagging.NewMemCache(), file, &loads, nil)
	note, err := seed.AddNote(ctx, "Sourdough", "Baking bread with a sourdough starter")
	if err != nil {
		t.Fatal(err)
	}

	// a fresh cache has to embed the note, and the model does not load: the
	// notebook opens all the same, and the note says why it has no tags
	var failed atomic.Int64
	nb, emb := lazyNotebook(t, tagging.NewMemCache(), file, &failed, errors.New("the model file is missing"))
	if state, err := emb.State(); state != sense.LazyFailed || err == nil {
		t.Fatalf("the embedder after a failed load: %v, %v", state, err)
	}
	v := noteView(t, nb, note.ID)
	if !strings.Contains(v.Error, "the model file is missing") || len(v.Auto) != 0 {
		t.Errorf("view with the embedder down: %+v", v)
	}
	// the failure is kept: one attempt, not one per call
	if failed.Load() != 1 || nb.EmbedderState() != "failed" {
		t.Errorf("%d load attempts, embedder %q", failed.Load(), nb.EmbedderState())
	}
}

func TestNew_ACancelledOpenIsAnError(t *testing.T) {
	file := filepath.Join(t.TempDir(), "notebook.json")
	var loads atomic.Int64
	seed, _ := lazyNotebook(t, tagging.NewMemCache(), file, &loads, nil)
	if _, err := seed.AddNote(context.Background(), "", "a note to embed"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := New(ctx, Config{
		Cache: tagging.NewMemCache(), Tagging: tagging.TaggerConfig{Model: "fake"},
		Chat: &sensetest.FakeCompleter{}, Embedder: sensetest.FakeEmbedder{}, File: file,
	})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("open under a cancelled context: %v", err)
	}
}

func TestHandler_InfoShowsTheModelStates(t *testing.T) {
	var loads atomic.Int64
	nb, _ := lazyNotebook(t, tagging.NewMemCache(), "", &loads, nil)
	srv := httptest.NewServer(Handler(nb))
	t.Cleanup(srv.Close)
	status, body := do(t, "GET", srv.URL+"/api/info", "")
	var info struct {
		Chat     string `json:"chat"`
		Embedder string `json:"embedder"`
		RunsOn   string `json:"runsOn"`
	}
	decode(t, body, &info)
	if status != 200 || info.Chat != "not started" || info.Embedder != "not started" || info.RunsOn != "a fake" {
		t.Errorf("info: %d %+v", status, info)
	}
	// a notebook that was not told shows what it was configured with
	plain := newNotebook(t, "")
	if plain.EmbedderState() != "" || plain.ChatState() != "" || plain.RunsOn() != "" {
		t.Errorf("states of a notebook without them: %q %q %q", plain.EmbedderState(), plain.ChatState(), plain.RunsOn())
	}
}

// TestNew_WarmLaterLeavesTheWarmUpToTheCaller: a notebook whose notes a
// fresh cache does not hold opens without embedding anything, and the
// caller's Warm embeds them.
func TestNew_WarmLaterLeavesTheWarmUpToTheCaller(t *testing.T) {
	ctx := context.Background()
	file := filepath.Join(t.TempDir(), "notebook.json")
	var loads atomic.Int64
	seed, _ := lazyNotebook(t, tagging.NewMemCache(), file, &loads, nil)
	if _, err := seed.AddNote(ctx, "Sourdough", "Baking bread with a sourdough starter"); err != nil {
		t.Fatal(err)
	}

	var calls atomic.Int64
	nb, err := New(ctx, Config{
		Cache: tagging.NewMemCache(), Tagging: tagging.TaggerConfig{Model: "fake"},
		Chat: &sensetest.FakeCompleter{}, Embedder: sensetest.FakeEmbedder{Calls: &calls}, File: file,
		WarmLater: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = nb.Close() })
	if calls.Load() != 0 {
		t.Errorf("New embedded %d times with WarmLater", calls.Load())
	}
	if st, err := nb.Warm(ctx); err != nil || st.Notes != 1 || calls.Load() == 0 {
		t.Errorf("Warm = %+v, %v after %d embedder calls", st, err, calls.Load())
	}
}

package autotag

import (
	"context"
	"errors"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/tagging"
)

// poisonedNotebook is a notebook whose embedder refuses every text that
// holds the word "poison", with a tag put by hand on a note it takes and on
// one it refuses. calls counts the embedder's calls.
func poisonedNotebook(t *testing.T) (nb *Notebook, calls *atomic.Int64, tag Tag, good, bad Note) {
	t.Helper()
	ctx := context.Background()
	calls = &atomic.Int64{}
	emb := langaitest.FakeEmbedder{Calls: calls, Fail: func(text string) error {
		if strings.Contains(text, "poison") {
			return errors.New("the model cannot take this text")
		}
		return nil
	}}
	nb, err := New(ctx, Config{
		Cache:    tagging.NewMemCache(),
		Tagging:  tagging.TaggerConfig{Model: "fake", NameMinScore: 0.2, MemberMinScore: 0.2},
		Chat:     &langaitest.FakeCompleter{},
		Embedder: emb,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = nb.Close() })
	if tag, err = nb.AddTag(ctx, "cooking / bread", false); err != nil {
		t.Fatal(err)
	}
	if good, err = nb.AddNote(ctx, "Sourdough", "bread dough with a starter, baked in a hot oven"); err != nil {
		t.Fatal(err)
	}
	if bad, err = nb.AddNote(ctx, "Mushrooms", "a poison warning for bread baked with foraged mushrooms"); err != nil {
		t.Fatal(err)
	}
	for _, n := range []Note{good, bad} {
		if err := nb.TagNote(ctx, n.ID, tag.ID); err != nil {
			t.Fatal(err)
		}
	}
	return nb, calls, tag, good, bad
}

// TestNotebook_ViewMarksATagBuiltWithoutAMember: a note the embedder cannot
// take is left out of its tag, which still tags by its name and its other
// note; the view says so on the tag, and on the note itself.
func TestNotebook_ViewMarksATagBuiltWithoutAMember(t *testing.T) {
	ctx := context.Background()
	nb, _, tag, good, bad := poisonedNotebook(t)
	v, err := nb.View(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(v.Tags) != 1 || !v.Tags[0].Incomplete || v.Tags[0].Hand != 2 {
		t.Errorf("tag view = %+v; want the tag marked incomplete", v.Tags)
	}
	if n := noteView(t, nb, good.ID); n.Error != "" || len(n.Auto) != 1 || n.Auto[0] != tag.ID {
		t.Errorf("the note the embedder takes = %+v; want it tagged", n)
	}
	if n := noteView(t, nb, bad.ID); !strings.Contains(n.Error, "cannot take this text") {
		t.Errorf("the note the embedder refuses = %+v; want its error", n)
	}

	// once the note is gone, the tag is whole again
	if err := nb.DeleteNote(ctx, bad.ID); err != nil {
		t.Fatal(err)
	}
	if v, err = nb.View(ctx); err != nil || v.Tags[0].Incomplete {
		t.Errorf("tag view after the note left = %+v, %v; want it complete", v.Tags, err)
	}
}

// TestNotebook_WarmEmbedsOnlyWhatTheCacheLacks: a Warm over a full cache
// calls the embedder for nothing; with a note it cannot take it fails,
// naming the note by its place, and still lists the tag left incomplete.
func TestNotebook_WarmEmbedsOnlyWhatTheCacheLacks(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	if _, err := nb.AddTag(ctx, "cooking / bread", false); err != nil {
		t.Fatal(err)
	}
	if _, err := nb.AddNote(ctx, "Sourdough", "bread dough with a starter"); err != nil {
		t.Fatal(err)
	}
	st, err := nb.Warm(ctx)
	if err != nil || st.Notes != 1 || st.Tags != 1 || st.Revision == "" || len(st.Incomplete) != 0 {
		t.Fatalf("Warm = %+v, %v", st, err)
	}

	poisoned, calls, tag, _, _ := poisonedNotebook(t)
	st, err = poisoned.Warm(ctx)
	if err == nil || !strings.Contains(err.Error(), "text 2 of 2") {
		t.Errorf("Warm with a note the embedder refuses: %v; want it named by its place", err)
	}
	if st.Notes != 2 || len(st.Incomplete) != 1 || st.Incomplete[0] != tag.ID {
		t.Errorf("Warm stats = %+v; want the tag listed as incomplete", st)
	}
	// what the cache holds is not embedded again: only the refused note is tried
	before := calls.Load()
	if _, err := poisoned.Warm(ctx); err == nil {
		t.Error("a second Warm took the refused note")
	}
	if n := calls.Load() - before; n > 2 {
		t.Errorf("a second Warm called the embedder %d times; want the refused note alone", n)
	}
}

func TestHandler_Warm(t *testing.T) {
	srv := serve(t)
	status, body := do(t, "POST", srv.URL+"/api/warm", "")
	var st WarmStats
	decode(t, body, &st)
	if status != 200 || st.Notes != 0 || st.Incomplete == nil {
		t.Errorf("warm: %d %s", status, body)
	}

	poisoned, _, _, _, _ := poisonedNotebook(t)
	bad := httptest.NewServer(Handler(poisoned))
	t.Cleanup(bad.Close)
	status, body = do(t, "POST", bad.URL+"/api/warm", "")
	var failed struct {
		Error string    `json:"error"`
		Stats WarmStats `json:"stats"`
	}
	decode(t, body, &failed)
	if status != 500 || failed.Error == "" || len(failed.Stats.Incomplete) != 1 {
		t.Errorf("warm with a note the embedder refuses: %d %s", status, body)
	}
}

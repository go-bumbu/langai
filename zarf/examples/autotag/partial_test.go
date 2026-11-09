package autotag

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/tagging"
)

// markedModel writes the same tags for every prompt but the ones that hold
// its mark, which it fails on.
type markedModel struct{ mark string }

func (m markedModel) Complete(_ context.Context, _, user string) (string, error) {
	if strings.Contains(user, m.mark) {
		return "", errors.New("model down")
	}
	return "bread\nbaking\n", nil
}

// partialNotebook is a notebook with one long note, read in several windows,
// whose last words are "the end".
func partialNotebook(t *testing.T, chat markedModel) (*Notebook, Note) {
	t.Helper()
	ctx := context.Background()
	nb, err := New(ctx, Config{
		Cache:    tagging.NewMemCache(),
		Tagging:  tagging.TaggerConfig{Model: "fake", NameMinScore: 0.2, MemberMinScore: 0.2},
		Chat:     chat,
		Embedder: sensetest.FakeEmbedder{},
		Info:     Info{MinScore: 0.3},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = nb.Close() })
	text := strings.Repeat("Baking bread at home takes flour, water, salt and time. ", 300) + "the end"
	n, err := nb.AddNote(ctx, "Bread", text)
	if err != nil {
		t.Fatal(err)
	}
	return nb, n
}

// TestNotebook_RecommendKeepsTheTagsOfAPartlyReadNote: a note of 16 KB is read
// in three windows. When the model fails on one of them the tags of the others
// are shown with what went wrong; only when it fails on all is there nothing
// to show.
func TestNotebook_RecommendKeepsTheTagsOfAPartlyReadNote(t *testing.T) {
	cases := []struct {
		name      string
		mark      string // the model fails on the windows holding it
		wantErr   bool
		wantTags  int
		wantError string // in Recommendation.Error
	}{
		{"every window read", "no window holds this", false, 2, ""},
		{"the last window fails", "the end", false, 2, "1 of 3 windows"},
		{"every window fails", "Baking bread", true, 0, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			nb, n := partialNotebook(t, markedModel{mark: c.mark})
			res, err := nb.Recommend(context.Background(), n.ID, RecommendOptions{})
			if (err != nil) != c.wantErr {
				t.Fatalf("Recommend error = %v, want an error: %v", err, c.wantErr)
			}
			if len(res.Tags) != c.wantTags {
				t.Errorf("got %d tags %+v, want %d", len(res.Tags), res.Tags, c.wantTags)
			}
			if c.wantError == "" && res.Error != "" {
				t.Errorf("Error = %q, want none", res.Error)
			}
			if c.wantError != "" && (!strings.Contains(res.Error, c.wantError) || !strings.Contains(res.Error, tagging.ErrPartial.Error())) {
				t.Errorf("Error = %q, want it to say it is partial and name %q", res.Error, c.wantError)
			}
		})
	}
}

// TestNotebook_CloseStopsTheTagger: after Close the Tagger answers nothing,
// which a View reports note by note, and closing again is fine.
func TestNotebook_CloseStopsTheTagger(t *testing.T) {
	nb, _ := partialNotebook(t, markedModel{mark: "no window holds this"})
	scoreError := func() string {
		t.Helper()
		v, err := nb.View(context.Background())
		if err != nil || len(v.Notes) != 1 {
			t.Fatalf("View = %d notes, %v; want the one note", len(v.Notes), err)
		}
		return v.Notes[0].Error
	}
	if msg := scoreError(); msg != "" {
		t.Fatalf("before Close the note's scores fail: %s", msg)
	}
	for range 2 {
		if err := nb.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}
	if msg := scoreError(); !strings.Contains(msg, sense.ErrClosed.Error()) {
		t.Errorf("after Close the note's error is %q, want %q", msg, sense.ErrClosed)
	}
}

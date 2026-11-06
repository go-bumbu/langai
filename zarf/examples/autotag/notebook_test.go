package autotag

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/tagging"
)

// cannedModel answers a prompt for flat tags with flat and one for tree tags
// with tree, and keeps the last system prompt.
type cannedModel struct {
	flat, tree string
	system     string
}

func (m *cannedModel) Complete(_ context.Context, system, _ string) (string, error) {
	m.system = system
	if strings.Contains(system, `separated by " / "`) {
		return m.tree, nil
	}
	return m.flat, nil
}

// newNotebook builds a notebook over a fresh MemCache. FakeEmbedder only
// counts shared words, so its scores run far below a real model's, and the
// cutoffs are lowered to match.
func newNotebook(t *testing.T, file string) *Notebook {
	nb, _ := newNotebookModel(t, file)
	return nb
}

func newNotebookModel(t *testing.T, file string) (*Notebook, *cannedModel) {
	t.Helper()
	ctx := context.Background()
	emb := langaitest.FakeEmbedder{}
	model := &cannedModel{flat: "bread\nbaking\ntravel\n", tree: "cooking / baking / sourdough\ncooking / bread\ntravel / city guides\n"}
	nb, err := New(ctx, Config{
		Cache:       tagging.NewMemCache(),
		Tagging:     tagging.TaggerConfig{Model: "fake", NameMinScore: 0.2, MemberMinScore: 0.2},
		Chat:        model,
		Embedder:    emb,
		File:        file,
		SamplesFile: filepath.Join("..", "..", "eval", "testdata", "golden.yaml"),
		Info:        Info{MinScore: 0.3},
	})
	if err != nil {
		t.Fatal(err)
	}
	return nb, model
}

func noteView(t *testing.T, nb *Notebook, id string) NoteView {
	t.Helper()
	v, err := nb.View(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range v.Notes {
		if n.ID == id {
			return n
		}
	}
	t.Fatalf("no note %q in the view", id)
	return NoteView{}
}

func TestNotebook_SuggestsExistingTagsAndLearnsFromHandTags(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	bread, err := nb.AddTag(ctx, "cooking / bread", false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nb.AddTag(ctx, "travel / hiking", false); err != nil {
		t.Fatal(err)
	}
	n, err := nb.AddNote(ctx, "Sourdough", "bread dough with a starter, baked bread in a hot oven")
	if err != nil {
		t.Fatal(err)
	}
	if got := noteView(t, nb, n.ID).Auto; !slices.Equal(got, []string{bread.ID}) {
		t.Errorf("auto tags by name: got %v, want [%s]", got, bread.ID)
	}
	if err := nb.TagNote(ctx, n.ID, bread.ID); err != nil {
		t.Fatal(err)
	}
	v, err := nb.View(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if v.Tags[0].ID != bread.ID || v.Tags[0].Hand != 1 || v.Tags[0].Auto != 1 {
		t.Errorf("tag counts: %+v", v.Tags[0])
	}
	if !slices.Equal(v.Notes[0].Tags, []string{bread.ID}) {
		t.Errorf("hand tags: %v", v.Notes[0].Tags)
	}
	if err := nb.UntagNote(ctx, n.ID, bread.ID); err != nil {
		t.Fatal(err)
	}
	if got := noteView(t, nb, n.ID).Tags; len(got) != 0 {
		t.Errorf("hand tags after untag: %v", got)
	}
}

func TestNotebook_AddTagReusesANameOrLeaf(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	a, _ := nb.AddTag(ctx, "Cooking / Baking", false)
	for _, name := range []string{"cooking / baking", "baking", " BAKING "} {
		b, err := nb.AddTag(ctx, name, false)
		if err != nil || b.ID != a.ID {
			t.Errorf("AddTag(%q) = %+v, %v; want the existing %s", name, b, err, a.ID)
		}
	}
	if a.ID != "cooking-baking" {
		t.Errorf("id: %q", a.ID)
	}
	if _, err := nb.AddTag(ctx, "  ", false); !errors.Is(err, ErrBadInput) {
		t.Errorf("blank name: %v", err)
	}
}

func TestNotebook_RenameAndDeleteTag(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	tag, _ := nb.AddTag(ctx, "bread", false)
	other, _ := nb.AddTag(ctx, "travel", false)
	n, _ := nb.AddNote(ctx, "", "bread")
	_ = nb.TagNote(ctx, n.ID, tag.ID)
	if _, err := nb.RenameTag(ctx, tag.ID, "travel"); !errors.Is(err, ErrBadInput) {
		t.Errorf("rename onto another tag's name: %v", err)
	}
	if got, err := nb.RenameTag(ctx, tag.ID, "cooking / bread"); err != nil || got.Name != "cooking / bread" || got.ID != tag.ID {
		t.Errorf("rename: %+v, %v", got, err)
	}
	if err := nb.DeleteTag(ctx, tag.ID); err != nil {
		t.Fatal(err)
	}
	v, _ := nb.View(ctx)
	if len(v.Tags) != 1 || v.Tags[0].ID != other.ID || len(v.Notes[0].Tags) != 0 {
		t.Errorf("after delete: tags %+v, note tags %v", v.Tags, v.Notes[0].Tags)
	}
	if err := nb.DeleteTag(ctx, tag.ID); !errors.Is(err, ErrNotFound) {
		t.Errorf("delete twice: %v", err)
	}
}

func TestNotebook_UnknownIDs(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	tag, _ := nb.AddTag(ctx, "bread", false)
	n, _ := nb.AddNote(ctx, "", "bread")
	for name, err := range map[string]error{
		"tag unknown note": nb.TagNote(ctx, "nope", tag.ID),
		"tag unknown tag":  nb.TagNote(ctx, n.ID, "nope"),
		"untag unknown":    nb.UntagNote(ctx, "nope", tag.ID),
		"delete unknown":   nb.DeleteNote(ctx, "nope"),
	} {
		if !errors.Is(err, ErrNotFound) {
			t.Errorf("%s: %v", name, err)
		}
	}
	if _, err := nb.Recommend(ctx, "nope", RecommendOptions{}); !errors.Is(err, ErrNotFound) {
		t.Errorf("recommend unknown: %v", err)
	}
	if _, err := nb.AddNote(ctx, "title", " "); !errors.Is(err, ErrBadInput) {
		t.Errorf("blank note: %v", err)
	}
}

func TestNotebook_RecommendScoresEveryTagAndMatchesExistingOnes(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	baking, _ := nb.AddTag(ctx, "cooking / baking", false)
	n, _ := nb.AddNote(ctx, "", "baking bread at home: bread, flour and baking")
	res, err := nb.Recommend(ctx, n.ID, RecommendOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if res.MinScore != 0.3 || len(res.Tags) != 3 {
		t.Fatalf("got %+v", res)
	}
	by := map[string]Recommended{}
	for i, r := range res.Tags {
		by[r.Name] = r
		if i > 0 && r.Score > res.Tags[i-1].Score {
			t.Errorf("not best first: %+v", res.Tags)
		}
		if r.Above != (r.Score >= res.MinScore) {
			t.Errorf("%q: above %v at %.3f", r.Name, r.Above, r.Score)
		}
	}
	if by["baking"].TagID != baking.ID || by["bread"].TagID != "" {
		t.Errorf("existing tags: %+v", by)
	}
	if by["travel"].Above || !by["bread"].Above {
		t.Errorf("cutoff: %+v", by)
	}
}

func TestNotebook_RecommendOptionsReachTheModel(t *testing.T) {
	ctx := context.Background()
	nb, model := newNotebookModel(t, "")
	_, _ = nb.AddTag(ctx, "cooking / bread", false)
	_, _ = nb.AddTag(ctx, "cooking", false)
	n, _ := nb.AddNote(ctx, "", "baking sourdough bread in a city kitchen")

	if _, err := nb.Recommend(ctx, n.ID, RecommendOptions{Language: "es"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(model.system, "in Spanish (es)") || strings.Contains(model.system, " / ") {
		t.Errorf("Spanish, flat:\n%s", model.system)
	}

	res, err := nb.Recommend(ctx, n.ID, RecommendOptions{Tree: true})
	if err != nil {
		t.Fatal(err)
	}
	// the notebook's tags, sorted, are the tree to extend
	if !strings.Contains(model.system, "one per line:\ncooking\ncooking / bread\n") || !strings.Contains(model.system, "language of the text") {
		t.Errorf("tree prompt:\n%s", model.system)
	}
	known := map[string]int{}
	ids := map[string]string{}
	for _, r := range res.Tags {
		known[r.Name], ids[r.Name] = r.Known, r.TagID
	}
	want := map[string]int{"cooking / baking / sourdough": 1, "cooking / bread": 2, "travel / city guides": 0}
	for name, k := range want {
		if got, ok := known[name]; !ok || got != k {
			t.Errorf("%q: known %d (present %v), want %d", name, got, ok, k)
		}
	}
	if ids["cooking / bread"] != "cooking-bread" || ids["cooking / baking / sourdough"] != "" {
		t.Errorf("tag ids: %v", ids)
	}
	if !res.Options.Tree {
		t.Errorf("options not echoed: %+v", res.Options)
	}

	if _, err := nb.Recommend(ctx, n.ID, RecommendOptions{Language: "not a language!"}); !errors.Is(err, ErrBadInput) {
		t.Errorf("bad language: %v", err)
	}
}

func TestNotebook_AddTagWithParentsMakesTheBranch(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	_, _ = nb.AddTag(ctx, "Cooking", false)
	got, err := nb.AddTag(ctx, "cooking/baking / sourdough", true)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := nb.View(ctx)
	var names []string
	for _, tg := range v.Tags {
		names = append(names, tg.Name)
	}
	// "Cooking" is reused, whatever its case; the rest of the branch is new
	if !slices.Equal(names, []string{"Cooking", "Cooking / baking", "Cooking / baking / sourdough"}) || got.Name != "Cooking / baking / sourdough" {
		t.Errorf("tags %q, returned %+v", names, got)
	}
	// a path matches only itself, never by its leaf
	_, _ = nb.AddTag(ctx, "errors", false)
	if p, _ := nb.AddTag(ctx, "dev / errors", false); p.Name != "dev / errors" {
		t.Errorf("path matched by leaf: %+v", p)
	}
	// nor does a level of a new branch: the root "web" is not "dev / web"
	_, _ = nb.AddTag(ctx, "dev / web", false)
	if p, _ := nb.AddTag(ctx, "web / agentic", true); p.Name != "web / agentic" {
		t.Errorf("branch grafted by leaf: %+v", p)
	}
}

func TestNotebook_RecommendKnowsTheDeepestExistingLevel(t *testing.T) {
	ctx := context.Background()
	nb, model := newNotebookModel(t, "")
	model.tree = "events / hackathons / basel\nweb / agentic\n"
	_, _ = nb.AddTag(ctx, "events / hackathons", false) // no "events" tag
	_, _ = nb.AddTag(ctx, "dev / web", false)
	n, _ := nb.AddNote(ctx, "", "a hackathon in basel about the agentic web")
	res, err := nb.Recommend(ctx, n.ID, RecommendOptions{Tree: true})
	if err != nil {
		t.Fatal(err)
	}
	known := map[string]int{}
	for _, r := range res.Tags {
		known[r.Name] = r.Known
	}
	if known["events / hackathons / basel"] != 2 || known["web / agentic"] != 0 {
		t.Errorf("known levels: %v", known)
	}
}

func TestNotebook_PersistsAndRestartsOnAFreshCache(t *testing.T) {
	ctx := context.Background()
	file := filepath.Join(t.TempDir(), "notebook.json")
	nb := newNotebook(t, file)
	tag, _ := nb.AddTag(ctx, "cooking / bread", false)
	n, _ := nb.AddNote(ctx, "Loaf", "bread")
	_ = nb.TagNote(ctx, n.ID, tag.ID)
	gone, _ := nb.AddNote(ctx, "", "deleted")
	_ = nb.DeleteNote(ctx, gone.ID)

	// a new notebook over an empty cache: the file is the truth
	again := newNotebook(t, file)
	got := noteView(t, again, n.ID)
	if got.Title != "Loaf" || !slices.Equal(got.Tags, []string{tag.ID}) || !slices.Equal(got.Auto, []string{tag.ID}) {
		t.Errorf("restarted note: %+v", got)
	}
	v, _ := again.View(ctx)
	if len(v.Notes) != 1 || v.Tags[0].Hand != 1 {
		t.Errorf("restarted view: %+v", v)
	}
	// ids keep counting past the deleted note
	if next, _ := again.AddNote(ctx, "", "more"); next.ID == n.ID || next.ID == gone.ID {
		t.Errorf("reused id %q", next.ID)
	}
}

func TestNotebook_LoadSamples(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	st, err := nb.LoadSamples(ctx, 0)
	if err != nil {
		t.Fatal(err)
	}
	if st.Tags != 15 || st.Notes != 84 || st.HandTags != 0 {
		t.Errorf("imported %+v, want 15 tags, 84 notes and no hand tags", st)
	}
	v, _ := nb.View(ctx)
	n := v.Notes[0]
	if n.ID != "en-go-modules" || !slices.Equal(n.Expected, []string{"dev / go"}) || len(n.Tags) != 0 || n.Title == "" {
		t.Errorf("first sample: %+v", n.Note)
	}
	if st, err := nb.LoadSamples(ctx, 0); err != nil || st.Tags != 0 || st.Notes != 0 {
		t.Errorf("second import: %+v, %v", st, err)
	}
	if _, err := nb.LoadSamples(ctx, -1); !errors.Is(err, ErrBadInput) {
		t.Errorf("negative per tag: %v", err)
	}
}

// handCounts maps each tag name to the notes it is on by hand.
func handCounts(t *testing.T, nb *Notebook) map[string]int {
	t.Helper()
	v, err := nb.View(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]int{}
	for _, tag := range v.Tags {
		got[tag.Name] = tag.Hand
	}
	return got
}

func TestNotebook_LoadSamplesPartlyTagged(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	st, err := nb.LoadSamples(ctx, 2)
	if err != nil {
		t.Fatal(err)
	}
	if st.Notes != 84 || st.HandTags == 0 {
		t.Fatalf("imported %+v", st)
	}
	n := handCounts(t, nb)
	total := 0
	for name, c := range n {
		if c < 1 || c > 2 {
			t.Errorf("%q is on %d notes by hand, want 1 or 2", name, c)
		}
		total += c
	}
	if total != st.HandTags {
		t.Errorf("hand tags: %d in the view, %d reported", total, st.HandTags)
	}
	// only expected tags go on by hand
	v, _ := nb.View(ctx)
	for _, note := range v.Notes {
		for _, id := range note.Tags {
			name := id
			for _, tag := range v.Tags {
				if tag.ID == id {
					name = tag.Name
				}
			}
			if !slices.Contains(note.Expected, name) {
				t.Errorf("%s got %q by hand, expects %v", note.ID, name, note.Expected)
			}
		}
	}
	// again: nothing new
	if st, err := nb.LoadSamples(ctx, 2); err != nil || st.Notes != 0 || st.HandTags != 0 {
		t.Errorf("second import: %+v, %v", st, err)
	}
}

func TestNotebook_LoadSamplesTagsAnEarlierImport(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	if _, err := nb.LoadSamples(ctx, 0); err != nil {
		t.Fatal(err)
	}
	st, err := nb.LoadSamples(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if st.Notes != 0 || st.HandTags != 15 {
		t.Errorf("training the baseline: %+v, want no new notes and one hand tag per tag", st)
	}
}

func TestNotebook_ViewScoresEveryTagAndCountsParts(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	nb.cfg.Info.ChunkBytes = 100
	bread, _ := nb.AddTag(ctx, "cooking / bread", false)
	tax, _ := nb.AddTag(ctx, "finance / taxes", false)
	long, _ := nb.AddNote(ctx, "Loaf", strings.Repeat("bread dough and a hot oven ", 8))
	short, _ := nb.AddNote(ctx, "", "bread")
	got := noteView(t, nb, long.ID)
	if len(got.Scores) != 2 || got.Scores[0].TagID != bread.ID || got.Scores[1].TagID != tax.ID {
		t.Fatalf("scores: %+v", got.Scores)
	}
	for _, s := range got.Scores {
		if picked := slices.Contains(got.Auto, s.TagID); picked != (s.Score >= s.Cutoff && s.ShadowedBy == "") {
			t.Errorf("%s: score %.3f, cutoff %.3f, picked %v", s.TagID, s.Score, s.Cutoff, picked)
		}
	}
	if got.Bytes != len(itemText(long)) || got.Parts != 3 {
		t.Errorf("long note: %d bytes in %d parts", got.Bytes, got.Parts)
	}
	if got := noteView(t, nb, short.ID); got.Parts != 1 {
		t.Errorf("short note: %d parts", got.Parts)
	}
}

func TestSlug(t *testing.T) {
	for in, want := range map[string]string{
		"cooking / baking": "cooking-baking",
		"  Go_Modules!  ":  "go-modules",
		"Reisen / Städte":  "reisen-städte",
		"///":              "",
		"a--b":             "a-b",
	} {
		if got := slug(in); got != want {
			t.Errorf("slug(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestNotebook_ViewSaysWhichTagShadowsAnother: "cooking" and "cooking /
// bread" both reach their cutoffs on a note about bread, which fits the child
// better; the view picks the child and says the parent is shadowed by it.
func TestNotebook_ViewSaysWhichTagShadowsAnother(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	cooking, _ := nb.AddTag(ctx, "cooking", false)
	bread, _ := nb.AddTag(ctx, "cooking / bread", false)
	note, _ := nb.AddNote(ctx, "", "cooking bread, bread and more bread")
	got := noteView(t, nb, note.ID)
	if !slices.Equal(got.Auto, []string{bread.ID}) {
		t.Fatalf("auto = %v, want only %s; scores %+v", got.Auto, bread.ID, got.Scores)
	}
	for _, s := range got.Scores {
		if s.TagID == cooking.ID && (s.Score < s.Cutoff || s.ShadowedBy != bread.ID) {
			t.Errorf("cooking: %+v, want it over its cutoff and shadowed by %s", s, bread.ID)
		}
	}
}

// TestNotebook_HandTagChangesSuggestionsAtOnce: a croissant note shares no
// word with the tag "cooking / bread", until croissant recipes are tagged
// with it by hand; the next View learns from those members, with no restart,
// and forgets them as soon as they are untagged.
func TestNotebook_HandTagChangesSuggestionsAtOnce(t *testing.T) {
	ctx := context.Background()
	nb := newNotebook(t, "")
	bread, _ := nb.AddTag(ctx, "cooking / bread", false)
	var recipes []Note
	for _, text := range []string{
		"Croissant dough: folding cold butter into layers",
		"Butter croissant dough, step by step",
		"Shaping croissant dough with butter",
	} {
		n, _ := nb.AddNote(ctx, "", text)
		recipes = append(recipes, n)
	}
	target, _ := nb.AddNote(ctx, "", "Croissant dough with cold butter")
	if got := noteView(t, nb, target.ID).Auto; len(got) != 0 {
		t.Fatalf("auto tags before the hand tags: %v, want none", got)
	}
	for _, r := range recipes {
		if err := nb.TagNote(ctx, r.ID, bread.ID); err != nil {
			t.Fatal(err)
		}
	}
	if got := noteView(t, nb, target.ID).Auto; !slices.Equal(got, []string{bread.ID}) {
		t.Errorf("auto tags after the hand tags: %v, want [%s]", got, bread.ID)
	}
	for _, r := range recipes {
		if err := nb.UntagNote(ctx, r.ID, bread.ID); err != nil {
			t.Fatal(err)
		}
	}
	if got := noteView(t, nb, target.ID).Auto; len(got) != 0 {
		t.Errorf("auto tags after the untags: %v, want none", got)
	}
}

// TestNotebook_ViewAndWritesDoNotDeadlock: View scores outside the notebook's
// lock, because scoring reads the notebook back as the Tagger's Catalog;
// holding a read lock across it would deadlock against a waiting writer.
func TestNotebook_ViewAndWritesDoNotDeadlock(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	nb := newNotebook(t, "")
	bread, _ := nb.AddTag(ctx, "cooking / bread", false)
	_, _ = nb.AddTag(ctx, "travel / hiking", false)
	var notes []Note
	for _, text := range []string{"bread dough in a hot oven", "a hike in the hills", "croissant dough with butter"} {
		n, _ := nb.AddNote(ctx, "", text)
		notes = append(notes, n)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 20)
	for g := range 20 {
		wg.Go(func() {
			note := notes[g%len(notes)]
			for i := range 10 {
				if _, err := nb.View(ctx); err != nil {
					errs <- err
					return
				}
				write := nb.TagNote
				if i%2 == 1 {
					write = nb.UntagNote
				}
				if err := write(ctx, note.ID, bread.ID); err != nil {
					errs <- err
					return
				}
			}
		})
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("View and the writes did not all return within 10 s: deadlock")
	}
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

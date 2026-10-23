package autotag

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/go-bumbu/langai/semsearch"
	"github.com/go-bumbu/langai/semsearch/eval"
	"github.com/go-bumbu/langai/semsearch/yzma"
)

// TestGoldenTagging replays the golden set through Suggest with the real
// model. When a document is tagged, a tag's members (the documents tagged
// with it by hand) are up to every other document carrying it (how many
// depends on the scenario). Gated like the yzma golden test:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=$HOME/.cache/semsearch-yzma go test ./autotag/ -run TestGoldenTagging -v
func TestGoldenTagging(t *testing.T) {
	if os.Getenv("SEMSEARCH_MODELS") == "" {
		t.Skip("set SEMSEARCH_MODELS=1 to run the golden tagging test with the real model")
	}
	dir := os.Getenv("SEMSEARCH_YZMA_TEST_DIR")
	if dir == "" {
		t.Skip("SEMSEARCH_YZMA_TEST_DIR not set; skipping real-model test")
	}
	ctx := context.Background()
	emb, err := yzma.New(ctx, yzma.Config{
		ModelsDir:     filepath.Join(dir, "models"),
		LibDir:        filepath.Join(dir, "llama-cpp", yzma.LibTag),
		ManagedLibDir: true,
	})
	if err != nil {
		t.Fatalf("yzma.New: %v", err)
	}
	ds, err := eval.Load(filepath.Join("..", "semsearch", "eval", "testdata", "golden.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	w := newGoldenWorld(ctx, t, emb, ds)
	got := map[string]float64{
		// no examples anywhere, so every tag is judged by its name alone, as in the
		// eval's name-only f1@0.46 (0.7071 in both when last measured; only the
		// floor below is enforced)
		"names only": w.tagging(ctx, t, func(int) int { return 0 }).f1(),
		// tags at different stages: 0, 1, 2, 3 or all examples, rotated
		"mixed examples": w.mixed(ctx, t).f1(),
		// the document's own tags don't exist, but the rest of their tag family
		// does: share that gets no tag. The share that gets one is an upper bound
		// on wrong tags (some picks are related family tags)
		"fits nothing, untagged": w.untagged(ctx, t),
		// two documents stitched into one page: share that gets both tags
		"two topics, both tags": w.bothTags(ctx, t),
	}
	floors := []struct {
		name  string
		floor float64
	}{
		// measured 2026-10-22 with embeddinggemma-300m Q8_0: 0.7071, 0.7515, 0.7500, 0.5882
		{"names only", 0.65},
		{"mixed examples", 0.70},
		{"fits nothing, untagged", 0.70},
		{"two topics, both tags", 0.48},
	}
	for _, f := range floors {
		t.Logf("%-24s %.4f (floor %.2f)", f.name, got[f.name], f.floor)
		// "not at or above", not "below": a scenario with nothing to count (no
		// documents, or no pairs to stitch) is 0/0, a NaN, and every comparison
		// with a NaN is false
		if !(got[f.name] >= f.floor) {
			t.Errorf("%s = %.4f, below its floor %.2f", f.name, got[f.name], f.floor)
		}
	}
	// the feature's premise is that hand-tagged examples help. The floors alone
	// would let "mixed examples" fall below "names only" unnoticed.
	if names, mixed := got["names only"], got["mixed examples"]; !(mixed > names) {
		t.Errorf("mixed examples = %.4f does not beat names only = %.4f: examples must help", mixed, names)
	}
}

// goldenWorld is the golden set loaded into one scope of a Tagger: every tag
// path a tag, and every document an item, its essence embedded once. links
// and present mirror the scope, so a scenario changes only what differs.
type goldenWorld struct {
	space   *Scope
	ds      *eval.Dataset
	tagIDs  map[string]string          // path -> id
	names   map[string]string          // id -> path
	nums    map[string]int             // id -> the tag's position in the dataset, from 1
	byTag   map[string][]string        // tag id -> ids of the documents carrying it, dataset order
	allTags []string                   // every tag's id, dataset order
	present map[string]bool            // tag id -> the tag is in the scope
	links   map[string]map[string]bool // tag id -> ids of the documents tagged with it by hand
}

func newGoldenWorld(ctx context.Context, t *testing.T, emb semsearch.Embedder, ds *eval.Dataset) *goldenWorld {
	t.Helper()
	tg, err := New(emb, NewMemStore(), Config{Model: "golden"})
	if err != nil {
		t.Fatal(err)
	}
	w := &goldenWorld{
		space: tg.Scope("1", "1"), ds: ds,
		tagIDs: map[string]string{}, names: map[string]string{}, nums: map[string]int{},
		byTag: map[string][]string{}, present: map[string]bool{}, links: map[string]map[string]bool{},
	}
	for i, path := range ds.Tags {
		id := strconv.Itoa(i + 1)
		w.tagIDs[path], w.names[id], w.nums[id] = id, path, i+1
		w.allTags = append(w.allTags, id)
		w.setTag(ctx, t, id)
	}
	for _, doc := range ds.Documents {
		for _, tg := range doc.Tags {
			w.byTag[w.tagIDs[tg]] = append(w.byTag[w.tagIDs[tg]], doc.ID)
		}
		if err := w.space.SetItem(ctx, doc.ID, doc.Essence()); err != nil {
			t.Fatal(err)
		}
	}
	return w
}

// setTag puts the tag in the scope under its path. A tag put back after
// DeleteTag starts without members, like a new one.
func (w *goldenWorld) setTag(ctx context.Context, t *testing.T, id string) {
	t.Helper()
	if err := w.space.SetTag(ctx, id, w.names[id]); err != nil {
		t.Fatal(err)
	}
	w.present[id], w.links[id] = true, map[string]bool{}
}

// setMembers gives every tag in the scope up to capOf(its position) members:
// the first documents carrying it, in dataset order, never one in skip. It
// tags and untags by hand only the documents that change.
func (w *goldenWorld) setMembers(ctx context.Context, t *testing.T, skip map[string]bool, capOf func(int) int) {
	t.Helper()
	for _, id := range w.allTags {
		if !w.present[id] {
			continue
		}
		want := map[string]bool{}
		for _, d := range w.byTag[id] {
			if len(want) == capOf(w.nums[id]) {
				break
			}
			if !skip[d] {
				want[d] = true
			}
		}
		for d := range w.links[id] {
			if want[d] {
				continue
			}
			if err := w.space.UntagByHand(ctx, d, id); err != nil {
				t.Fatal(err)
			}
			delete(w.links[id], d)
		}
		for d := range want {
			if w.links[id][d] {
				continue
			}
			if err := w.space.TagByHand(ctx, d, id); err != nil {
				t.Fatal(err)
			}
			w.links[id][d] = true
		}
	}
}

func (w *goldenWorld) suggest(ctx context.Context, t *testing.T, itemID string) []string {
	t.Helper()
	got, err := w.space.Suggest(ctx, itemID)
	if err != nil {
		t.Fatalf("Suggest: %v", err)
	}
	return got
}

func (w *goldenWorld) truth(doc eval.Document) []string {
	ids := make([]string, len(doc.Tags))
	for i, tg := range doc.Tags {
		ids[i] = w.tagIDs[tg]
	}
	return ids
}

type tagCounts struct{ tp, fp, fn int }

func (c *tagCounts) add(got, want []string) {
	for _, g := range got {
		if slices.Contains(want, g) {
			c.tp++
		} else {
			c.fp++
		}
	}
	for _, x := range want {
		if !slices.Contains(got, x) {
			c.fn++
		}
	}
}

func (c tagCounts) f1() float64 { return 2 * float64(c.tp) / float64(2*c.tp+c.fp+c.fn) }

func (w *goldenWorld) tagging(ctx context.Context, t *testing.T, capOf func(int) int) tagCounts {
	var c tagCounts
	for _, doc := range w.ds.Documents {
		w.setMembers(ctx, t, map[string]bool{doc.ID: true}, capOf)
		c.add(w.suggest(ctx, t, doc.ID), w.truth(doc))
	}
	return c
}

func (w *goldenWorld) mixed(ctx context.Context, t *testing.T) tagCounts {
	var all tagCounts
	stages := []int{0, 1, 2, 3, math.MaxInt}
	for r := range stages {
		c := w.tagging(ctx, t, func(n int) int { return stages[(n+r)%len(stages)] })
		all.tp, all.fp, all.fn = all.tp+c.tp, all.fp+c.fp, all.fn+c.fn
	}
	return all
}

// untagged deletes each document's own tags, asks for its suggestions and
// puts the tags back, and reports the share of documents that get none.
func (w *goldenWorld) untagged(ctx context.Context, t *testing.T) float64 {
	n := 0
	for _, doc := range w.ds.Documents {
		own := w.truth(doc)
		for _, id := range own {
			if err := w.space.DeleteTag(ctx, id); err != nil {
				t.Fatal(err)
			}
			delete(w.links, id)
			delete(w.present, id)
		}
		w.setMembers(ctx, t, map[string]bool{doc.ID: true}, func(int) int { return math.MaxInt })
		if len(w.suggest(ctx, t, doc.ID)) == 0 {
			n++
		}
		for _, id := range own {
			w.setTag(ctx, t, id)
		}
	}
	return float64(n) / float64(len(w.ds.Documents))
}

// bothTags stitches pairs of single-tag documents with different leaf tags
// into one page (title, excerpt and the first 1000 bytes of each), about 100
// pairs spread over the dataset, and reports how often both tags are
// suggested. Each page is the item "page", deleted at the end.
func (w *goldenWorld) bothTags(ctx context.Context, t *testing.T) float64 {
	var single []eval.Document
	for _, doc := range w.ds.Documents {
		if len(doc.Tags) == 1 && strings.Contains(doc.Tags[0], " / ") {
			single = append(single, doc)
		}
	}
	var pairs [][2]eval.Document
	for i := range single {
		for j := i + 1; j < len(single); j++ {
			if single[i].Tags[0] != single[j].Tags[0] {
				pairs = append(pairs, [2]eval.Document{single[i], single[j]})
			}
		}
	}
	both, n := 0, 0
	for k := 0; k < len(pairs); k += max(1, len(pairs)/100) {
		a, b := pairs[k][0], pairs[k][1]
		page := eval.Document{
			Title: a.Title + " / " + b.Title, Excerpt: a.Excerpt + " " + b.Excerpt, Site: a.Site,
			Text: cutBytes(a.Text, 1000) + "\n\n" + cutBytes(b.Text, 1000),
		}
		if err := w.space.SetItem(ctx, "page", page.Essence()); err != nil {
			t.Fatal(err)
		}
		w.setMembers(ctx, t, map[string]bool{a.ID: true, b.ID: true}, func(int) int { return math.MaxInt })
		got := w.suggest(ctx, t, "page")
		n++
		if slices.Contains(got, w.tagIDs[a.Tags[0]]) && slices.Contains(got, w.tagIDs[b.Tags[0]]) {
			both++
		}
	}
	if err := w.space.DeleteItem(ctx, "page"); err != nil {
		t.Fatal(err)
	}
	return float64(both) / float64(n)
}

func cutBytes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

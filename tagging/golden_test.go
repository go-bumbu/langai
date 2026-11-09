package tagging

import (
	"context"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/provider/yzma"
	"github.com/go-bumbu/langai/zarf/eval"
)

// TestGoldenTagging replays the golden set through Suggest with the real
// model. When a document is tagged, a tag's members (the documents tagged
// with it by hand) are up to every other document carrying it (how many
// depends on the scenario). Gated like the yzma golden test:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=$HOME/.cache/semsearch-yzma go test ./tagging/ -run TestGoldenTagging -v
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
	ds, err := eval.Load(filepath.Join("..", "zarf", "eval", "testdata", "golden.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	w := newGoldenWorld(t, emb, ds, 0)
	untagged, unrelated, _ := w.untagged(ctx, t, w.truth)
	noWrongTag, _, closestWrong := w.untagged(ctx, t, w.family)
	got := map[string]float64{
		// no examples anywhere, so every tag is judged by its name alone (the
		// eval's name-only f1@0.46, without the path rule, was 0.7071 when last
		// measured; only the floor below is enforced)
		"names only": w.tagging(ctx, t, func(int) int { return 0 }).f1(),
		// tags at different stages: 0, 1, 2, 3 or all examples, rotated
		"mixed examples": w.mixed(ctx, t).f1(),
		// the document's own tags don't exist, but the rest of their tag family
		// does: share that gets no tag but ancestors of its own, which its path
		// implies ("cooking" for a recipe when "cooking / recipes" is gone). The
		// other picks are siblings and children: related tags, not unrelated ones
		"own tags gone, none but ancestors": unrelated,
		// the whole family of each of the document's tags is gone (the tag, its
		// parent, siblings and children), so no tag left fits it: share that
		// gets no tag. One minus this is the wrong-tag rate
		"fits no tag, gets none": noWrongTag,
		// two documents stitched into one page: share that gets both tags
		"two topics, both tags": w.bothTags(ctx, t),
		// the same with ChunkBytes 2000: every essence is one piece, so the
		// other scenarios do not change, but a stitched page (2.3 to 2.9 KB)
		// is two, scored by the best of its halves and their centroid (see
		// itemScore)
		"two topics, ChunkBytes 2000": newGoldenWorld(t, emb, ds, 2000).bothTags(ctx, t),
	}
	// with only its own tags gone, the share that gets no tag at all, ancestors
	// included: logged, not enforced (0.7500 under the centroid rule without the
	// path rule, 0.6429 under this one, which gives a recipe "cooking" once
	// "cooking / recipes" is gone)
	t.Logf("%-33s %.4f", "own tags gone, no tag at all", untagged)
	// how far the best unrelated tag of any document is from its cutoff
	t.Logf("%-33s %s", "fits no tag, closest wrong tag", closestWrong)
	floors := []struct {
		name  string
		floor float64
	}{
		// measured 2026-11-06 with embeddinggemma-300m Q8_0, the mean rule and
		// the path rule: 0.7889, 0.8597, 0.8095, 0.8039. The centroid rule
		// before them, at 0.46 and 0.70: 0.7071, 0.7561, 0.8333, 0.5882. Two
		// topics with ChunkBytes 2000, measured 2026-11-07: 0.8725 (0.8431 when
		// a text of two pieces scored by their mean, 0.7745 by the better one).
		// Fits no tag, measured 2026-11-09: 1.0000, none of the 84 documents got
		// a tag, the closest being "cooking / baking" for de-zuckerarm, 0.0157
		// under its cutoff; its floor is that less 0.05, so four documents may
		// get a wrong tag before it fails
		{"names only", 0.74},
		{"mixed examples", 0.78},
		{"own tags gone, none but ancestors", 0.75},
		{"fits no tag, gets none", 0.95},
		{"two topics, both tags", 0.70},
		{"two topics, ChunkBytes 2000", 0.82},
	}
	for _, f := range floors {
		t.Logf("%-33s %.4f (floor %.2f)", f.name, got[f.name], f.floor)
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

// goldenWorld is the golden set as an application holds it, and the Catalog
// a Tagger reads it through: every tag path a tag, and every document an
// item, its essence its text. links and present are the application's
// records, so a scenario changes only what differs, and every change goes up
// a revision: rev, the space's, on every change, and a tag's own when its
// members change.
type goldenWorld struct {
	space   *Scope
	ds      *eval.Dataset
	tagIDs  map[string]string          // path -> id
	names   map[string]string          // id -> path
	nums    map[string]int             // id -> the tag's position in the dataset, from 1
	byTag   map[string][]string        // tag id -> ids of the documents carrying it, dataset order
	essence map[string]string          // document id -> its essence
	allTags []string                   // every tag's id, dataset order
	present map[string]bool            // tag id -> the tag is in the space
	links   map[string]map[string]bool // tag id -> ids of the documents tagged with it by hand
	rev     int                        // the space's revision
	tagRev  map[string]int             // tag id -> its revision
}

func newGoldenWorld(t *testing.T, emb langai.Embedder, ds *eval.Dataset, chunkBytes int) *goldenWorld {
	t.Helper()
	w := &goldenWorld{
		ds:     ds,
		tagIDs: map[string]string{}, names: map[string]string{}, nums: map[string]int{},
		byTag: map[string][]string{}, essence: map[string]string{}, present: map[string]bool{},
		links: map[string]map[string]bool{}, tagRev: map[string]int{},
	}
	tg, err := NewTagger(emb, w, NewMemCache(), TaggerConfig{Model: "golden", ChunkBytes: chunkBytes})
	if err != nil {
		t.Fatal(err)
	}
	w.space = tg.Scope("1", "1")
	for i, path := range ds.Tags {
		id := strconv.Itoa(i + 1)
		w.tagIDs[path], w.names[id], w.nums[id] = id, path, i+1
		w.allTags = append(w.allTags, id)
		w.setTag(id)
	}
	for _, doc := range ds.Documents {
		for _, tg := range doc.Tags {
			w.byTag[w.tagIDs[tg]] = append(w.byTag[w.tagIDs[tg]], doc.ID)
		}
		w.essence[doc.ID] = doc.Essence()
	}
	return w
}

// Tags returns the tags in the space, in dataset order.
func (w *goldenWorld) Tags(context.Context, string, string) ([]CatalogTag, error) {
	var out []CatalogTag
	for _, id := range w.allTags {
		if w.present[id] {
			out = append(out, CatalogTag{ID: id, Name: w.names[id], Revision: strconv.Itoa(w.tagRev[id])})
		}
	}
	return out, nil
}

// Members returns the essences of the documents tagged with each tag by hand,
// by document id.
func (w *goldenWorld) Members(_ context.Context, _, _ string, tagIDs []string) (map[string][]string, error) {
	out := map[string][]string{}
	for _, id := range tagIDs {
		for _, d := range slices.Sorted(maps.Keys(w.links[id])) {
			out[id] = append(out[id], w.essence[d])
		}
	}
	return out, nil
}

// setTag puts the tag in the space under its path. A tag put back after
// deleteTag starts without members, like a new one, so its revision goes up
// as well as the space's.
func (w *goldenWorld) setTag(id string) {
	w.present[id], w.links[id] = true, map[string]bool{}
	w.tagRev[id]++
	w.rev++
}

// deleteTag deletes the tag and its hand-made links.
func (w *goldenWorld) deleteTag(id string) {
	delete(w.links, id)
	delete(w.present, id)
	w.rev++
}

// link tags the document with the tag by hand, or untags it, which changes
// the tag's members.
func (w *goldenWorld) link(tagID, docID string, on bool) {
	if on {
		w.links[tagID][docID] = true
	} else {
		delete(w.links[tagID], docID)
	}
	w.tagRev[tagID]++
	w.rev++
}

// setMembers gives every tag in the space up to capOf(its position) members:
// the first documents carrying it, in dataset order, never one in skip. It
// tags and untags by hand only the documents that change.
func (w *goldenWorld) setMembers(skip map[string]bool, capOf func(int) int) {
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
			if !want[d] {
				w.link(id, d, false)
			}
		}
		for d := range want {
			if !w.links[id][d] {
				w.link(id, d, true)
			}
		}
	}
}

// suggest returns the tags suggested for the document's essence at the
// space's current revision.
func (w *goldenWorld) suggest(ctx context.Context, t *testing.T, doc eval.Document) []string {
	t.Helper()
	got, err := w.space.Suggest(ctx, strconv.Itoa(w.rev), doc.Essence())
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
		w.setMembers(map[string]bool{doc.ID: true}, capOf)
		c.add(w.suggest(ctx, t, doc), w.truth(doc))
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

// untagged deletes the tags drop names for each document (its own tags, or
// their whole families), asks for its suggestions and puts the tags back. It
// reports the share of documents that get none, the share that get none but
// ancestors of their own tags, and the tag left in the space that came
// closest to being picked, or was picked by most: its score less its cutoff,
// with the document and the tag, which says how much room the first share has.
func (w *goldenWorld) untagged(ctx context.Context, t *testing.T, drop func(eval.Document) []string) (none, noUnrelated float64, closest string) {
	t.Helper()
	n, ok, margin := 0, 0, math.Inf(-1)
	for _, doc := range w.ds.Documents {
		own := drop(doc)
		for _, id := range own {
			w.deleteTag(id)
		}
		w.setMembers(map[string]bool{doc.ID: true}, func(int) int { return math.MaxInt })
		scores, err := w.space.Scores(ctx, strconv.Itoa(w.rev), doc.Essence())
		if err != nil {
			t.Fatalf("Scores: %v", err)
		}
		var got []string // what Suggest returns: the picked ids
		for _, sc := range scores {
			if sc.Picked() {
				got = append(got, sc.ID)
			}
			if m := sc.Score - sc.Cutoff; m > margin {
				margin = m
				closest = fmt.Sprintf("%+.4f (%s, %q at %.4f)", m, doc.ID, w.names[sc.ID], sc.Score)
			}
		}
		if len(got) == 0 {
			n++
		}
		if !slices.ContainsFunc(got, func(id string) bool { return !w.ancestorOfOwn(doc, id) }) {
			ok++
		}
		for _, id := range own {
			w.setTag(id)
		}
	}
	all := float64(len(w.ds.Documents))
	return float64(n) / all, float64(ok) / all, closest
}

// family returns the tags of the families of the document's own tags: every
// tag under the top level of one of them, so the tag itself, its parent, its
// siblings and its children. What is left in the space has nothing to do with
// the document.
func (w *goldenWorld) family(doc eval.Document) []string {
	var ids []string
	for _, id := range w.allTags {
		if slices.ContainsFunc(doc.Tags, func(own string) bool { return topLevel(own) == topLevel(w.names[id]) }) {
			ids = append(ids, id)
		}
	}
	return ids
}

// topLevel is the first level of a tag path.
func topLevel(path string) string {
	top, _, _ := strings.Cut(path, " / ")
	return top
}

// ancestorOfOwn reports whether the tag is an ancestor of one of the
// document's own tags.
func (w *goldenWorld) ancestorOfOwn(doc eval.Document, tagID string) bool {
	return slices.ContainsFunc(doc.Tags, func(own string) bool { return strings.HasPrefix(own, w.names[tagID]+" / ") })
}

// bothTags stitches pairs of single-tag documents with different leaf tags
// into one page (title, excerpt and the first 1000 bytes of each), about 100
// pairs spread over the dataset, and reports how often both tags are
// suggested. A page is a text the application holds nowhere: it is only
// suggested for.
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
		w.setMembers(map[string]bool{a.ID: true, b.ID: true}, func(int) int { return math.MaxInt })
		got := w.suggest(ctx, t, page)
		n++
		if slices.Contains(got, w.tagIDs[a.Tags[0]]) && slices.Contains(got, w.tagIDs[b.Tags[0]]) {
			both++
		}
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

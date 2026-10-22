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

// TestGoldenTagging replays the golden set through Assign with the real
// model. When a document is tagged, a tag's members (its hand-tagged
// examples) are up to every other document carrying it (how many depends on
// the scenario). Gated like the yzma golden test:
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

// goldenWorld is the golden set loaded into a fakeTagStore: one scope, every
// tag path a tag, every document's essence embedded once and kept as a
// caller would cache it.
type goldenWorld struct {
	tg      *Tagger
	src     *fakeTagStore
	ds      *eval.Dataset
	tagIDs  map[string]string    // path -> id
	nums    map[string]int       // id -> the tag's position in the dataset, from 1
	byTag   map[string][]string  // tag id -> ids of the documents carrying it, dataset order
	vecs    map[string][]float32 // document id -> essence embedding
	allTags []Tag
	tags    []Tag // the scope's current tags
}

func newGoldenWorld(ctx context.Context, t *testing.T, emb semsearch.Embedder, ds *eval.Dataset) *goldenWorld {
	t.Helper()
	w := &goldenWorld{
		ds: ds, src: newFakeTagStore(),
		tagIDs: map[string]string{}, nums: map[string]int{}, byTag: map[string][]string{}, vecs: map[string][]float32{},
	}
	for i, path := range ds.Tags {
		id := strconv.Itoa(i + 1)
		w.tagIDs[path], w.nums[id] = id, i+1
		w.allTags = append(w.allTags, Tag{ID: id, Name: path, NameHash: path})
	}
	essences := make([]string, len(ds.Documents))
	for i, doc := range ds.Documents {
		essences[i] = doc.Essence()
		for _, tg := range doc.Tags {
			w.byTag[w.tagIDs[tg]] = append(w.byTag[w.tagIDs[tg]], doc.ID)
		}
	}
	vecs, err := emb.Embed(ctx, semsearch.RoleQuery, essences)
	if err != nil {
		t.Fatal(err)
	}
	for i, doc := range ds.Documents {
		w.vecs[doc.ID] = vecs[i]
	}
	w.tags = w.allTags
	w.tg = New(emb, semsearch.NewMemStore(), Config{Model: "golden"})
	return w
}

// setMembers gives every tag up to capOf(its position) members: the first
// documents carrying it, in dataset order, never one in skip.
func (w *goldenWorld) setMembers(skip map[string]bool, capOf func(int) int) {
	for id, docs := range w.byTag {
		var vecs [][]float32
		var hashes []string
		for _, d := range docs {
			if len(vecs) == capOf(w.nums[id]) {
				break
			}
			if skip[d] {
				continue
			}
			vecs = append(vecs, w.vecs[d])
			hashes = append(hashes, d)
		}
		w.src.members[id], w.src.memberHashes[id] = vecs, hashes
	}
}

func (w *goldenWorld) assign(ctx context.Context, t *testing.T, vec []float32) []string {
	t.Helper()
	got, err := w.tg.Assign(ctx, Scope{Index: "1", Partition: "1"}, w.src, w.tags, vec)
	if err != nil {
		t.Fatalf("Assign: %v", err)
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
		c.add(w.assign(ctx, t, w.vecs[doc.ID]), w.truth(doc))
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

func (w *goldenWorld) untagged(ctx context.Context, t *testing.T) float64 {
	n := 0
	for _, doc := range w.ds.Documents {
		own := w.truth(doc)
		w.tags = nil
		for _, tg := range w.allTags {
			if !slices.Contains(own, tg.ID) {
				w.tags = append(w.tags, tg)
			}
		}
		w.setMembers(map[string]bool{doc.ID: true}, func(int) int { return math.MaxInt })
		if len(w.assign(ctx, t, w.vecs[doc.ID])) == 0 {
			n++
		}
	}
	w.tags = w.allTags
	return float64(n) / float64(len(w.ds.Documents))
}

// bothTags stitches pairs of single-tag documents with different leaf tags
// into one page (title, excerpt and the first 1000 bytes of each), about 100
// pairs spread over the dataset, and reports how often both tags are assigned.
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
		vec, err := w.tg.Embed(ctx, page.Essence())
		if err != nil {
			t.Fatal(err)
		}
		w.setMembers(map[string]bool{a.ID: true, b.ID: true}, func(int) int { return math.MaxInt })
		got := w.assign(ctx, t, vec)
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

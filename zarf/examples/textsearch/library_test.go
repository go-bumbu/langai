package textsearch

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/search"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/vector"
)

const breadDoc = `# Baking bread

Notes on baking bread at home.

## Sourdough starter

A sourdough starter is flour and water left to ferment. Feed the starter every day until it rises.

## Oven

Bake the loaf in a hot oven with steam for a crisp crust.
`

func write(t *testing.T, path, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
		t.Fatal(err)
	}
}

// otherModel is the FakeEmbedder under another name: the same vectors, but
// to a store another model.
type otherModel struct{ sensetest.FakeEmbedder }

func (otherModel) Model() string { return "another-model" }

// fixture is a documents directory with a Markdown file with sections, a
// text file, one in a subdirectory, and files the scan must leave alone.
type fixture struct {
	dir   string
	lib   *Library
	store *vector.MemStore
	calls *atomic.Int64
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	dir := t.TempDir()
	write(t, filepath.Join(dir, "bread.md"), breadDoc)
	write(t, filepath.Join(dir, "modules.txt"), "Go modules pin every dependency of a build to an exact version in go.mod.")
	write(t, filepath.Join(dir, "sub dir", "hike.md"), "# Hut to hut\n\nA hike of several days in the mountains, sleeping in huts.\n")
	write(t, filepath.Join(dir, "picture.png"), "not a document")
	write(t, filepath.Join(dir, ".hidden", "secret.md"), "never read")
	f := &fixture{dir: dir, store: vector.NewMemStore(), calls: &atomic.Int64{}}
	f.lib = f.library(t, sensetest.FakeEmbedder{Calls: f.calls})
	return f
}

// library builds a Library over the fixture's directory and store with the
// given embedder.
func (f *fixture) library(t *testing.T, emb sense.Embedder) *Library {
	t.Helper()
	ix, err := search.New(emb, f.store, "examples", "local", "docs")
	if err != nil {
		t.Fatal(err)
	}
	coord := vector.Coord{Index: "examples", Partition: "local", Collection: "docs"}
	lib, err := New(Config{
		Dir: f.dir, Index: ix,
		Info:       Info{Model: "fake", Dimensions: 64, Store: "memory"},
		ModelState: func() string { return "loaded" },
		Stored:     func(ctx context.Context) (string, error) { return f.store.Model(ctx, coord) },
		Examples:   []string{"bread"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return lib
}

func scanned(t *testing.T) *fixture {
	t.Helper()
	f := newFixture(t)
	if _, err := f.lib.Scan(context.Background()); err != nil {
		t.Fatal(err)
	}
	return f
}

func doc(t *testing.T, lib *Library, id string) Doc {
	t.Helper()
	d, ok := lib.Doc(id)
	if !ok {
		t.Fatalf("no document %q in the catalog", id)
	}
	return d
}

func TestNew_Validates(t *testing.T) {
	ix, err := search.New(sensetest.FakeEmbedder{}, vector.NewMemStore(), "examples", "local", "docs")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := New(Config{Index: ix}); err == nil {
		t.Error("an empty directory name was accepted")
	}
	if _, err := New(Config{Dir: t.TempDir()}); err == nil {
		t.Error("a nil index was accepted")
	}
	lib, err := New(Config{Dir: t.TempDir(), Index: ix})
	if err != nil {
		t.Fatal(err)
	}
	if lib.Limit() != 10 || lib.Info().Chunker != (&search.MarkdownChunker{}).Fingerprint() || lib.ModelState() != "" || lib.StoredModel(context.Background()) != "" {
		t.Errorf("defaults: limit %d, chunker %q", lib.Limit(), lib.Info().Chunker)
	}
	// a chunker without a fingerprint is named by its type, as the index does
	plain, err := New(Config{Dir: t.TempDir(), Index: ix, Chunker: lineChunker{}})
	if err != nil {
		t.Fatal(err)
	}
	if got := plain.Info().Chunker; got != "textsearch.lineChunker" {
		t.Errorf("chunker without a fingerprint: %q", got)
	}
}

// lineChunker cuts a text into its lines.
type lineChunker struct{}

func (lineChunker) Chunk(text string) []string { return strings.Split(strings.TrimSpace(text), "\n") }

func TestScan_IndexesEveryDocumentInOneBatch(t *testing.T) {
	f := newFixture(t)
	if got := f.lib.LastScan(); !got.At.IsZero() {
		t.Errorf("stats before the first scan: %+v", got)
	}
	stats, err := f.lib.Scan(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.Indexed != 3 || stats.Unchanged != 0 || stats.Failed != 0 || stats.Removed != 0 || stats.Pending != 0 || stats.Error != "" {
		t.Errorf("first scan: %+v", stats)
	}
	docs := f.lib.Docs()
	if len(docs) != 3 || docs[0].ID != "bread.md" || docs[2].ID != "sub dir/hike.md" {
		t.Fatalf("catalog: %+v", docs)
	}
	bread := doc(t, f.lib, "bread.md")
	if bread.Title != "Baking bread" || bread.State != StateIndexed || bread.Chunks != 3 {
		t.Errorf("bread.md: %+v", bread)
	}
	if d := doc(t, f.lib, "modules.txt"); !strings.HasPrefix(d.Title, "Go modules pin") || d.Chunks != 1 {
		t.Errorf("modules.txt: %+v", d)
	}
}

func TestScan_SkipsUnchangedDocuments(t *testing.T) {
	f := scanned(t)
	// nothing changed: the batch embeds nothing
	before := f.calls.Load()
	stats, err := f.lib.Scan(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.Indexed != 0 || stats.Unchanged != 3 || f.calls.Load() != before {
		t.Errorf("second scan: %+v, %d embed calls", stats, f.calls.Load()-before)
	}
	if doc(t, f.lib, "bread.md").State != StateUnchanged || f.lib.LastScan().Unchanged != 3 {
		t.Errorf("state after an unchanged scan: %+v", doc(t, f.lib, "bread.md"))
	}
}

func TestScan_ReconcilesWithTheDirectory(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	// one file edited, one removed, one added
	write(t, filepath.Join(f.dir, "modules.txt"), "Go modules: go.sum records a hash of every dependency.")
	if err := os.Remove(filepath.Join(f.dir, "sub dir", "hike.md")); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(f.dir, "new.markdown"), "# Tax return\n\nWhat a freelancer may deduct.\n")
	stats, err := f.lib.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Indexed != 2 || stats.Unchanged != 1 || stats.Removed != 1 {
		t.Errorf("scan after changes: %+v", stats)
	}
	if _, ok := f.lib.Doc("sub dir/hike.md"); ok {
		t.Error("the removed file is still in the catalog")
	}
	stored, err := f.store.Docs(ctx, vector.Coord{Index: "examples", Partition: "local", Collection: "docs"}, "", 0)
	if err != nil || len(stored) != 3 {
		t.Errorf("the store holds %d documents (%v), want 3", len(stored), err)
	}
	for _, d := range stored {
		if d.ID == "sub dir/hike.md" {
			t.Error("the removed file is still in the index")
		}
	}
}

func TestScan_ADocumentThatCannotBeIndexedFailsAlone(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	// no words: the fake embeds it to the zero vector, which no store takes
	write(t, filepath.Join(f.dir, "dots.txt"), "… !!! ???")
	// longer than the scan reads
	write(t, filepath.Join(f.dir, "huge.txt"), strings.Repeat("word ", maxDocBytes/5+1))
	stats, err := f.lib.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Indexed != 3 || stats.Failed != 2 || len(stats.Failures) != 2 {
		t.Errorf("scan: %+v", stats)
	}
	if d := doc(t, f.lib, "dots.txt"); d.State != StateFailed || d.Error == "" || stats.Failures["dots.txt"] != d.Error {
		t.Errorf("dots.txt: %+v", d)
	}
	if d := doc(t, f.lib, "huge.txt"); d.State != StateFailed || !strings.Contains(d.Error, "MiB") {
		t.Errorf("huge.txt: %+v", d)
	}
	if _, err := f.lib.Similar(ctx, "dots.txt", 0); !errors.Is(err, search.ErrNotIndexed) {
		t.Errorf("similar to a document that is not indexed: %v", err)
	}
	if v, err := f.lib.View("huge.txt"); err != nil || len(v.Chunks) != 0 {
		t.Errorf("view of a file too long to read: %+v, %v", v, err)
	}
}

func TestScan_AnEmbedderThatIsDownStopsTheBatch(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	down := f.library(t, sensetest.FakeEmbedder{Err: errors.New("the model is gone")})
	stats, err := down.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Indexed != 0 || stats.Pending != 3 || !strings.Contains(stats.Error, "the model is gone") || stats.Mismatch {
		t.Errorf("scan with the embedder down: %+v", stats)
	}
	if d := doc(t, down, "bread.md"); d.State != StatePending {
		t.Errorf("a document the batch did not reach: %+v", d)
	}
}

func TestScan_AnotherModelIsAMismatchUntilReset(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	other := f.library(t, otherModel{})
	// the files changed, so the batch has something to write
	write(t, filepath.Join(f.dir, "bread.md"), breadDoc+"\nMore about flour.\n")
	stats, err := other.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !stats.Mismatch || stats.Indexed != 0 || stats.Error == "" {
		t.Fatalf("scan with another model: %+v", stats)
	}
	if got := other.StoredModel(ctx); got != sensetest.FakeEmbedderModel {
		t.Errorf("stored model %q, want the one that built the collection", got)
	}
	if _, err := other.Search(ctx, "bread", 0); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("search with another model: %v", err)
	}

	if err := other.Reset(ctx); err != nil {
		t.Fatal(err)
	}
	if d := doc(t, other, "bread.md"); d.State != StatePending || !other.LastScan().At.IsZero() {
		t.Errorf("after the reset: %+v", d)
	}
	stats, err = other.Scan(ctx)
	if err != nil || stats.Indexed != 3 || stats.Mismatch {
		t.Errorf("scan after the reset: %+v, %v", stats, err)
	}
	if got := other.StoredModel(ctx); got != "another-model" {
		t.Errorf("stored model after the rebuild: %q", got)
	}
}

func TestScan_OneAtATimeAndAMissingDirectory(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	f.lib.scanning.Store(true)
	if _, err := f.lib.Scan(ctx); !errors.Is(err, ErrScanning) {
		t.Errorf("a second scan: %v", err)
	}
	if err := f.lib.Reset(ctx); !errors.Is(err, ErrScanning) {
		t.Errorf("a reset during a scan: %v", err)
	}
	if !f.lib.Scanning() {
		t.Error("Scanning is false during a scan")
	}
	f.lib.scanning.Store(false)

	// a directory that is not there yet holds no documents
	ix, err := search.New(sensetest.FakeEmbedder{}, vector.NewMemStore(), "examples", "local", "docs")
	if err != nil {
		t.Fatal(err)
	}
	empty, err := New(Config{Dir: filepath.Join(t.TempDir(), "missing"), Index: ix})
	if err != nil {
		t.Fatal(err)
	}
	stats, err := empty.Scan(ctx)
	if err != nil || stats.Indexed != 0 || len(empty.Docs()) != 0 {
		t.Errorf("scan of a missing directory: %+v, %v", stats, err)
	}

	// a scan whose context has ended reports it
	done, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := f.lib.Scan(done); !errors.Is(err, context.Canceled) {
		t.Errorf("a cancelled scan: %v", err)
	}
}

func TestSearch_ReturnsTheMatchingChunkWithItsHeading(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	res, err := f.lib.Search(ctx, "feed the sourdough starter", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Hits) == 0 || len(res.Hits) > 2 || res.Query == "" || res.Stored {
		t.Fatalf("search: %+v", res)
	}
	top := res.Hits[0]
	if top.Doc.ID != "bread.md" || top.Doc.Title != "Baking bread" || top.Seq != 1 || top.Heading != "Baking bread › Sourdough starter" ||
		!strings.HasPrefix(top.Text, "A sourdough starter") || top.Score <= 0 {
		t.Errorf("top hit: %+v", top)
	}
	// a text without headings has no heading path: the chunk is the text
	res, err = f.lib.Search(ctx, "go modules dependency version", 1)
	if err != nil || len(res.Hits) != 1 || res.Hits[0].Doc.ID != "modules.txt" || res.Hits[0].Heading != "" || !strings.HasPrefix(res.Hits[0].Text, "Go modules") {
		t.Errorf("search of a plain text: %+v, %v", res, err)
	}
	if res, err := f.lib.Search(ctx, "   ", 0); err != nil || len(res.Hits) != 0 {
		t.Errorf("a blank query: %+v, %v", res, err)
	}
}

func TestSimilar_SearchesWithStoredVectors(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	write(t, filepath.Join(f.dir, "loaf.md"), "# A loaf\n\nBake the sourdough loaf in a hot oven with steam.\n")
	if _, err := f.lib.Scan(ctx); err != nil {
		t.Fatal(err)
	}
	before := f.calls.Load()
	res, err := f.lib.Similar(ctx, "bread.md", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Stored || res.Doc == nil || res.Doc.ID != "bread.md" || len(res.Hits) != 3 || res.Hits[0].Doc.ID != "loaf.md" {
		t.Errorf("similar: %+v", res)
	}
	for _, h := range res.Hits {
		if h.Doc.ID == "bread.md" {
			t.Error("the document is among its own look-alikes")
		}
	}
	if f.calls.Load() != before {
		t.Error("more like this ran the embedder")
	}
	if _, err := f.lib.Similar(ctx, "nope.md", 0); !errors.Is(err, ErrNotFound) {
		t.Errorf("similar to an unknown document: %v", err)
	}
}

func TestSearch_AHitWhoseFileIsUnknownKeepsItsID(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	// another catalog over the same index, which has not scanned
	fresh := f.library(t, sensetest.FakeEmbedder{})
	res, err := fresh.Search(ctx, "sourdough starter", 1)
	if err != nil || len(res.Hits) != 1 || res.Hits[0].Doc.ID != "bread.md" || res.Hits[0].Doc.Name != "bread.md" || res.Hits[0].Heading != "" {
		t.Errorf("hit without a catalog entry: %+v, %v", res, err)
	}
}

func TestView_ShowsTheChunksThatWereEmbedded(t *testing.T) {
	f := scanned(t)
	v, err := f.lib.View("bread.md")
	if err != nil {
		t.Fatal(err)
	}
	if len(v.Chunks) != 3 || v.Changed || v.Text != breadDoc {
		t.Fatalf("view: %d chunks, changed %v", len(v.Chunks), v.Changed)
	}
	if c := v.Chunks[0]; c.Seq != 0 || c.Heading != "Baking bread" || c.Text != "Notes on baking bread at home." || c.Bytes != len("Baking bread\n\nNotes on baking bread at home.") {
		t.Errorf("first chunk: %+v", c)
	}
	if c := v.Chunks[2]; c.Heading != "Baking bread › Oven" || !strings.HasPrefix(c.Text, "Bake the loaf") {
		t.Errorf("last chunk: %+v", c)
	}

	// the file changed since the scan: the view says so
	write(t, filepath.Join(f.dir, "bread.md"), breadDoc+"\n## Flour\n\nUse strong flour.\n")
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(filepath.Join(f.dir, "bread.md"), later, later); err != nil {
		t.Fatal(err)
	}
	if v, err = f.lib.View("bread.md"); err != nil || !v.Changed || len(v.Chunks) != 4 {
		t.Errorf("view of a changed file: %d chunks, changed %v, %v", len(v.Chunks), v.Changed, err)
	}
	if _, err := f.lib.View("nope.md"); !errors.Is(err, ErrNotFound) {
		t.Errorf("view of an unknown document: %v", err)
	}
	if err := os.Remove(filepath.Join(f.dir, "modules.txt")); err != nil {
		t.Fatal(err)
	}
	if _, err := f.lib.View("modules.txt"); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("view of a file that is gone: %v", err)
	}
}

func TestHelpers(t *testing.T) {
	for text, want := range map[string]string{
		"# Title\n\ntext":            "Title",
		"\n\n  first line\nsecond":   "first line",
		"":                           "fallback",
		strings.Repeat("é", 150):     strings.Repeat("é", 100) + "…",
		"### Deep heading ###\nrest": "Deep heading ###",
		"   \n\t\n#\n## Real\n":      "Real",
		"plain text without heading": "plain text without heading",
	} {
		if got := title(text, "fallback"); got != want {
			t.Errorf("title(%q) = %q, want %q", text, got, want)
		}
	}
	heads := headings("# One\n\ntext\n## Two words\n####### seven\n#nospace\n   ### Three  \n")
	if len(heads) != 3 || !heads["One"] || !heads["Two words"] || !heads["Three"] {
		t.Errorf("headings: %v", heads)
	}
	if got := clean("a\x00b\xffc"); got != "ab�c" {
		t.Errorf("clean: %q", got)
	}
	for _, c := range []struct{ chunk, heading, text string }{
		{"One › Two words\n\nbody", "One › Two words", "body"},
		{"One", "One", ""},
		{"Not a heading\n\nbody", "", "Not a heading\n\nbody"},
		{"One\nTwo words\n\nbody", "", "One\nTwo words\n\nbody"},
	} {
		if h, text := splitHeading(c.chunk, heads); h != c.heading || text != c.text {
			t.Errorf("splitHeading(%q) = %q, %q", c.chunk, h, text)
		}
	}
	if h, text := splitHeading("One\n\nbody", nil); h != "" || text != "One\n\nbody" {
		t.Errorf("splitHeading without headings: %q, %q", h, text)
	}
}

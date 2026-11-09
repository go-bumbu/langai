package mediasearch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/media"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
)

// otherImageModel is the FakeImageEmbedder under another name: the same
// vectors, but to a store another model.
type otherImageModel struct{ countingEmbedder }

func (*otherImageModel) Model() string { return "another-image-model" }

// libraryOn builds a Library over dir and store with the given embedder and
// sampler, as newFixture does, told how to ask the store for its model.
func libraryOn(t *testing.T, dir string, store vector.Store, emb langai.ImageEmbedder, sampler media.Sampler, opts ...search.MediaOption) *Library {
	t.Helper()
	coord := vector.Coord{Index: "examples", Partition: "local", Collection: "media"}
	ix, err := search.NewMediaIndex(emb, store, coord.Index, coord.Partition, coord.Collection, append([]search.MediaOption{search.WithSampler(sampler)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	cache := t.TempDir()
	lib, err := New(Config{
		Dir: dir, Index: ix, Sampler: sampler, CatalogFile: filepath.Join(cache, "catalog.json"), CacheDir: filepath.Join(cache, "thumbs"),
		Info:   Info{Model: "fake", Dimensions: emb.Dimensions(), Provider: "cpu"},
		Stored: func(ctx context.Context) (string, error) { return store.Model(ctx, coord) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return lib
}

// mismatched is the fixture's directory indexed by one model, and a Library
// of another model over the same store.
func mismatched(t *testing.T) (*Library, *otherImageModel) {
	t.Helper()
	f := newFixture(t)
	store := vector.NewMemStore()
	first := libraryOn(t, f.dir, store, &countingEmbedder{}, f.sampler)
	if _, err := first.Scan(context.Background()); err != nil {
		t.Fatal(err)
	}
	other := &otherImageModel{}
	return libraryOn(t, f.dir, store, other, f.sampler), other
}

func TestScan_AnotherModelStopsTheScanUntilReset(t *testing.T) {
	ctx := context.Background()
	lib, emb := mismatched(t)
	stats, err := lib.Scan(ctx)
	if !errors.Is(err, vector.ErrModelMismatch) {
		t.Fatalf("scan over another model's index: %v", err)
	}
	// the index asks the store for the collection's model before it decodes
	// or embeds anything: the first file stops the scan, no file counts as
	// failed, nothing was embedded, and every file is still listed
	if !stats.Mismatch || stats.Error == "" || stats.Indexed != 0 || stats.Failed != 0 {
		t.Errorf("stats: %+v", stats)
	}
	if got := lib.LastScan(); !got.Mismatch {
		t.Errorf("last scan: %+v", got)
	}
	if n := emb.calls.Load(); n != 0 {
		t.Errorf("the mismatch cost %d embeddings", n)
	}
	if items := lib.Items(); len(items) != 4 {
		t.Errorf("catalog holds %d items, want all 4", len(items))
	}
	if it, _ := lib.Item("broken.jpg"); it.Error != "" {
		t.Errorf("the file the scan stopped at carries the collection's error: %+v", it)
	}
	if got := lib.StoredModel(ctx); got != langaitest.FakeImageEmbedderModel {
		t.Errorf("stored model %q, want the one that built the index", got)
	}
	if _, err := lib.Similar(ctx, "red.png", 0, false); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("search over another model's index: %v", err)
	}

	if err := lib.Reset(ctx); err != nil {
		t.Fatal(err)
	}
	stats, err = lib.Scan(ctx)
	if err != nil || stats.Mismatch || stats.Indexed != 3 || stats.Failed != 1 {
		t.Errorf("scan after the reset: %+v, %v", stats, err)
	}
	if got := lib.StoredModel(ctx); got != "another-image-model" {
		t.Errorf("stored model after the rebuild: %q", got)
	}
}

func post(t *testing.T, url string) (int, []byte) {
	t.Helper()
	resp, err := http.Post(url, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, body
}

func TestHandler_AMismatchIsAConflictWithAReset(t *testing.T) {
	lib, _ := mismatched(t)
	srv := httptest.NewServer(Handler(lib))
	t.Cleanup(srv.Close)
	var e struct {
		Error    string `json:"error"`
		Mismatch bool   `json:"mismatch"`
	}
	status, body := post(t, srv.URL+"/api/rescan")
	decode(t, body, &e)
	if status != 409 || !e.Mismatch || e.Error == "" {
		t.Errorf("rescan over another model's index: %d %+v", status, e)
	}
	_, body, _ = get(t, srv.URL+"/api/info")
	var info struct {
		Stored   string    `json:"storedModel"`
		LastScan ScanStats `json:"lastScan"`
	}
	decode(t, body, &info)
	if !info.LastScan.Mismatch || info.Stored != langaitest.FakeImageEmbedderModel {
		t.Errorf("info after a mismatch: %+v", info)
	}
	status, body, _ = get(t, srv.URL+"/api/similar/red.png")
	e.Mismatch = false
	decode(t, body, &e)
	if status != 409 || !e.Mismatch {
		t.Errorf("similar over another model's index: %d %s", status, body)
	}

	lib.scanning.Store(true)
	if status, _ := post(t, srv.URL+"/api/reset"); status != 409 {
		t.Errorf("reset during a scan: %d", status)
	}
	lib.scanning.Store(false)
	if status, _ := post(t, srv.URL+"/api/reset"); status != 204 {
		t.Errorf("reset: %d", status)
	}
	status, body = post(t, srv.URL+"/api/rescan")
	var stats ScanStats
	decode(t, body, &stats)
	if status != 200 || stats.Indexed != 3 || stats.Mismatch {
		t.Errorf("rescan after the reset: %d %s", status, body)
	}
}

func TestScan_NamesAnImageThatIsTooLarge(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	// no picture of the fixture fits in 64 bytes
	lib := libraryOn(t, f.dir, vector.NewMemStore(), &countingEmbedder{}, f.sampler, search.WithMaxImageBytes(64))
	stats, err := lib.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Failed != 3 || stats.Indexed != 1 { // the two pictures and the broken file fail, the clip is indexed
		t.Errorf("stats: %+v", stats)
	}
	it, _ := lib.Item("red.png")
	if it.Reason != ReasonTooLarge || it.Error == "" {
		t.Errorf("red.png: %+v", it)
	}
	// a file that fails for another reason has none: it is broken, not large
	if it, _ := lib.Item("broken.jpg"); it.Reason != "" || it.Error == "" {
		t.Errorf("broken.jpg: %+v", it)
	}

	// an upload over the limit is refused as what it is
	srv := httptest.NewServer(Handler(lib))
	t.Cleanup(srv.Close)
	status, body := upload(t, srv.URL+"/api/search", "big.png", pngBytes(t, pic(64, 48, red, green)))
	var e struct {
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(body, &e); err != nil {
		t.Fatal(err)
	}
	if status != http.StatusRequestEntityTooLarge || e.Reason != ReasonTooLarge {
		t.Errorf("upload over the limit: %d %s", status, body)
	}
}

func TestScan_NamesAVideoWithoutFrames(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	// the sampler takes no frame from the clip, nor from an upload
	empty := &fakeSampler{frames: map[string][]media.Frame{"clip.mp4": {}, "upload": {}}}
	lib := libraryOn(t, f.dir, vector.NewMemStore(), &countingEmbedder{}, empty)
	if _, err := lib.Scan(ctx); err != nil {
		t.Fatal(err)
	}
	it, _ := lib.Item("clip.mp4")
	if it.Reason != ReasonNoFrames || it.Error == "" {
		t.Errorf("clip.mp4: %+v", it)
	}
	// nothing is stored for it, so the next scan tries it again
	if _, err := lib.Similar(ctx, "clip.mp4", 0, false); !errors.Is(err, search.ErrNotIndexed) {
		t.Errorf("similar to a video without frames: %v", err)
	}
	if reasonOf(errors.New("anything else")) != "" {
		t.Error("an unnamed error got a reason")
	}

	// a query video without frames has nothing to search with: no hits
	res, err := lib.Search(ctx, "query.mp4", strings.NewReader("a video no frame comes out of"), 0)
	if err != nil || len(res.Hits) != 0 || res.Frames != 0 {
		t.Errorf("search with a video without frames: %+v, %v", res, err)
	}
}

func TestWriteError_NamesTheLibrarysFailures(t *testing.T) {
	for _, c := range []struct {
		err    error
		status int
		key    string
	}{
		{fmt.Errorf("index: %w", search.ErrNoFrames), http.StatusUnprocessableEntity, "reason"},
		{fmt.Errorf("index: %w", search.ErrTooLarge), http.StatusRequestEntityTooLarge, "reason"},
		{fmt.Errorf("index: %w", vector.ErrModelMismatch), http.StatusConflict, "mismatch"},
	} {
		rec := httptest.NewRecorder()
		writeError(rec, c.err)
		var body map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		if rec.Code != c.status || body[c.key] == nil || body["error"] == "" {
			t.Errorf("%v: %d %v", c.err, rec.Code, body)
		}
	}
}

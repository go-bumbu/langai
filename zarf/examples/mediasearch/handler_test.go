package mediasearch

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
)

func serve(t *testing.T) (*fixture, *httptest.Server) {
	t.Helper()
	f := scanned(t)
	srv := httptest.NewServer(Handler(f.lib))
	t.Cleanup(srv.Close)
	return f, srv
}

func get(t *testing.T, url string) (int, []byte, string) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, body, resp.Header.Get("Content-Type")
}

func decode(t *testing.T, body []byte, v any) {
	t.Helper()
	if err := json.Unmarshal(body, v); err != nil {
		t.Fatalf("%v: %s", err, body)
	}
}

func TestHandler_PageInfoAndItems(t *testing.T) {
	_, srv := serve(t)
	status, body, ct := get(t, srv.URL+"/")
	if status != 200 || !strings.Contains(ct, "text/html") || !bytes.Contains(body, []byte("Media search")) {
		t.Errorf("page: %d %s", status, ct)
	}
	for _, file := range []string{"app.js", "style.css"} {
		if status, _, _ := get(t, srv.URL+"/"+file); status != 200 {
			t.Errorf("%s: %d", file, status)
		}
	}
	status, body, _ = get(t, srv.URL+"/api/info")
	var info struct {
		Info   Info `json:"info"`
		Images int  `json:"images"`
		Videos int  `json:"videos"`
		Limit  int  `json:"limit"`
	}
	decode(t, body, &info)
	if status != 200 || info.Info.Model != "fake" || info.Images != 3 || info.Videos != 1 || info.Limit != 12 {
		t.Errorf("info: %d %+v", status, info)
	}
	status, body, _ = get(t, srv.URL+"/api/items")
	var items []Item
	decode(t, body, &items)
	if status != 200 || len(items) != 4 {
		t.Errorf("items: %d %d", status, len(items))
	}
}

func TestHandler_SimilarStoredOrReembedded(t *testing.T) {
	_, srv := serve(t)
	status, body, _ := get(t, srv.URL+"/api/similar/sub%20dir/green%20blue.png?limit=2&embed=1")
	var res Result
	decode(t, body, &res)
	if status != 200 || len(res.Hits) != 2 || !res.Hits[0].Self || res.Stored || res.Frames != 1 {
		t.Errorf("similar, re-embedded: %d %+v", status, res)
	}
	// the broken file has no stored vectors, and fails to decode when re-embedded
	if status, body, _ := get(t, srv.URL+"/api/similar/broken.jpg"); status != 404 || !bytes.Contains(body, []byte("no stored vectors")) {
		t.Errorf("broken, stored: %d %s", status, body)
	}
	if status, _, _ := get(t, srv.URL+"/api/similar/broken.jpg?embed=1"); status != 500 {
		t.Errorf("broken, re-embedded: %d", status)
	}
}

func TestHandler_SimilarAndPictures(t *testing.T) {
	_, srv := serve(t)
	status, body, _ := get(t, srv.URL+"/api/similar/sub%20dir/green%20blue.png?limit=2")
	var res Result
	decode(t, body, &res)
	if status != 200 || len(res.Hits) != 2 || res.Hits[0].Self || !res.Stored {
		t.Errorf("similar: %d %+v", status, res)
	}
	if status, body, _ := get(t, srv.URL+"/api/similar/nope.png"); status != 404 || !bytes.Contains(body, []byte("error")) {
		t.Errorf("unknown: %d %s", status, body)
	}

	if status, _, ct := get(t, srv.URL+"/thumb/sub%20dir/green%20blue.png"); status != 200 || ct != "image/jpeg" {
		t.Errorf("thumb: %d %s", status, ct)
	}
	if status, _, _ := get(t, srv.URL+"/thumb/nope.png"); status != 404 {
		t.Errorf("thumb unknown: %d", status)
	}
	if status, _, ct := get(t, srv.URL+"/storyboard/clip.mp4"); status != 200 || ct != "image/jpeg" {
		t.Errorf("storyboard: %d %s", status, ct)
	}
	if status, _, _ := get(t, srv.URL+"/storyboard/red.png"); status != 400 {
		t.Errorf("storyboard of an image: %d", status)
	}
	status, body, _ = get(t, srv.URL+"/api/frames/clip.mp4")
	var board Board
	decode(t, body, &board)
	if status != 200 || len(board.Times) != 2 {
		t.Errorf("frames: %d %+v", status, board)
	}
	if status, _, ct := get(t, srv.URL+"/file/red.png"); status != 200 || ct != "image/png" {
		t.Errorf("file: %d %s", status, ct)
	}
	if status, _, _ := get(t, srv.URL+"/file/notes.txt"); status != 404 {
		t.Errorf("file outside the catalog: %d", status)
	}
}

func upload(t *testing.T, url, name string, content []byte) (int, []byte) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	part, err := mw.CreateFormFile("file", name)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write(content)
	_ = mw.Close()
	resp, err := http.Post(url, mw.FormDataContentType(), &buf)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, body
}

func TestHandler_SearchUploadAndRescan(t *testing.T) {
	f, srv := serve(t)
	status, body := upload(t, srv.URL+"/api/search?limit=1", "q.png", pngBytes(t, pic(32, 24, green, blue)))
	var res Result
	decode(t, body, &res)
	if status != 200 || len(res.Hits) != 1 || res.Hits[0].Item.ID != "sub dir/green blue.png" {
		t.Errorf("upload: %d %+v", status, res)
	}
	if status, _ := upload(t, srv.URL+"/api/search", "q.txt", []byte("x")); status != 400 {
		t.Errorf("text upload: %d", status)
	}
	resp, err := http.Post(srv.URL+"/api/search", "text/plain", strings.NewReader("x"))
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Errorf("no multipart: %d", resp.StatusCode)
	}

	resp, err = http.Post(srv.URL+"/api/rescan", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	body, _ = io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	var stats ScanStats
	decode(t, body, &stats)
	if resp.StatusCode != 200 || stats.Skipped != 3 {
		t.Errorf("rescan: %d %+v", resp.StatusCode, stats)
	}
	f.lib.scanning.Store(true)
	resp, err = http.Post(srv.URL+"/api/rescan", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != 409 {
		t.Errorf("rescan while scanning: %d", resp.StatusCode)
	}
	f.lib.scanning.Store(false)
}

// TestHandler_TextSearch: a text query goes through the index's text
// embedder; without the query parameter it is a 400, without a text
// embedder a 500 naming the option.
func TestHandler_TextSearch(t *testing.T) {
	_, srv := serve(t)
	status, body, _ := get(t, srv.URL+"/api/text?q=green&limit=2")
	var res Result
	decode(t, body, &res)
	ids := map[string]bool{}
	for _, h := range res.Hits {
		ids[h.Item.ID] = true
	}
	// the green-and-blue photo and the clip's green-and-blue frame, the red
	// ones left behind
	if status != 200 || res.Text != "green" || len(res.Hits) != 2 || !ids["sub dir/green blue.png"] || !ids["clip.mp4"] {
		t.Errorf("text: %d %+v", status, res)
	}
	if status, _, _ := get(t, srv.URL+"/api/text"); status != 400 {
		t.Errorf("no q: %d", status)
	}
	f := newFixture(t) // no text embedder on this one
	ix, _ := search.NewMediaIndex(langaitest.FakeImageEmbedder{}, vector.NewMemStore(), "i", "p", "c")
	f.lib.cfg.Index = ix
	if _, err := f.lib.SearchText(context.Background(), "green", 0); err == nil || !strings.Contains(err.Error(), "WithTextEmbedder") {
		t.Errorf("no text embedder: %v", err)
	}
}

func TestHandler_InfoSaysWhereTheModelAndTheScanStand(t *testing.T) {
	f := scanned(t)
	cfg := f.lib.cfg
	cfg.ModelState = func() string { return "not started" }
	lib, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	lib.scanning.Store(true)
	srv := httptest.NewServer(Handler(lib))
	t.Cleanup(srv.Close)
	_, body, _ := get(t, srv.URL+"/api/info")
	var info struct {
		Scanning   bool   `json:"scanning"`
		ModelState string `json:"modelState"`
	}
	decode(t, body, &info)
	if !info.Scanning || info.ModelState != "not started" {
		t.Errorf("info: %+v", info)
	}
	if f.lib.Scanning() || f.lib.ModelState() != "" {
		t.Errorf("a library that is idle and was told nothing: %v %q", f.lib.Scanning(), f.lib.ModelState())
	}
}

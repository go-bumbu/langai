package mediasearch

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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

func TestHandler_SimilarAndPictures(t *testing.T) {
	_, srv := serve(t)
	status, body, _ := get(t, srv.URL+"/api/similar/sub%20dir/green%20blue.png?limit=2")
	var res Result
	decode(t, body, &res)
	if status != 200 || len(res.Hits) != 2 || !res.Hits[0].Self {
		t.Errorf("similar: %d %+v", status, res)
	}
	if status, body, _ := get(t, srv.URL+"/api/similar/nope.png"); status != 404 || !bytes.Contains(body, []byte("error")) {
		t.Errorf("unknown: %d %s", status, body)
	}
	if status, _, _ := get(t, srv.URL+"/api/similar/broken.jpg"); status != 500 {
		t.Errorf("broken: %d", status)
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

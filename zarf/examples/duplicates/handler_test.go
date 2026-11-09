package duplicates

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
	srv := httptest.NewServer(Handler(f.finder))
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

func decodeJSON(t *testing.T, body []byte, v any) {
	t.Helper()
	if err := json.Unmarshal(body, v); err != nil {
		t.Fatalf("%v: %s", err, body)
	}
}

// post sends data as the multipart field name.
func post(t *testing.T, url, field string, data []byte) (int, []byte) {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	if field != "" {
		part, err := w.CreateFormFile(field, "query.jpg")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = part.Write(data)
	}
	_ = w.Close()
	resp, err := http.Post(url, w.FormDataContentType(), &buf)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, body
}

func TestHandler_PageInfoAndFiles(t *testing.T) {
	_, srv := serve(t)
	status, body, ct := get(t, srv.URL+"/")
	if status != 200 || !strings.Contains(ct, "text/html") || !bytes.Contains(body, []byte("Duplicates")) {
		t.Errorf("page: %d %s", status, ct)
	}
	for _, file := range []string{"app.js", "style.css"} {
		if status, _, _ := get(t, srv.URL+"/"+file); status != 200 {
			t.Errorf("%s: %d", file, status)
		}
	}
	status, body, _ = get(t, srv.URL+"/api/info")
	var info struct {
		Kinds    []Kind            `json:"kinds"`
		Default  string            `json:"default"`
		Pictures int               `json:"pictures"`
		Failed   map[string]string `json:"failed"`
		Limit    int               `json:"limit"`
		LastScan ScanStats         `json:"lastScan"`
		Scanning bool              `json:"scanning"`
	}
	decodeJSON(t, body, &info)
	if status != 200 || len(info.Kinds) != 4 || info.Default != DefaultKind || info.Pictures != 6 || info.Failed["broken.jpg"] == "" ||
		info.Limit != 12 || info.LastScan.Hashed != 5 || info.Scanning {
		t.Errorf("info: %d %+v", status, info)
	}
	// a picture of the catalog is served, by its path with a space in it
	status, _, ct = get(t, srv.URL+"/file/sub%20dir/a-small.jpg")
	if status != 200 || ct != "image/jpeg" {
		t.Errorf("file: %d %s", status, ct)
	}
	if status, _, _ := get(t, srv.URL+"/file/notes.txt"); status != 404 {
		t.Errorf("a file outside the catalog: %d", status)
	}
}

func TestHandler_GroupsAtTwoThresholds(t *testing.T) {
	_, srv := serve(t)
	status, body, _ := get(t, srv.URL+"/api/groups?kind=dhash-64&threshold=0")
	var g Groups
	decodeJSON(t, body, &g)
	if status != 200 || g.Kind.Name != "dhash-64" || len(g.Groups) != 1 || len(g.Groups[0].Members) != 2 || !g.Groups[0].Exact {
		t.Errorf("groups at 0: %d %+v", status, g)
	}
	// no parameters: the default hash at its own threshold
	status, body, _ = get(t, srv.URL+"/api/groups")
	g = Groups{}
	decodeJSON(t, body, &g)
	if status != 200 || g.Kind.Name != DefaultKind || g.Threshold != 4 || len(g.Groups) != 1 || len(g.Groups[0].Members) != 3 {
		t.Errorf("default groups: %d %+v", status, g)
	}
	for _, q := range []string{"?threshold=x", "?threshold=-2", "?threshold=999", "?kind=md5"} {
		if status, _, _ := get(t, srv.URL+"/api/groups"+q); status != 400 {
			t.Errorf("groups%s: %d", q, status)
		}
	}
}

func TestHandler_SearchWithAnUpload(t *testing.T) {
	_, srv := serve(t)
	status, body := post(t, srv.URL+"/api/search?kind=phash-64&threshold=4&limit=3", "file", upload(t))
	var res Result
	decodeJSON(t, body, &res)
	if status != 200 || len(res.Hits) != 3 || res.Copies != 3 || res.QueryHash == "" {
		t.Errorf("search: %d %+v", status, res)
	}
	if status, _ := post(t, srv.URL+"/api/search", "", nil); status != 400 {
		t.Errorf("search without a file: %d", status)
	}
	for _, q := range []string{"?threshold=x", "?limit=x", "?kind=md5"} {
		if status, _ := post(t, srv.URL+"/api/search"+q, "file", upload(t)); status != 400 {
			t.Errorf("search%s: %d", q, status)
		}
	}
	if status, _ := post(t, srv.URL+"/api/search", "file", []byte("not a picture")); status != 500 {
		t.Errorf("search with a file that is no picture: %d", status)
	}
}

func TestHandler_SimilarToACatalogPicture(t *testing.T) {
	_, srv := serve(t)
	status, body, _ := get(t, srv.URL+"/api/similar/sub%20dir/a-small.jpg?threshold=6")
	var res Result
	decodeJSON(t, body, &res)
	if status != 200 || !res.Stored || res.Query == nil || res.Query.ID != "sub dir/a-small.jpg" || res.Copies != 2 || res.Threshold != 6 {
		t.Errorf("similar: %d %+v", status, res)
	}
	if status, _, _ := get(t, srv.URL+"/api/similar/nope.png"); status != 404 {
		t.Errorf("similar to an unknown picture: %d", status)
	}
	if status, _, _ := get(t, srv.URL+"/api/similar/broken.jpg"); status != 404 {
		t.Errorf("similar to a picture that is not indexed: %d", status)
	}
	for _, q := range []string{"?threshold=x", "?limit=x"} {
		if status, _, _ := get(t, srv.URL+"/api/similar/a.png"+q); status != 400 {
			t.Errorf("similar%s: %d", q, status)
		}
	}
}

func TestHandler_Rescan(t *testing.T) {
	f, srv := serve(t)
	resp, err := http.Post(srv.URL+"/api/rescan", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	var stats ScanStats
	decodeJSON(t, body, &stats)
	if resp.StatusCode != 200 || stats.Unchanged != 5 || stats.Failed != 1 {
		t.Errorf("rescan: %d %+v", resp.StatusCode, stats)
	}
	f.finder.scanning.Store(true)
	resp, err = http.Post(srv.URL+"/api/rescan", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != 409 {
		t.Errorf("rescan during a scan: %d", resp.StatusCode)
	}
}

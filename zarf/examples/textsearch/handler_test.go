package textsearch

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
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

func do(t *testing.T, method, url string) (int, []byte, string) {
	t.Helper()
	req, err := http.NewRequest(method, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
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

func TestHandler_PageInfoAndDocs(t *testing.T) {
	_, srv := serve(t)
	status, body, ct := do(t, "GET", srv.URL+"/")
	if status != 200 || !strings.Contains(ct, "text/html") || !bytes.Contains(body, []byte("Text search")) {
		t.Errorf("page: %d %s", status, ct)
	}
	for _, file := range []string{"app.js", "style.css"} {
		if status, _, _ := do(t, "GET", srv.URL+"/"+file); status != 200 {
			t.Errorf("%s: %d", file, status)
		}
	}
	status, body, _ = do(t, "GET", srv.URL+"/api/info")
	var info struct {
		Info       Info      `json:"info"`
		Docs       int       `json:"docs"`
		Chunks     int       `json:"chunks"`
		Limit      int       `json:"limit"`
		Scanning   bool      `json:"scanning"`
		ModelState string    `json:"modelState"`
		Examples   []string  `json:"examples"`
		LastScan   ScanStats `json:"lastScan"`
		Stored     *string   `json:"storedModel"`
	}
	decode(t, body, &info)
	if status != 200 || info.Info.Model != "fake" || info.Info.Chunker != "markdown/2 max=1200" || info.Docs != 3 || info.Chunks != 5 ||
		info.Limit != 10 || info.Scanning || info.ModelState != "loaded" || len(info.Examples) != 1 || info.LastScan.Indexed != 3 || info.Stored != nil {
		t.Errorf("info: %d %+v", status, info)
	}
	status, body, _ = do(t, "GET", srv.URL+"/api/docs")
	var docs []Doc
	decode(t, body, &docs)
	if status != 200 || len(docs) != 3 {
		t.Errorf("docs: %d %d", status, len(docs))
	}
}

func TestHandler_SearchSimilarAndDoc(t *testing.T) {
	_, srv := serve(t)
	status, body, _ := do(t, "GET", srv.URL+"/api/search?q=sourdough+starter&limit=1")
	var res Result
	decode(t, body, &res)
	if status != 200 || len(res.Hits) != 1 || res.Hits[0].Doc.ID != "bread.md" || res.Hits[0].Heading == "" {
		t.Errorf("search: %d %+v", status, res)
	}
	if status, _, _ := do(t, "GET", srv.URL+"/api/search"); status != 400 {
		t.Errorf("search without a query: %d", status)
	}

	// an id with a slash and a space is a path
	status, body, _ = do(t, "GET", srv.URL+"/api/similar/sub%20dir/hike.md")
	res = Result{}
	decode(t, body, &res)
	if status != 200 || !res.Stored || res.Doc == nil || res.Doc.ID != "sub dir/hike.md" || len(res.Hits) != 2 {
		t.Errorf("similar: %d %+v", status, res)
	}
	if status, _, _ := do(t, "GET", srv.URL+"/api/similar/nope.md"); status != 404 {
		t.Errorf("similar to an unknown document: %d", status)
	}

	status, body, _ = do(t, "GET", srv.URL+"/api/doc/sub%20dir/hike.md")
	var view DocView
	decode(t, body, &view)
	if status != 200 || view.Doc.Title != "Hut to hut" || len(view.Chunks) != 1 || view.Chunks[0].Heading != "Hut to hut" {
		t.Errorf("doc: %d %+v", status, view)
	}
	if status, _, _ := do(t, "GET", srv.URL+"/api/doc/..%2f..%2fetc%2fpasswd"); status != 404 {
		t.Errorf("a path outside the catalog: %d", status)
	}
}

func TestHandler_RescanAndReset(t *testing.T) {
	f, srv := serve(t)
	write(t, filepath.Join(f.dir, "new.md"), "# New\n\nA new document about taxes.\n")
	status, body, _ := do(t, "POST", srv.URL+"/api/rescan")
	var stats ScanStats
	decode(t, body, &stats)
	if status != 200 || stats.Indexed != 1 || stats.Unchanged != 3 {
		t.Errorf("rescan: %d %+v", status, stats)
	}

	f.lib.scanning.Store(true)
	if status, _, _ := do(t, "POST", srv.URL+"/api/rescan"); status != 409 {
		t.Errorf("rescan during a scan: %d", status)
	}
	if status, _, _ := do(t, "POST", srv.URL+"/api/reset"); status != 409 {
		t.Errorf("reset during a scan: %d", status)
	}
	f.lib.scanning.Store(false)

	if status, _, _ := do(t, "POST", srv.URL+"/api/reset"); status != 204 {
		t.Errorf("reset: %d", status)
	}
	// the index is empty: more like this has nothing stored to search with
	if status, _, _ := do(t, "GET", srv.URL+"/api/similar/bread.md"); status != 404 {
		t.Errorf("similar after a reset: %d", status)
	}
}

func TestHandler_AMismatchIsAConflictThatNamesTheStoredModel(t *testing.T) {
	f := scanned(t)
	other := f.library(t, otherModel{})
	write(t, filepath.Join(f.dir, "bread.md"), breadDoc+"\nMore.\n")
	if _, err := other.Scan(t.Context()); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(Handler(other))
	t.Cleanup(srv.Close)
	status, body, _ := do(t, "GET", srv.URL+"/api/search?q=bread")
	var e struct {
		Error    string `json:"error"`
		Mismatch bool   `json:"mismatch"`
	}
	decode(t, body, &e)
	if status != 409 || !e.Mismatch || e.Error == "" {
		t.Errorf("search with another model: %d %+v", status, e)
	}
	_, body, _ = do(t, "GET", srv.URL+"/api/info")
	var info struct {
		Stored   string    `json:"storedModel"`
		LastScan ScanStats `json:"lastScan"`
	}
	decode(t, body, &info)
	if !info.LastScan.Mismatch || info.Stored != "sensetest/fake-embedder" {
		t.Errorf("info after a mismatch: %+v", info)
	}
}

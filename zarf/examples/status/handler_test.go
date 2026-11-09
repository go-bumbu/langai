package status

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/search"
)

func serve(t *testing.T) (*fixture, *httptest.Server) {
	t.Helper()
	f := newFixture(t)
	srv := httptest.NewServer(Handler(f.svc))
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

func TestHandler_PageAndStatus(t *testing.T) {
	_, srv := serve(t)
	status, body, ct := do(t, "GET", srv.URL+"/")
	if status != 200 || !strings.Contains(ct, "text/html") || !bytes.Contains(body, []byte("Status")) {
		t.Errorf("page: %d %s", status, ct)
	}
	for _, file := range []string{"app.js", "style.css"} {
		if status, _, _ := do(t, "GET", srv.URL+"/"+file); status != 200 {
			t.Errorf("%s: %d", file, status)
		}
	}
	status, body, ct = do(t, "GET", srv.URL+"/api/status")
	var snap Snapshot
	decode(t, body, &snap)
	if status != 200 || !strings.Contains(ct, "application/json") || len(snap.Models) != 4 || len(snap.Collections) != 2 || snap.Cache == nil || len(snap.Pages) != 1 {
		t.Errorf("status: %d %s %+v", status, ct, snap)
	}
}

func TestHandler_LoadResetAndPrune(t *testing.T) {
	f, srv := serve(t)
	status, body, _ := do(t, "POST", srv.URL+"/api/models/text/load")
	var m ModelStatus
	decode(t, body, &m)
	if status != 200 || m.State != "loaded" {
		t.Errorf("load: %d %+v", status, m)
	}
	// a failed load is a state the page shows, not a failed request
	status, body, _ = do(t, "POST", srv.URL+"/api/models/image/load")
	decode(t, body, &m)
	if status != 200 || m.State != "failed" || m.Error == "" || m.Recovery == "" {
		t.Errorf("failed load: %d %+v", status, m)
	}
	if status, _, _ := do(t, "POST", srv.URL+"/api/models/hash/load"); status != 400 {
		t.Errorf("load of a model without a loader: %d", status)
	}
	if status, _, _ := do(t, "POST", srv.URL+"/api/models/nope/load"); status != 404 {
		t.Errorf("load of an unknown model: %d", status)
	}

	if _, err := f.index.IndexBatch(t.Context(), []search.Doc{{ID: "a", Text: "bread"}}); err != nil {
		t.Fatal(err)
	}
	status, body, _ = do(t, "POST", srv.URL+"/api/collections/docs/reset")
	var c CollectionStatus
	decode(t, body, &c)
	if status != 200 || c.Docs != 0 || f.resets != 1 {
		t.Errorf("reset: %d %+v", status, c)
	}
	if status, _, _ := do(t, "POST", srv.URL+"/api/collections/fixed/reset"); status != 400 {
		t.Errorf("reset of a collection without one: %d", status)
	}
	if status, _, _ := do(t, "POST", srv.URL+"/api/collections/nope/reset"); status != 404 {
		t.Errorf("reset of an unknown collection: %d", status)
	}

	status, body, _ = do(t, "POST", srv.URL+"/api/cache/prune")
	var cache CacheStatus
	decode(t, body, &cache)
	if status != 200 || cache.LastPrune == nil || cache.LastPrune.Removed != 3 {
		t.Errorf("prune: %d %+v", status, cache)
	}
	// an action is a POST: a GET of its path finds no file
	if status, _, _ := do(t, "GET", srv.URL+"/api/cache/prune"); status != 404 {
		t.Errorf("GET on an action: %d", status)
	}
}

func TestHandler_AFailingActionIsAnError(t *testing.T) {
	svc, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(Handler(svc))
	t.Cleanup(srv.Close)
	status, body, _ := do(t, "POST", srv.URL+"/api/cache/prune")
	if status != 400 || !bytes.Contains(body, []byte("not available")) {
		t.Errorf("prune without a cache: %d %s", status, body)
	}
}

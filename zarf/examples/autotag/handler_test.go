package autotag

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func do(t *testing.T, method, url, body string) (int, []byte) {
	t.Helper()
	req, err := http.NewRequest(method, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, b
}

func decode(t *testing.T, body []byte, v any) {
	t.Helper()
	if err := json.Unmarshal(body, v); err != nil {
		t.Fatalf("%v: %s", err, body)
	}
}

func serve(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(Handler(newNotebook(t, "")))
	t.Cleanup(srv.Close)
	return srv
}

func TestHandler_PageInfoAndSamples(t *testing.T) {
	srv := serve(t)
	status, body := do(t, "GET", srv.URL+"/", "")
	if status != 200 || !bytes.Contains(body, []byte("Auto tagging")) {
		t.Errorf("page: %d", status)
	}
	for _, file := range []string{"app.js", "style.css"} {
		if status, _ := do(t, "GET", srv.URL+"/"+file, ""); status != 200 {
			t.Errorf("%s: %d", file, status)
		}
	}
	status, body = do(t, "GET", srv.URL+"/api/info", "")
	var info struct {
		Info Info `json:"info"`
	}
	decode(t, body, &info)
	if status != 200 || !info.Info.Samples || info.Info.MinScore != 0.3 {
		t.Errorf("info: %d %s", status, body)
	}
	var st SampleStats
	status, body = do(t, "POST", srv.URL+"/api/samples", "")
	decode(t, body, &st)
	if status != 200 || st.Notes != 84 {
		t.Errorf("samples: %d %s", status, body)
	}
}

func TestHandler_TagsAndNotes(t *testing.T) {
	srv := serve(t)
	var tag Tag
	status, body := do(t, "POST", srv.URL+"/api/tags", `{"name":"cooking / bread"}`)
	decode(t, body, &tag)
	if status != 200 || tag.ID != "cooking-bread" {
		t.Fatalf("add tag: %d %s", status, body)
	}
	var note Note
	status, body = do(t, "POST", srv.URL+"/api/notes", `{"title":"Loaf","text":"bread and more bread"}`)
	decode(t, body, &note)
	if status != 200 || note.ID == "" {
		t.Fatalf("add note: %d %s", status, body)
	}
	if status, body := do(t, "PUT", srv.URL+"/api/notes/"+note.ID+"/tags/"+tag.ID, ""); status != 204 {
		t.Errorf("tag note: %d %s", status, body)
	}
	var v View
	status, body = do(t, "GET", srv.URL+"/api/notebook", "")
	decode(t, body, &v)
	if status != 200 || len(v.Notes) != 1 || len(v.Notes[0].Tags) != 1 || len(v.Notes[0].Auto) != 1 || v.Tags[0].Hand != 1 {
		t.Errorf("notebook: %d %s", status, body)
	}
	var rec Recommendation
	status, body = do(t, "GET", srv.URL+"/api/notes/"+note.ID+"/recommend", "")
	decode(t, body, &rec)
	if status != 200 || len(rec.Tags) != 3 {
		t.Errorf("recommend: %d %s", status, body)
	}
	if status, body := do(t, "PUT", srv.URL+"/api/tags/"+tag.ID, `{"name":"bread"}`); status != 200 {
		t.Errorf("rename: %d %s", status, body)
	}
	if status, _ := do(t, "DELETE", srv.URL+"/api/notes/"+note.ID+"/tags/"+tag.ID, ""); status != 204 {
		t.Errorf("untag: %d", status)
	}
	if status, _ := do(t, "DELETE", srv.URL+"/api/tags/"+tag.ID, ""); status != 204 {
		t.Errorf("delete tag: %d", status)
	}
	if status, _ := do(t, "DELETE", srv.URL+"/api/notes/"+note.ID, ""); status != 204 {
		t.Errorf("delete note: %d", status)
	}
}

func TestHandler_Errors(t *testing.T) {
	srv := serve(t)
	for _, c := range []struct {
		method, path, body string
		want               int
	}{
		{"POST", "/api/tags", `{"name":" "}`, 400},
		{"POST", "/api/tags", `not json`, 400},
		{"POST", "/api/notes", `{"text":""}`, 400},
		{"PUT", "/api/tags/nope", `{"name":"x"}`, 404},
		{"DELETE", "/api/tags/nope", ``, 404},
		{"DELETE", "/api/notes/nope", ``, 404},
		{"PUT", "/api/notes/nope/tags/nope", ``, 404},
		{"GET", "/api/notes/nope/recommend", ``, 404},
		{"POST", "/api/samples?hand=x", ``, 400},
		{"POST", "/api/samples?hand=-1", ``, 400},
	} {
		if status, body := do(t, c.method, srv.URL+c.path, c.body); status != c.want {
			t.Errorf("%s %s: %d %s, want %d", c.method, c.path, status, body, c.want)
		}
	}
}

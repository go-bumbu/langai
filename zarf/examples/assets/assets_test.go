package assets

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandler_ServesTheSharedFiles(t *testing.T) {
	srv := httptest.NewServer(http.StripPrefix("/assets", Handler()))
	t.Cleanup(srv.Close)
	for file, want := range map[string]string{"base.css": "--accent", "ui.js": "window.ui"} {
		resp, err := http.Get(srv.URL + "/assets/" + file)
		if err != nil {
			t.Fatal(err)
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != 200 || !strings.Contains(string(body), want) {
			t.Errorf("%s: status %d, %q missing", file, resp.StatusCode, want)
		}
	}
	resp, err := http.Get(srv.URL + "/assets/missing.css")
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Errorf("missing file: status %d, want 404", resp.StatusCode)
	}
}

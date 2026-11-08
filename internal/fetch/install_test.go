package fetch

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testRecord = "install.json"

// libArchive serves a release tarball with one library and its symlink,
// plus whatever extra entries the test adds.
func libArchive(t *testing.T, extra ...tarEntry) Asset {
	t.Helper()
	entries := append([]tarEntry{
		{name: "lib/libx.so.1", body: "x"},
		{name: "lib/libx.so", link: "libx.so.1"},
		{name: "include/x.h", body: "h"},
	}, extra...)
	_, a := serve(t, tarGzBytes(t, "rel-1.0", entries))
	return a
}

// install describes an install of a into dir with the test's extraction.
func install(dir string, a Asset) DirInstall {
	return DirInstall{
		Dir: dir, What: "libx 1.0", Archive: a, RecordFile: testRecord, Record: []byte(`{"v":1}`),
		Extract: func(archive, dir string) error {
			n, err := ExtractTarGz(archive, dir, wantLib, cap)
			if err == nil && n == 0 {
				err = errors.New("no libraries")
			}
			return err
		},
	}
}

// names lists the entries of dir, sorted.
func names(t *testing.T, dir string) string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = e.Name()
	}
	return strings.Join(out, ",")
}

func TestInstallDir_DownloadsExtractsAndRecords(t *testing.T) {
	a := libArchive(t)
	dir := filepath.Join(t.TempDir(), "libs", "1.0")
	if err := dl(cap).InstallDir(context.Background(), install(dir, a)); err != nil {
		t.Fatal(err)
	}
	if got := names(t, dir); got != testRecord+",libx.so,libx.so.1" {
		t.Errorf("installed %s", got)
	}
	if rec, _ := os.ReadFile(filepath.Join(dir, testRecord)); string(rec) != `{"v":1}` {
		t.Errorf("record = %q", rec)
	}
	if got := names(t, filepath.Dir(dir)); got != "1.0" {
		t.Errorf("left next to the directory: %s", got)
	}
}

// TestInstallDir_FailureLeavesNothing: a failed download or extraction never
// produces the directory, and leaves no staging behind.
func TestInstallDir_FailureLeavesNothing(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(srv.Close)
	_, empty := serve(t, tarGzBytes(t, "rel-1.0", []tarEntry{{name: "README", body: "no libs"}}))
	for name, c := range map[string]struct {
		a    Asset
		want string
	}{
		"download":   {Asset{URL: srv.URL + "/missing", SHA256: strings.Repeat("0", 64)}, "download libx 1.0"},
		"extraction": {empty, "extract libx 1.0"},
	} {
		dir := filepath.Join(t.TempDir(), "libs", "1.0")
		err := dl(cap).InstallDir(context.Background(), install(dir, c.a))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", name, err, c.want)
		}
		if got := names(t, filepath.Dir(dir)); got != "" {
			t.Errorf("%s: left behind: %s", name, got)
		}
	}
}

func TestEnsureFile(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	present := filepath.Join(dir, "present.bin")
	if err := os.WriteFile(present, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	// a present file is kept: nothing is requested
	if err := dl(cap).EnsureFile(ctx, "model m", Asset{URL: "http://127.0.0.1:1/unreachable", SHA256: strings.Repeat("0", 64)}, present); err != nil {
		t.Errorf("present file: %v", err)
	}
	absent := filepath.Join(dir, "absent.bin")
	err := dl(cap).EnsureFile(ctx, "model m", Asset{}, absent)
	if err == nil || !strings.Contains(err.Error(), "no URL") || !strings.Contains(err.Error(), absent) || !strings.Contains(err.Error(), "model m") {
		t.Errorf("absent without URL: err = %v", err)
	}
	_, a := serve(t, []byte("downloaded model"))
	dest := filepath.Join(dir, "sub", "model.bin")
	if err := dl(cap).EnsureFile(ctx, "model m", a, dest); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); string(got) != "downloaded model" {
		t.Errorf("downloaded %q", got)
	}
	a.SHA256 = strings.Repeat("1", 64)
	bad := filepath.Join(dir, "bad.bin")
	if err := dl(cap).EnsureFile(ctx, "model m", a, bad); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Errorf("bad digest: err = %v", err)
	}
	if _, err := os.Stat(bad); err == nil {
		t.Error("a file with the wrong digest appeared under its final name")
	}
}

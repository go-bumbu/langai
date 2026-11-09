package fetch

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"
)

// serve returns an httptest server that answers every path with body, and
// the asset describing it (digest and size filled in).
func serve(t *testing.T, body []byte) (*httptest.Server, Asset) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	sum := sha256.Sum256(body)
	return srv, Asset{URL: srv.URL + "/file", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(body))}
}

const cap = 1 << 20

// dl is a Downloader with the given cap and nothing else set.
func dl(limit int64) Downloader { return Downloader{Limit: limit} }

func TestFile_WritesVerifiedFile(t *testing.T) {
	_, a := serve(t, []byte("hello"))
	dest := filepath.Join(t.TempDir(), "out.bin")
	if err := dl(cap).File(context.Background(), "test", a, dest); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(dest)
	if err != nil || string(got) != "hello" {
		t.Fatalf("dest = %q, %v; want hello", got, err)
	}
	entries, _ := os.ReadDir(filepath.Dir(dest))
	if len(entries) != 1 {
		t.Fatalf("temp file left behind: %v", entries)
	}
}

func TestFile_RejectsDigestMismatch(t *testing.T) {
	_, a := serve(t, []byte("hello"))
	a.SHA256 = strings.Repeat("0", 64)
	dest := filepath.Join(t.TempDir(), "out.bin")
	err := dl(cap).File(context.Background(), "test", a, dest)
	if err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("err = %v, want a digest error", err)
	}
	if _, statErr := os.Stat(dest); statErr == nil {
		t.Fatal("a file with the wrong digest must not appear under its final name")
	}
	entries, _ := os.ReadDir(filepath.Dir(dest))
	if len(entries) != 0 {
		t.Fatalf("temp file left behind: %v", entries)
	}
}

func TestFile_RejectsWrongSizeAndHTTPErrors(t *testing.T) {
	_, a := serve(t, []byte("hello"))
	a.Size = 4
	err := dl(cap).File(context.Background(), "test", a, filepath.Join(t.TempDir(), "out.bin"))
	if err == nil || !strings.Contains(err.Error(), "bytes") {
		t.Fatalf("err = %v, want a size error", err)
	}
	srv := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(srv.Close)
	a = Asset{URL: srv.URL + "/missing", SHA256: strings.Repeat("0", 64)}
	err = dl(cap).File(context.Background(), "test", a, filepath.Join(t.TempDir(), "out.bin"))
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("err = %v, want the HTTP status", err)
	}
}

func TestFile_CapsTheDownloadAtTheLargerOfLimitAndSize(t *testing.T) {
	body := bytes.Repeat([]byte("x"), 100)
	_, a := serve(t, body)
	dest := filepath.Join(t.TempDir(), "out.bin")
	// the pinned size raises a smaller cap
	if err := dl(10).File(context.Background(), "test", a, dest); err != nil {
		t.Fatalf("a download of its pinned size must pass a smaller cap: %v", err)
	}
	a.Size = 0
	err := dl(10).File(context.Background(), "test", a, filepath.Join(t.TempDir(), "out.bin"))
	if err == nil || !strings.Contains(err.Error(), "cap") {
		t.Fatalf("err = %v, want the cap error", err)
	}
}

func TestFile_UsesTheClientAndUserAgent(t *testing.T) {
	var agent string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		agent = r.UserAgent()
		_, _ = w.Write([]byte("x"))
	}))
	t.Cleanup(srv.Close)
	sum := sha256.Sum256([]byte("x"))
	a := Asset{URL: srv.URL, SHA256: hex.EncodeToString(sum[:])}
	if err := (Downloader{Client: srv.Client(), UserAgent: "langai-test", Limit: cap}).File(context.Background(), "test", a, filepath.Join(t.TempDir(), "x")); err != nil {
		t.Fatal(err)
	}
	if agent != "langai-test" {
		t.Errorf("user agent = %q", agent)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := dl(cap).File(ctx, "test", a, filepath.Join(t.TempDir(), "x")); err == nil {
		t.Error("a cancelled context must fail the download")
	}
}

func TestUnpinned_ReportsDigestAndSize(t *testing.T) {
	body := []byte("pin me")
	_, a := serve(t, body)
	dest := filepath.Join(t.TempDir(), "sub", "out.bin")
	sum, n, err := dl(cap).Unpinned(context.Background(), a.URL, dest)
	if err != nil {
		t.Fatal(err)
	}
	if sum != a.SHA256 || n != int64(len(body)) {
		t.Errorf("got %s, %d; want %s, %d", sum, n, a.SHA256, len(body))
	}
	if got, _ := os.ReadFile(dest); string(got) != "pin me" {
		t.Errorf("dest holds %q", got)
	}
	srv := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(srv.Close)
	if _, _, err := dl(cap).Unpinned(context.Background(), srv.URL, filepath.Join(t.TempDir(), "x")); err == nil {
		t.Error("404: want an error")
	}
}

type tarEntry struct {
	name, link string // link set = symlink to link
	body       string
}

// tarGz writes tarGzBytes to a file and returns its path.
func tarGz(t *testing.T, dir string, entries []tarEntry) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "libs.tar.gz")
	if err := os.WriteFile(p, tarGzBytes(t, dir, entries), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// tarGzBytes builds a gzipped tarball in memory with entries nested under dir/,
// like a release archive.
func tarGzBytes(t *testing.T, dir string, entries []tarEntry) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		hdr := &tar.Header{Name: dir + "/" + e.name, Mode: 0o755}
		if e.link != "" {
			hdr.Typeflag = tar.TypeSymlink
			hdr.Linkname = e.link
		} else {
			hdr.Typeflag = tar.TypeReg
			hdr.Size = int64(len(e.body))
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if e.link == "" {
			if _, err := tw.Write([]byte(e.body)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// wantLib accepts the .so files directly under lib/.
func wantLib(p string) bool {
	return path.Base(path.Dir(p)) == "lib" && strings.Contains(path.Base(p), ".so")
}

func TestExtractTarGz_FlattensFiltersAndRecreatesSymlinks(t *testing.T) {
	archive := tarGz(t, "rel-1.0", []tarEntry{
		{name: "lib/libx.so.1.0", body: "x"},
		{name: "lib/libx.so.1", link: "libx.so.1.0"},
		{name: "lib/libx.so", link: "libx.so.1"},
		{name: "lib/liby.so", body: "y"},
		{name: "lib/debug/libx.so.1.0", body: "debug symbols"}, // same base name, other directory: skipped
		{name: "lib/pkgconfig/libx.pc", body: "pc"},
		{name: "include/x.h", body: "h"},
		{name: "bin/tool", body: "#!elf"},
	})
	dir := t.TempDir()
	n, err := ExtractTarGz(archive, dir, wantLib, cap)
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Errorf("extracted %d entries, want 4", n)
	}
	entries, _ := os.ReadDir(dir)
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	if want := "libx.so,libx.so.1,libx.so.1.0,liby.so"; strings.Join(names, ",") != want {
		t.Fatalf("extracted %v, want %s", names, want)
	}
	if target, err := os.Readlink(filepath.Join(dir, "libx.so")); err != nil || target != "libx.so.1" {
		t.Fatalf("libx.so -> %q, %v; want a symlink to libx.so.1", target, err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "libx.so")); string(got) != "x" {
		t.Fatalf("following the symlink chain read %q, want the real library", got)
	}
	if info, _ := os.Stat(filepath.Join(dir, "liby.so")); info.Mode()&0o100 == 0 {
		t.Error("an extracted library must be executable")
	}
}

// TestExtractTarGz_RejectsEscapingSymlinks: a symlink may only point at a
// bare name in the same directory that the filter accepts; anything else is
// an attack surface, not a packaging quirk.
func TestExtractTarGz_RejectsEscapingSymlinks(t *testing.T) {
	for _, link := range []string{"../../etc/ld.so.preload", "/usr/lib/libx.so", "tool", "../lib/libx.so"} {
		archive := tarGz(t, "x", []tarEntry{{name: "lib/libx.so", link: link}})
		if _, err := ExtractTarGz(archive, t.TempDir(), wantLib, cap); err == nil {
			t.Errorf("symlink to %q was accepted", link)
		}
	}
}

func TestExtractTarGz_NothingWantedAndCap(t *testing.T) {
	archive := tarGz(t, "x", []tarEntry{{name: "README", body: "no libs here"}})
	if n, err := ExtractTarGz(archive, t.TempDir(), wantLib, cap); err != nil || n != 0 {
		t.Errorf("got %d, %v; want 0 entries and no error", n, err)
	}
	archive = tarGz(t, "x", []tarEntry{{name: "lib/libbig.so", body: strings.Repeat("b", 100)}})
	if _, err := ExtractTarGz(archive, t.TempDir(), wantLib, 10); err == nil || !strings.Contains(err.Error(), "cap") {
		t.Errorf("err = %v, want the cap error", err)
	}
	if _, err := ExtractTarGz(filepath.Join(t.TempDir(), "missing.tgz"), t.TempDir(), wantLib, cap); err == nil {
		t.Error("missing archive: want an error")
	}
	notGz := filepath.Join(t.TempDir(), "x.tgz")
	_ = os.WriteFile(notGz, []byte("plain"), 0o600)
	if _, err := ExtractTarGz(notGz, t.TempDir(), wantLib, cap); err == nil {
		t.Error("not a gzip: want an error")
	}
}

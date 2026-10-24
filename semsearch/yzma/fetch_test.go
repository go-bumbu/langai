package yzma

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// serve returns an httptest server that answers every path with body, and
// the asset describing it (digest and size filled in).
func serve(t *testing.T, body []byte) (*httptest.Server, asset) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	sum := sha256.Sum256(body)
	return srv, asset{URL: srv.URL + "/file", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(body))}
}

// discardLogger is the logger for tests whose log lines do not matter;
// TestNew_LoadsThroughTheAPI checks the lines themselves.
func discardLogger() *slog.Logger { return slog.New(slog.DiscardHandler) }

func TestFetchFile_WritesVerifiedFile(t *testing.T) {
	_, a := serve(t, []byte("hello"))
	dest := filepath.Join(t.TempDir(), "out.bin")
	if err := fetchFile(context.Background(), "test file", a, dest, discardLogger()); err != nil {
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

// TestFetchFile_RejectsDigestMismatch: a corrupted or replaced download must
// never appear under its final name, and must not leave temp files around.
func TestFetchFile_RejectsDigestMismatch(t *testing.T) {
	_, a := serve(t, []byte("hello"))
	a.SHA256 = strings.Repeat("0", 64)
	dest := filepath.Join(t.TempDir(), "out.bin")
	err := fetchFile(context.Background(), "test file", a, dest, discardLogger())
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

func TestFetchFile_RejectsWrongSize(t *testing.T) {
	_, a := serve(t, []byte("hello"))
	a.Size = 4
	err := fetchFile(context.Background(), "test file", a, filepath.Join(t.TempDir(), "out.bin"), discardLogger())
	if err == nil || !strings.Contains(err.Error(), "bytes") {
		t.Fatalf("err = %v, want a size error", err)
	}
}

func TestFetchFile_ReportsHTTPErrors(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(srv.Close)
	a := asset{URL: srv.URL + "/missing", SHA256: strings.Repeat("0", 64)}
	err := fetchFile(context.Background(), "test file", a, filepath.Join(t.TempDir(), "out.bin"), discardLogger())
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("err = %v, want the HTTP status", err)
	}
}

type tarEntry struct {
	name, link string // link set = symlink to link
	body       string
}

// tarGz builds a gzipped tarball in memory with entries nested under dir/,
// like a llama.cpp release archive.
func tarGz(t *testing.T, dir string, entries []tarEntry) []byte {
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

// releaseLibs mirrors the shape of the pinned ggml-org archive: versioned
// library files, symlink chains, CPU variants, and the tool/impl libraries
// the install must leave behind.
func releaseLibs() []tarEntry {
	return []tarEntry{
		{name: "libllama.so.0.6.0", body: "llama"},
		{name: "libllama.so.0", link: "libllama.so.0.6.0"},
		{name: "libllama.so", link: "libllama.so.0"},
		{name: "libggml-base.so.0.26.0", body: "base"},
		{name: "libggml-base.so.0", link: "libggml-base.so.0.26.0"},
		{name: "libggml-base.so", link: "libggml-base.so.0"},
		{name: "libggml.so.0.26.0", body: "ggml"},
		{name: "libggml.so.0", link: "libggml.so.0.26.0"},
		{name: "libggml.so", link: "libggml.so.0"},
		{name: "libggml-cpu-x64.so", body: "cpu"},
		{name: "libggml-cpu-zen4.so", body: "cpu-zen4"},
		{name: "libggml-rpc.so", body: "rpc"},         // skipped: RPC backend
		{name: "libmtmd.so.0.6.0", body: "mtmd"},      // skipped: multimodal
		{name: "libllama-common.so.0.6.0", body: "c"}, // skipped: tools
		{name: "libllama-cli-impl.so", body: "cli"},   // skipped: tools
		{name: "llama-cli", body: "#!elf"},            // skipped: executable
	}
}

func writeArchive(t *testing.T, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "libs.tar.gz")
	if err := os.WriteFile(p, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestExtractLibs_FlattensFiltersAndRecreatesSymlinks(t *testing.T) {
	archive := writeArchive(t, tarGz(t, "llama-b11429", releaseLibs()))
	dir := t.TempDir()
	if err := extractLibs(archive, dir); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	want := []string{
		"libggml-base.so", "libggml-base.so.0", "libggml-base.so.0.26.0",
		"libggml-cpu-x64.so", "libggml-cpu-zen4.so",
		"libggml.so", "libggml.so.0", "libggml.so.0.26.0",
		"libllama.so", "libllama.so.0", "libllama.so.0.6.0",
	}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("extracted %v\nwant      %v", names, want)
	}
	if target, err := os.Readlink(filepath.Join(dir, "libllama.so")); err != nil || target != "libllama.so.0" {
		t.Fatalf("libllama.so -> %q, %v; want a symlink to libllama.so.0", target, err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "libllama.so")); string(got) != "llama" {
		t.Fatalf("following the symlink chain read %q, want the real library", got)
	}
}

// TestExtractLibs_RejectsEscapingSymlinks: a symlink may only point at a bare
// library name in the same directory; anything else is an attack surface, not
// a packaging quirk.
func TestExtractLibs_RejectsEscapingSymlinks(t *testing.T) {
	for _, link := range []string{"../../etc/ld.so.preload", "/usr/lib/libllama.so", "libmtmd.so"} {
		archive := writeArchive(t, tarGz(t, "x", []tarEntry{{name: "libllama.so", link: link}}))
		if err := extractLibs(archive, t.TempDir()); err == nil {
			t.Errorf("symlink to %q was accepted", link)
		}
	}
}

func TestExtractLibs_RejectsArchiveWithoutLibraries(t *testing.T) {
	archive := writeArchive(t, tarGz(t, "x", []tarEntry{{name: "README", body: "no libs here"}}))
	if err := extractLibs(archive, t.TempDir()); err == nil {
		t.Fatal("an archive without llama.cpp libraries must be an error")
	}
}

func TestWantLib(t *testing.T) {
	yes := []string{"libllama.so", "libllama.so.0", "libllama.so.0.6.0", "libggml.so", "libggml-base.so.0.26.0", "libggml-cpu-x64.so", "libggml-cpu-armv8.2_1.so", "libggml-vulkan.so"}
	no := []string{"libggml-rpc.so", "libmtmd.so", "libllama-common.so.0", "libllama-cli-impl.so", "llama-cli", "README.md", "libllama.dylib"}
	for _, n := range yes {
		if !wantLib(n) {
			t.Errorf("wantLib(%q) = false, want true", n)
		}
	}
	for _, n := range no {
		if wantLib(n) {
			t.Errorf("wantLib(%q) = true, want false", n)
		}
	}
}

func TestEnsureLibsFor_ManagedDownloadsAndInstalls(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the pinned archives hold Linux .so files")
	}
	_, a := serve(t, tarGz(t, "llama-b11429", releaseLibs()))
	libAssets["test/arch"] = a
	t.Cleanup(func() { delete(libAssets, "test/arch") })

	libDir := filepath.Join(t.TempDir(), "llama-cpp", LibTag)
	// a directory left behind by an interrupted install (no libllama.so) is replaced
	if err := os.MkdirAll(libDir, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(libDir, "stale"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, true, discardLogger()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(libDir, "libllama.so")); err != nil {
		t.Fatalf("libllama.so missing after install: %v", err)
	}
	if _, err := os.Stat(filepath.Join(libDir, "stale")); err == nil {
		t.Fatal("the interrupted install's leftovers survived")
	}
	entries, _ := os.ReadDir(filepath.Dir(libDir))
	if len(entries) != 1 {
		t.Fatalf("staging directory left behind: %v", entries)
	}
	// installed libraries are reused: the second call must not need the network
	libAssets["test/arch"] = asset{URL: "http://127.0.0.1:1/unreachable", SHA256: a.SHA256}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, true, discardLogger()); err != nil {
		t.Fatalf("installed libraries must be reused, got %v", err)
	}
}

func TestEnsureLibsFor_UnsupportedPlatform(t *testing.T) {
	err := ensureLibsFor(context.Background(), "plan9/mips", filepath.Join(t.TempDir(), "libs"), true, discardLogger())
	if err == nil || !strings.Contains(err.Error(), "plan9/mips") || !strings.Contains(err.Error(), "yzma install") {
		t.Fatalf("err = %v, want an error naming the platform and the manual install path", err)
	}
}

func TestEnsureLibsFor_OperatorDirWithoutLibraries(t *testing.T) {
	dir := t.TempDir()
	err := ensureLibsFor(context.Background(), "linux/amd64", dir, false, discardLogger())
	if err == nil || !strings.Contains(err.Error(), "yzma install") {
		t.Fatalf("err = %v, want an error telling the operator how to install", err)
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Fatalf("the install wrote into an operator directory: %v", entries)
	}
}

// TestEnsureLibsFor_OperatorDirChecksInstallRecord: yzma's Go structs mirror
// one llama.cpp ABI, so a directory installed with `yzma install` for another
// build is refused rather than loaded.
func TestEnsureLibsFor_OperatorDirChecksInstallRecord(t *testing.T) {
	cases := []struct {
		name   string
		record string // "" = no record file (manual install)
		ok     bool
	}{
		{"no record", "", true},
		{"tagged release install", `{"tag":"v0.6.0","upstream_tag":"b11429"}`, true},
		{"nightly install", `{"tag":"b11429"}`, true},
		{"other build", `{"tag":"v0.7.0","upstream_tag":"b11900"}`, false},
		{"garbage record", `{not json`, false},
	}
	for _, c := range cases {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, libFile()), []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
		if c.record != "" {
			if err := os.WriteFile(filepath.Join(dir, installRecordFile), []byte(c.record), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		err := ensureLibsFor(context.Background(), "linux/amd64", dir, false, discardLogger())
		if (err == nil) != c.ok {
			t.Errorf("%s: err = %v, want ok=%v", c.name, err, c.ok)
		}
		if err != nil && !c.ok && c.name == "other build" && !strings.Contains(err.Error(), LibTag) {
			t.Errorf("%s: error should name the pinned build %s: %v", c.name, LibTag, err)
		}
	}
}

func TestEnsureModelFile_DownloadsOnceIntoModelsDir(t *testing.T) {
	_, a := serve(t, []byte("GGUF-bytes"))
	orig := defaultModelAsset
	defaultModelAsset = a
	t.Cleanup(func() { defaultModelAsset = orig })

	dir := filepath.Join(t.TempDir(), "models")
	if err := ensureModelFile(context.Background(), DefaultSpec(dir), discardLogger()); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, defaultModelFile)
	if got, _ := os.ReadFile(path); string(got) != "GGUF-bytes" {
		t.Fatalf("model file %s = %q", path, got)
	}
	// a present model is reused: the second call must not need the network
	defaultModelAsset = asset{URL: "http://127.0.0.1:1/unreachable", SHA256: a.SHA256}
	if err := ensureModelFile(context.Background(), DefaultSpec(dir), discardLogger()); err != nil {
		t.Fatalf("a present model must be reused without downloading, got %v", err)
	}
}

// TestPinnedAssetsAreWellFormed catches a typo in a pin before it costs a
// download: https URLs, 64 hex digits, a known size.
func TestPinnedAssetsAreWellFormed(t *testing.T) {
	check := func(name string, a asset) {
		if !strings.HasPrefix(a.URL, "https://") {
			t.Errorf("%s: URL %q is not https", name, a.URL)
		}
		if len(a.SHA256) != 64 || strings.Trim(a.SHA256, "0123456789abcdef") != "" {
			t.Errorf("%s: SHA256 %q is not 64 lowercase hex digits", name, a.SHA256)
		}
		if a.Size <= 0 {
			t.Errorf("%s: size must be known", name)
		}
	}
	check("model", defaultModelAsset)
	for plat, a := range libAssets {
		check(plat, a)
	}
	if _, ok := libAssets["linux/amd64"]; !ok {
		t.Error("linux/amd64 (the release target) has no pinned llama.cpp build")
	}
}

// TestDownload_ReportsDigestAndSize: an unpinned download lands under its
// final name and reports what to pin, with no temp file left behind.
func TestDownload_ReportsDigestAndSize(t *testing.T) {
	_, a := serve(t, []byte("hello"))
	dest := filepath.Join(t.TempDir(), "models", "out.gguf")
	sum, n, err := Download(context.Background(), a.URL, dest, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sum != a.SHA256 || n != a.Size {
		t.Fatalf("Download = %s, %d; want %s, %d", sum, n, a.SHA256, a.Size)
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

func TestDownload_ReportsHTTPErrors(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(srv.Close)
	dest := filepath.Join(t.TempDir(), "out.gguf")
	_, _, err := Download(context.Background(), srv.URL+"/missing", dest, nil)
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("err = %v, want the HTTP status", err)
	}
	if _, statErr := os.Stat(dest); statErr == nil {
		t.Fatal("a failed download appeared under its final name")
	}
}

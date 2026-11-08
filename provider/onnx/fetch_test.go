package onnx

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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

func discardLogger() *slog.Logger { return slog.New(slog.DiscardHandler) }

// writeInstallRecord records version and variant in dir, as an install does.
func writeInstallRecord(dir, variant string) error {
	data, err := json.Marshal(installRecord{Version: LibVersion, Variant: variant})
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, installRecordFile), data, 0o600)
}

type tarEntry struct {
	name, link string // link set = symlink to link
	body       string
}

// tarGz builds a gzipped tarball in memory with entries nested under dir/,
// like a release archive.
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

// releaseLibs mirrors the shape of the release tarballs: the versioned
// library with its symlink chain, the providers' shared library, and the
// headers, pkgconfig, cmake and (on macOS) debug symbol files the install
// must leave behind. The names follow this platform's so the archive
// installs here.
func releaseLibs() []tarEntry {
	return []tarEntry{
		{name: "lib/" + libFile(), body: "ort"},
		{name: "lib/" + plainLibFile() + ".1", link: libFile()},
		{name: "lib/" + plainLibFile(), link: plainLibFile() + ".1"},
		{name: "lib/libonnxruntime_providers_shared.so", body: "providers"},
		{name: "lib/" + libFile() + ".dSYM/Contents/Resources/DWARF/" + libFile(), body: "debug symbols"},
		{name: "lib/pkgconfig/libonnxruntime.pc", body: "pc"},
		{name: "lib/cmake/onnxruntime/onnxruntimeConfig.cmake", body: "cmake"},
		{name: "include/onnxruntime_c_api.h", body: "h"},
		{name: "VERSION_NUMBER", body: LibVersion},
	}
}

func writeArchive(t *testing.T, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "libs.tgz")
	if err := os.WriteFile(p, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestWantLib(t *testing.T) {
	yes := []string{"onnxruntime-linux-x64-1.31.1/lib/libonnxruntime.so.1.31.1", "x/lib/libonnxruntime.so", "x/lib/libonnxruntime.so.1",
		"x/lib/libonnxruntime_providers_shared.so", "x/lib/libonnxruntime_providers_cuda.so", "x/lib/libonnxruntime.1.31.1.dylib", "lib/libonnxruntime.dylib"}
	no := []string{"x/lib/pkgconfig/libonnxruntime.pc", "x/include/onnxruntime_c_api.h", "x/lib/cmake/onnxruntime/onnxruntimeConfig.cmake",
		"x/lib/libonnxruntime.1.31.1.dylib.dSYM/Contents/Resources/DWARF/libonnxruntime.1.31.1.dylib", "x/libonnxruntime.so", "x/lib/libfoo.so", "x/VERSION_NUMBER"}
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

func TestExtractLibs_KeepsTheLibrariesOnly(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks")
	}
	archive := writeArchive(t, tarGz(t, "onnxruntime-x-"+LibVersion, releaseLibs()))
	dir := t.TempDir()
	if err := extractLibs(archive, dir); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(dir)
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	if len(names) != 4 {
		t.Fatalf("extracted %v, want the library, its two symlinks and the providers' library", names)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, plainLibFile())); string(got) != "ort" {
		t.Fatalf("following the symlink chain read %q, want the real library (not the debug symbols)", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "libonnxruntime.pc")); err == nil {
		t.Error("pkgconfig file extracted")
	}
	// an archive without the versioned library is no install
	archive = writeArchive(t, tarGz(t, "x", []tarEntry{{name: "lib/libonnxruntime_providers_shared.so", body: "p"}}))
	if err := extractLibs(archive, t.TempDir()); err == nil || !strings.Contains(err.Error(), libFile()) {
		t.Errorf("archive without the library: err = %v", err)
	}
}

func TestLibraryPath_PrefersTheVersionedFile(t *testing.T) {
	dir := t.TempDir()
	if _, err := libraryPath(dir); err == nil {
		t.Error("empty dir: want an error")
	}
	plain := filepath.Join(dir, plainLibFile())
	if err := os.WriteFile(plain, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if p, err := libraryPath(dir); err != nil || p != plain {
		t.Errorf("plain only: %q, %v", p, err)
	}
	versioned := filepath.Join(dir, libFile())
	if err := os.WriteFile(versioned, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if p, err := libraryPath(dir); err != nil || p != versioned {
		t.Errorf("both: %q, %v; want the versioned one", p, err)
	}
}

func TestEnsureLibsFor_ManagedInstallsAndRefillsOnAnotherVariant(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks")
	}
	_, cpu := serve(t, tarGz(t, "ort-cpu", releaseLibs()))
	_, cuda := serve(t, tarGz(t, "ort-cuda", append(releaseLibs(), tarEntry{name: "lib/libonnxruntime_providers_cuda.so", body: "cuda"})))
	libAssets["test/arch/cpu"] = cpu
	libAssets["test/arch/cuda"] = cuda
	t.Cleanup(func() { delete(libAssets, "test/arch/cpu"); delete(libAssets, "test/arch/cuda") })

	libDir := filepath.Join(t.TempDir(), "onnxruntime", LibVersion)
	// a directory left behind by an interrupted install (no library) is replaced
	if err := os.MkdirAll(libDir, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(libDir, "stale"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, true, variantCPU, discardLogger()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(libDir, libFile())); err != nil {
		t.Fatalf("library missing after install: %v", err)
	}
	if _, err := os.Stat(filepath.Join(libDir, "stale")); err == nil {
		t.Fatal("the interrupted install's leftovers survived")
	}
	rec, err := readInstallRecord(libDir)
	if err != nil || rec == nil || rec.Version != LibVersion || rec.Variant != variantCPU {
		t.Fatalf("record = %+v, %v", rec, err)
	}
	if entries, _ := os.ReadDir(filepath.Dir(libDir)); len(entries) != 1 {
		t.Fatalf("staging directory left behind: %v", entries)
	}
	// installed libraries are reused: the second call must not need the network
	libAssets["test/arch/cpu"] = asset{URL: "http://127.0.0.1:1/unreachable", SHA256: cpu.SHA256}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, true, variantCPU, discardLogger()); err != nil {
		t.Fatalf("installed libraries must be reused, got %v", err)
	}
	// another variant refills the directory
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, true, variantCUDA, discardLogger()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(libDir, "libonnxruntime_providers_cuda.so")); err != nil {
		t.Errorf("the CUDA build was not installed: %v", err)
	}
	if rec, _ := readInstallRecord(libDir); rec == nil || rec.Variant != variantCUDA {
		t.Errorf("record after refill = %+v", rec)
	}
	// so does a directory without a record (not installed by this package)
	if err := os.Remove(filepath.Join(libDir, installRecordFile)); err != nil {
		t.Fatal(err)
	}
	libAssets["test/arch/cuda"] = asset{URL: "http://127.0.0.1:1/unreachable", SHA256: cuda.SHA256}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, true, variantCUDA, discardLogger()); err == nil {
		t.Error("a managed directory without a record must be reinstalled (here: fail to download)")
	}
}

func TestEnsureLibsFor_OperatorDirectory(t *testing.T) {
	libDir := t.TempDir()
	err := ensureLibsFor(context.Background(), "test/arch", libDir, false, variantCPU, discardLogger())
	if err == nil || !strings.Contains(err.Error(), libFile()) || !strings.Contains(err.Error(), "managed") {
		t.Fatalf("empty operator dir: err = %v", err)
	}
	if err := os.WriteFile(filepath.Join(libDir, libFile()), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, false, variantCPU, discardLogger()); err != nil {
		t.Fatalf("operator dir with the library and no record: %v", err)
	}
	if err := writeInstallRecord(libDir, variantCUDA); err != nil {
		t.Fatal(err)
	}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, false, variantCUDA, discardLogger()); err != nil {
		t.Fatalf("operator dir with a matching record: %v", err)
	}
	err = ensureLibsFor(context.Background(), "test/arch", libDir, false, variantCPU, discardLogger())
	if err == nil || !strings.Contains(err.Error(), variantCUDA) {
		t.Fatalf("operator dir with another variant: err = %v", err)
	}
	if err := os.WriteFile(filepath.Join(libDir, installRecordFile), []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, false, variantCPU, discardLogger()); err == nil {
		t.Error("a corrupt record must be an error")
	}
}

func TestEnsureLibsFor_UnsupportedPlatform(t *testing.T) {
	err := ensureLibsFor(context.Background(), "plan9/mips", filepath.Join(t.TempDir(), "libs"), true, variantCPU, discardLogger())
	if err == nil || !strings.Contains(err.Error(), "plan9/mips") {
		t.Fatalf("err = %v, want an error naming the platform", err)
	}
	err = ensureLibsFor(context.Background(), "darwin/arm64", filepath.Join(t.TempDir(), "libs"), true, variantCUDA, discardLogger())
	if err == nil || !strings.Contains(err.Error(), "cuda") {
		t.Fatalf("err = %v, want an error naming the variant", err)
	}
}

func TestEnsureLibsFor_DownloadFailureLeavesNothing(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(srv.Close)
	libAssets["test/arch/cpu"] = asset{URL: srv.URL + "/missing", SHA256: strings.Repeat("0", 64)}
	t.Cleanup(func() { delete(libAssets, "test/arch/cpu") })
	libDir := filepath.Join(t.TempDir(), "libs")
	err := ensureLibsFor(context.Background(), "test/arch", libDir, true, variantCPU, discardLogger())
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("err = %v, want the HTTP status", err)
	}
	if entries, _ := os.ReadDir(filepath.Dir(libDir)); len(entries) != 0 {
		t.Fatalf("something left behind: %v", entries)
	}
}

func TestEnsureModelFile(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	present := filepath.Join(dir, "present.onnx")
	if err := os.WriteFile(present, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ensureModelFile(ctx, ModelSpec{Name: "m", File: present}, discardLogger()); err != nil {
		t.Errorf("present file: %v", err)
	}
	err := ensureModelFile(ctx, ModelSpec{Name: "m", File: filepath.Join(dir, "absent.onnx")}, discardLogger())
	if err == nil || !strings.Contains(err.Error(), "no URL") {
		t.Errorf("absent without URL: err = %v", err)
	}
	_, a := serve(t, []byte("downloaded model"))
	dest := filepath.Join(dir, "sub", "model.onnx")
	spec := ModelSpec{Name: "m", File: dest, URL: a.URL, SHA256: a.SHA256, Size: a.Size}
	if err := ensureModelFile(ctx, spec, discardLogger()); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); string(got) != "downloaded model" {
		t.Errorf("downloaded %q", got)
	}
	spec.SHA256 = strings.Repeat("1", 64)
	spec.File = filepath.Join(dir, "bad.onnx")
	if err := ensureModelFile(ctx, spec, discardLogger()); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Errorf("bad digest: err = %v", err)
	}
}

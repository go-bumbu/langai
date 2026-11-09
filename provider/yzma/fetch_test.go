package yzma

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
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
		{name: "libmtmd.so.0.6.0", body: "mtmd"},      // kept: the Vision embedder
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
		"libmtmd.so.0.6.0",
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
	for _, link := range []string{"../../etc/ld.so.preload", "/usr/lib/libllama.so", "libllama-cli-impl.so"} {
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
	yes := []string{"libllama.so", "libllama.so.0", "libllama.so.0.6.0", "libggml.so", "libggml-base.so.0.26.0", "libggml-cpu-x64.so", "libggml-cpu-armv8.2_1.so", "libggml-vulkan.so", "libmtmd.so", "libmtmd.so.0.6.0", "libllama.dylib", "libggml-metal.dylib", "libmtmd.0.dylib"}
	no := []string{"libggml-rpc.so", "libllama-common.so.0", "libllama-cli-impl.so", "libllama-server-impl.so", "llama-cli", "llama-mtmd-cli", "README.md"}
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

// ensureLibsFor asks ensureLibs for build b in libDir on plat, in a process
// that has loaded nothing and loads nothing.
func ensureLibsFor(ctx context.Context, plat, libDir string, managed bool, b build, logger *slog.Logger) error {
	return ensureLibs(ctx, libsWant{plat: plat, dir: libDir, managed: managed, build: b, explicit: true}, logger)
}

func TestEnsureLibsFor_ManagedDownloadsAndInstalls(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the pinned archives hold Linux .so files")
	}
	_, a := serve(t, tarGz(t, "llama-b11429", releaseLibs()))
	libAssets["test/arch/cpu"] = a
	t.Cleanup(func() { delete(libAssets, "test/arch/cpu") })

	libDir := filepath.Join(t.TempDir(), "llama-cpp", LibTag)
	// a directory of this package's that lost its libraries (it has the
	// record, no libllama.so) is replaced
	if err := os.MkdirAll(libDir, 0o750); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"stale", libRecordFile} {
		if err := os.WriteFile(filepath.Join(libDir, name), []byte("{}"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, true, "cpu", discardLogger()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(libDir, "libllama.so")); err != nil {
		t.Fatalf("libllama.so missing after install: %v", err)
	}
	if _, err := os.Stat(filepath.Join(libDir, "stale")); err == nil {
		t.Fatal("the broken install's leftovers survived")
	}
	// the directory and its lock file: no archive, no staging
	entries, _ := os.ReadDir(filepath.Dir(libDir))
	if len(entries) != 2 {
		t.Fatalf("staging left behind: %v", entries)
	}
	// installed libraries are reused: the second call must not need the network
	libAssets["test/arch/cpu"] = asset{URL: "http://127.0.0.1:1/unreachable", SHA256: a.SHA256}
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, true, "cpu", discardLogger()); err != nil {
		t.Fatalf("installed libraries must be reused, got %v", err)
	}
}

// TestNeedsReinstall: a managed directory serves a request as it is when
// its record names the pinned llama.cpp build and a backend that covers the
// one wanted (a GPU build runs the CPU too). A record of another tag is the
// install an older version of this package left: its libraries would fail
// the version check at every load, so it is replaced whatever the backend.
func TestNeedsReinstall(t *testing.T) {
	for _, tc := range []struct {
		name   string
		record string // the record file's content, "" for none
		build  build
		want   bool
	}{
		{"no record, cpu wanted", "", "cpu", false},
		{"no record, a GPU build wanted", "", "vulkan", true},
		{"the pinned cpu build, cpu wanted", `{"tag":"` + LibTag + `","backend":"cpu"}`, "cpu", false},
		{"the pinned cpu build, vulkan wanted", `{"tag":"` + LibTag + `","backend":"cpu"}`, "vulkan", true},
		{"the pinned vulkan build, cpu wanted", `{"tag":"` + LibTag + `","backend":"vulkan"}`, "cpu", false},
		{"the pinned vulkan build, vulkan wanted", `{"tag":"` + LibTag + `","backend":"vulkan"}`, "vulkan", false},
		{"the pinned vulkan build, cuda wanted", `{"tag":"` + LibTag + `","backend":"vulkan"}`, "cuda", true},
		{"an older tag's cpu build, cpu wanted", `{"tag":"b0001","backend":"cpu"}`, "cpu", true},
		{"an older tag's vulkan build, cpu wanted", `{"tag":"b0001","backend":"vulkan"}`, "cpu", true},
		{"an older tag's vulkan build, vulkan wanted", `{"tag":"b0001","backend":"vulkan"}`, "vulkan", true},
		{"a record without a tag", `{"backend":"cpu"}`, "cpu", true},
		{"a record that does not parse", `{"tag":`, "cpu", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.record != "" {
				if err := os.WriteFile(filepath.Join(dir, libRecordFile), []byte(tc.record), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if got := needsReinstall(dir, tc.build); got != tc.want {
				t.Errorf("needsReinstall = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestEnsureLibsFor_ReplacesAnOlderTagsInstall: a managed directory an
// older pinned build was installed into has the libraries and a record, and
// is not ready: the pinned build is installed over it, for a CPU request
// too, instead of loading libraries that then fail the version check for as
// long as the process lives.
func TestEnsureLibsFor_ReplacesAnOlderTagsInstall(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the pinned archives hold Linux .so files")
	}
	_, a := serve(t, tarGz(t, "llama-"+LibTag, releaseLibs()))
	libAssets["test/arch/cpu"] = a
	t.Cleanup(func() { delete(libAssets, "test/arch/cpu") })
	libDir := filepath.Join(t.TempDir(), "llama-cpp", "current")
	if err := os.MkdirAll(libDir, 0o750); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		libFile():             "the old build",
		"libggml-old-only.so": "gone in the new build",
		libRecordFile:         `{"tag":"b0001","backend":"cpu"}`,
	} {
		if err := os.WriteFile(filepath.Join(libDir, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	var logs bytes.Buffer
	if err := ensureLibsFor(context.Background(), "test/arch", libDir, true, "cpu", slog.New(slog.NewTextHandler(&logs, nil))); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(filepath.Join(libDir, libFile())); string(got) != "llama" {
		t.Errorf("%s holds %q, want the pinned build's", libFile(), got)
	}
	if _, err := os.Stat(filepath.Join(libDir, "libggml-old-only.so")); err == nil {
		t.Error("a library of the old build survived")
	}
	var rec libRecord
	data, _ := os.ReadFile(filepath.Join(libDir, libRecordFile))
	if err := json.Unmarshal(data, &rec); err != nil || rec.Tag != LibTag || rec.Backend != "cpu" {
		t.Errorf("record = %+v, %v; want the pinned tag", rec, err)
	}
	if !strings.Contains(logs.String(), "b0001") {
		t.Errorf("the log does not say what was replaced:\n%s", logs.String())
	}
}

// TestEnsureLibsFor_ManagedNeverWipesForeignFiles: ManagedLibDir on a
// directory that holds something else (LibDir "/opt/app") must not delete
// it. Without the libraries nothing can be installed there, and with
// libraries of unknown origin they are used for the CPU and not replaced by
// a GPU build.
func TestEnsureLibsFor_ManagedNeverWipesForeignFiles(t *testing.T) {
	_, a := serve(t, tarGz(t, "llama-b11429", releaseLibs()))
	libAssets["test/arch/cpu"], libAssets["test/arch/vulkan"] = a, a
	t.Cleanup(func() { delete(libAssets, "test/arch/cpu"); delete(libAssets, "test/arch/vulkan") })
	app := t.TempDir()
	precious := filepath.Join(app, "app.conf")
	if err := os.WriteFile(precious, []byte("precious"), 0o600); err != nil {
		t.Fatal(err)
	}
	err := ensureLibsFor(context.Background(), "test/arch", app, true, "cpu", discardLogger())
	if err == nil || !strings.Contains(err.Error(), "refusing to replace") || !strings.Contains(err.Error(), libRecordFile) {
		t.Fatalf("err = %v, want a refusal naming the record file", err)
	}
	if err := os.WriteFile(filepath.Join(app, libFile()), []byte("someone's build"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ensureLibsFor(context.Background(), "test/arch", app, true, "cpu", discardLogger()); err != nil {
		t.Errorf("libraries that are there serve the CPU as they are: %v", err)
	}
	if err := ensureLibsFor(context.Background(), "test/arch", app, true, "vulkan", discardLogger()); err == nil || !strings.Contains(err.Error(), "refusing to replace") {
		t.Errorf("a GPU build over foreign libraries: err = %v", err)
	}
	if got, _ := os.ReadFile(precious); string(got) != "precious" {
		t.Fatal("the directory's own file is gone")
	}
}

// TestEnsureLibs_NeverReplacesLoadedLibraries: once the process runs the
// libraries of a directory they stay. A build picked from the hardware
// gives way with a warning; one asked for by name is an error.
func TestEnsureLibs_NeverReplacesLoadedLibraries(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the pinned archives hold Linux .so files")
	}
	_, cpu := serve(t, tarGz(t, "llama-b11429", releaseLibs()))
	libAssets["test/arch/cpu"] = cpu
	libAssets["test/arch/vulkan"] = asset{URL: "http://127.0.0.1:1/unreachable", SHA256: cpu.SHA256}
	t.Cleanup(func() { delete(libAssets, "test/arch/cpu"); delete(libAssets, "test/arch/vulkan") })
	libDir := filepath.Join(t.TempDir(), "llama-cpp", LibTag)
	loader := &runtimeLoader{}
	f := newFakeAPI()
	want := libsWant{plat: "test/arch", dir: libDir, managed: true, build: "cpu", explicit: true, loadedDir: loader.loadedDir,
		use: func() error { return loader.load(f, libDir, nil) }}
	if err := ensureLibs(context.Background(), want, discardLogger()); err != nil {
		t.Fatal(err)
	}
	if loader.loadedDir() != libDir || f.count("load") != 1 {
		t.Fatalf("loaded from %q, %d loads", loader.loadedDir(), f.count("load"))
	}
	want.build = "vulkan"
	err := ensureLibs(context.Background(), want, discardLogger())
	if err == nil || !strings.Contains(err.Error(), "already loaded") || !strings.Contains(err.Error(), "vulkan") {
		t.Errorf("explicit vulkan over loaded cpu libraries: err = %v", err)
	}
	var logs bytes.Buffer
	want.explicit = false
	if err := ensureLibs(context.Background(), want, slog.New(slog.NewTextHandler(&logs, nil))); err != nil {
		t.Errorf("a build picked from the hardware must give way: %v", err)
	}
	if !strings.Contains(logs.String(), "level=WARN") || !strings.Contains(logs.String(), "already loaded") {
		t.Errorf("no warning that the loaded build stays:\n%s", logs.String())
	}
	if f.count("load") != 1 {
		t.Errorf("the libraries were loaded %d times", f.count("load"))
	}
}

// TestEnsureLibs_ConcurrentOpensInstallOnceAndLoadAfter: two goroutines on a
// fresh managed directory take turns; one downloads, and neither loads
// while the other installs.
func TestEnsureLibs_ConcurrentOpensInstallOnceAndLoadAfter(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the pinned archives hold Linux .so files")
	}
	var requests atomic.Int32
	body := tarGz(t, "llama-b11429", releaseLibs())
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	sum := sha256.Sum256(body)
	libAssets["test/arch/cpu"] = asset{URL: srv.URL + "/libs.tgz", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(body))}
	t.Cleanup(func() { delete(libAssets, "test/arch/cpu") })
	libDir := filepath.Join(t.TempDir(), "llama-cpp", LibTag)
	var wg sync.WaitGroup
	errs := make([]error, 6)
	for i := range errs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = ensureLibs(context.Background(), libsWant{plat: "test/arch", dir: libDir, managed: true, build: "cpu",
				use: func() error {
					// what a load needs: the library, whole
					got, err := os.ReadFile(filepath.Join(libDir, "libllama.so"))
					if err == nil && string(got) != "llama" {
						err = errors.New("a half-installed library")
					}
					return err
				}}, discardLogger())
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("open %d: %v", i, err)
		}
	}
	if n := requests.Load(); n != 1 {
		t.Errorf("the build was downloaded %d times", n)
	}
}

func TestEnsureLibsFor_UnsupportedPlatform(t *testing.T) {
	err := ensureLibsFor(context.Background(), "plan9/mips", filepath.Join(t.TempDir(), "libs"), true, "cpu", discardLogger())
	if err == nil || !strings.Contains(err.Error(), "plan9/mips") || !strings.Contains(err.Error(), "yzma install") {
		t.Fatalf("err = %v, want an error naming the platform and the manual install path", err)
	}
}

func TestEnsureLibsFor_OperatorDirWithoutLibraries(t *testing.T) {
	dir := t.TempDir()
	err := ensureLibsFor(context.Background(), "linux/amd64", dir, false, "cpu", discardLogger())
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
		err := ensureLibsFor(context.Background(), "linux/amd64", dir, false, "cpu", discardLogger())
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
	if _, ok := libAssets["linux/amd64/cpu"]; !ok {
		t.Error("linux/amd64/cpu (the release target) has no pinned llama.cpp build")
	}
}

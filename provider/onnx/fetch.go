package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/go-bumbu/langai/internal/fetch"
)

// Pinned runtime. The onnxruntime_go bindings are written against one
// release's C API headers, so the build downloaded here is bumped together
// with the binding version in go.mod; pins_test.go fails until pinnedBinding
// matches the module version again. The digests are the ones GitHub
// publishes for the release assets.
const (
	// pinnedBinding is the onnxruntime_go release whose headers LibVersion
	// matches.
	pinnedBinding = "v1.38.0"
	// LibVersion is the ONNX Runtime release a managed LibDir downloads;
	// name a managed directory after it so a bump never loads stale native
	// code. runtimeLoader.load refuses a library of another minor version.
	LibVersion = "1.31.1"
	// libMajorMinor is what the loaded library's version must start with.
	libMajorMinor = "1.31"
	// installRecordFile records which build a managed directory holds, so a
	// change of Provider refills it; an operator directory with one is
	// checked against it.
	installRecordFile = "onnxruntime-install.json"
	// maxDownloadBytes caps any single download; a download pinned to a
	// larger exact size (the CUDA build) is capped at that size instead.
	maxDownloadBytes = 1 << 30
	// maxLibraryBytes caps one extracted library; the CUDA provider's is
	// the largest by far.
	maxLibraryBytes = 2 << 30
	// downloadTimeout bounds the first-use downloads as a whole: the model
	// is ~370 MB and the CUDA build ~450 MB.
	downloadTimeout = 30 * time.Minute
	// userAgent names this package's downloads.
	userAgent = "langai-onnx"
)

// Variants of the runtime build; the key of libAssets is platform/variant.
const (
	variantCPU  = "cpu"
	variantCUDA = "cuda"
)

// asset is one downloadable file with its expected digest.
type asset = fetch.Asset

// libAssets are the release builds of LibVersion per GOOS/GOARCH and
// variant. The CPU builds cover the three platforms the release ships as
// tarballs (the macOS one includes CoreML); the CUDA build is the CUDA 12
// one for linux/amd64. Other platforms and builds (Windows, CUDA 13,
// TensorRT, ROCm) go into an operator-supplied LibDir.
var libAssets = map[string]asset{
	"linux/amd64/" + variantCPU: {
		URL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz",
		SHA256: "c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9",
		Size:   11413655,
	},
	"linux/arm64/" + variantCPU: {
		URL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-aarch64-1.31.1.tgz",
		SHA256: "5317505e2da99be65f8e2775ba3c2afd6d2961fd78aba1bd8f56159f48820c78",
		Size:   10332381,
	},
	"darwin/arm64/" + variantCPU: {
		URL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-osx-arm64-1.31.1.tgz",
		SHA256: "2e1344a1926ee50eb25b38153cb6992c3567515aef0e677a1a520a1798896d6f",
		Size:   36936435,
	},
	"linux/amd64/" + variantCUDA: {
		URL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-gpu_cuda12-1.31.1.tgz",
		SHA256: "57cd51f677eca19a269cdba31b0c16c15d73a8feea41f170aa8b750447ab1f70",
		Size:   450233079,
	},
}

// httpClient is the client downloads go through. The default client honours
// HTTP(S)_PROXY; tests point assets at an httptest server instead.
var httpClient = http.DefaultClient

// platform is the first part of the libAssets key for this binary.
func platform() string { return runtime.GOOS + "/" + runtime.GOARCH }

// variantFor is the runtime build a provider needs: CUDA has its own, the
// others are in the default build of their platform.
func variantFor(p Provider) string {
	if p == ProviderCUDA {
		return variantCUDA
	}
	return variantCPU
}

// libFile is the versioned library file of LibVersion, the one the
// bindings are pointed at.
func libFile() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime." + LibVersion + ".dylib"
	case "windows":
		return "onnxruntime.dll"
	}
	return "libonnxruntime.so." + LibVersion
}

// plainLibFile is the unversioned name, what a manual install may be left
// with; the version check after load tells whether it is the right one.
func plainLibFile() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	case "windows":
		return "onnxruntime.dll"
	}
	return "libonnxruntime.so"
}

// libraryPath is the library to load from libDir: the versioned file, else
// the plain one.
func libraryPath(libDir string) (string, error) {
	for _, name := range []string{libFile(), plainLibFile()} {
		p := filepath.Join(libDir, name)
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no %s in %s", libFile(), libDir)
}

// installRecord is what a managed install writes next to the libraries.
type installRecord struct {
	Version string `json:"version"`
	Variant string `json:"variant"`
}

// readInstallRecord returns the directory's record, nil when it has none.
func readInstallRecord(libDir string) (*installRecord, error) {
	data, err := os.ReadFile(filepath.Join(libDir, installRecordFile)) //nolint:gosec // libDir is operator configuration and installRecordFile a fixed name; the content is only parsed as JSON
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var rec installRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, fmt.Errorf("parse %s in %s: %w", installRecordFile, libDir, err)
	}
	return &rec, nil
}

// downloader is how this package's downloads are made (see fetch.Downloader):
// through httpClient, capped at maxDownloadBytes, each logged to logger under
// this package's component.
func downloader(logger *slog.Logger) fetch.Downloader {
	return fetch.Downloader{Client: httpClient, UserAgent: userAgent, Limit: maxDownloadBytes, Logger: logger.With(logComponent)}
}

// ensureModelFile makes sure spec.File exists, downloading spec.URL into it
// (verified against spec.SHA256 and, when set, spec.Size) when it is absent;
// a download is logged to logger.
func ensureModelFile(ctx context.Context, spec ModelSpec, logger *slog.Logger) error {
	a := asset{URL: spec.URL, SHA256: spec.SHA256, Size: spec.Size}
	return downloader(logger).EnsureFile(ctx, "model "+spec.Name, a, spec.File)
}

// ensureLibs makes sure libDir holds the ONNX Runtime build the variant
// needs. A managed directory (owned by this package) is filled from
// libAssets when empty or when its record names another version or
// variant, and the download and install are logged to logger; an operator
// directory is only checked, never written to.
func ensureLibs(ctx context.Context, libDir string, managed bool, variant string, logger *slog.Logger) error {
	return ensureLibsFor(ctx, platform(), libDir, managed, variant, logger)
}

func ensureLibsFor(ctx context.Context, plat, libDir string, managed bool, variant string, logger *slog.Logger) error {
	_, present := libraryPath(libDir)
	rec, err := readInstallRecord(libDir)
	if err != nil {
		return err
	}
	switch {
	case present == nil && !managed:
		return checkInstallRecord(libDir, rec, variant)
	case present == nil && rec != nil && rec.Version == LibVersion && rec.Variant == variant:
		return nil
	case present == nil:
		logger.Info("replacing the onnxruntime install", logComponent, slog.String("dir", libDir), slog.Any("record", rec), slog.String("want", LibVersion+"/"+variant))
	case !managed:
		return fmt.Errorf("no %s in %s: unpack the lib/ directory of the onnxruntime %s release there (the %s build for %s), or use a managed LibDir, which gets it downloaded", libFile(), libDir, LibVersion, variant, plat)
	}
	a, ok := libAssets[plat+"/"+variant]
	if !ok {
		return fmt.Errorf("no pinned onnxruntime %s build for %s: use an unmanaged LibDir holding the release's lib/ directory", variant, plat)
	}
	record, err := json.Marshal(installRecord{Version: LibVersion, Variant: variant})
	if err != nil {
		return err
	}
	// the staged install (see fetch.InstallDir) also replaces a half-filled
	// directory from an older crash (no library, see above)
	err = downloader(logger).InstallDir(ctx, fetch.DirInstall{
		Dir: libDir, What: "onnxruntime " + LibVersion + " (" + variant + ")", Archive: a, Extract: extractLibs,
		RecordFile: installRecordFile, Record: record,
	})
	if err != nil {
		return err
	}
	logger.Info("installed onnxruntime "+LibVersion, logComponent, slog.String("dir", libDir), slog.String("variant", variant))
	return nil
}

// checkInstallRecord refuses an operator directory whose record names
// another version or variant than the one asked for. A directory without a
// record (manual install) passes here: the version is checked after load,
// and a build without the provider fails when the session is created.
func checkInstallRecord(libDir string, rec *installRecord, variant string) error {
	if rec == nil || (rec.Version == LibVersion && rec.Variant == variant) {
		return nil
	}
	return fmt.Errorf("%s holds onnxruntime %s (%s) but %s (%s) is wanted: install that build there, or use a managed LibDir", libDir, rec.Version, rec.Variant, LibVersion, variant)
}

// wantLib reports whether an archive entry is one of the runtime libraries:
// a libonnxruntime*.so* or *.dylib directly under lib/ (the core library,
// its symlinks, and the provider libraries of the CUDA build). The headers,
// pkgconfig and cmake files, and the macOS debug symbols, which carry the
// library's own name inside a .dSYM directory, are skipped.
func wantLib(p string) bool {
	if path.Base(path.Dir(p)) != "lib" {
		return false
	}
	name := path.Base(p)
	return strings.HasPrefix(name, "libonnxruntime") && (strings.Contains(name, ".so") || strings.Contains(name, ".dylib"))
}

// extractLibs copies the runtime libraries out of a release tarball into
// dir, flattened (see fetch.ExtractTarGz), and checks the versioned library
// is among them.
func extractLibs(archive, dir string) error {
	if _, err := fetch.ExtractTarGz(archive, dir, wantLib, maxLibraryBytes); err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(dir, libFile())); err != nil {
		return fmt.Errorf("no %s in %s", libFile(), archive)
	}
	return nil
}

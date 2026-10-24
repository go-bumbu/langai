package yzma

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// Pinned artifacts. They belong to one yzma release: yzma's Go structs mirror
// exactly one llama.cpp ABI, and the model was verified against this build.
// Bump them together with the yzma version in go.mod; pins_test.go fails
// until pinnedYzma matches the module version again. The digests come from
// the manifest yzma verifies against (llama-cpp-builder release v0.6.0,
// asset v0.6.0.json) and from the model's LFS metadata on Hugging Face.
const (
	// pinnedYzma is the yzma release the pins below were taken from.
	pinnedYzma = "v1.29.1"
	// LibTag is the llama.cpp build (ggml-org release tag) that yzma
	// pinnedYzma installs by default; name a managed library directory after
	// it so a bump never loads stale native code.
	LibTag = "b11429"
	// libVersion is the version that build reports through llama.Version;
	// runtimeLoader.load (backend.go) refuses libraries reporting another.
	libVersion = "0.6.0"
	// defaultModelFile is the GGUF behind DefaultModel inside ModelsDir.
	defaultModelFile = "embeddinggemma-300M-Q8_0.gguf"
	// installRecordFile is written by `yzma install`; an operator directory
	// that has one is checked against LibTag before anything is loaded.
	installRecordFile = "yzma-install.json"
	// maxDownloadBytes caps any single download or extracted library; the largest evaluation candidate (bge-m3 Q8_0) is 635 MB.
	maxDownloadBytes = 1 << 30
	// downloadTimeout bounds the first-use downloads as a whole.
	downloadTimeout = 15 * time.Minute
)

// asset is one downloadable file with its expected digest.
type asset struct {
	URL    string
	SHA256 string
	Size   int64 // exact size when known, 0 to skip the check
}

// defaultModelAsset is unsloth's Q8_0 GGUF conversion of DefaultModel, which
// Google publishes under the Gemma Terms of Use.
var defaultModelAsset = asset{
	URL:    "https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf",
	SHA256: "a0f7b4e13c397a6e1b32c2de75b1f65a14c92ec524d5f674d94a4290a1c4969b",
	Size:   328577056,
}

// libAssets are the CPU builds of LibTag per GOOS/GOARCH. linux/amd64 is the
// release target; linux/arm64 is pinned because its digest is published in
// the same manifest. Other platforms install the libraries with the yzma CLI
// into an operator-supplied LibDir (see Config.ManagedLibDir).
var libAssets = map[string]asset{
	"linux/amd64": {
		URL:    "https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-ubuntu-x64.tar.gz",
		SHA256: "f6d25dde8f51133143d1453da4fd5f73b145127177612a283bf7995957af3392",
		Size:   17693462,
	},
	"linux/arm64": {
		URL:    "https://github.com/hybridgroup/llama-cpp-builder/releases/download/v0.6.0/llama-v0.6.0-bin-ubuntu-cpu-arm64.tar.gz",
		SHA256: "296e1cac3c68912c8d77bbaea9a51b2c19aecf67173b7d2d1ff52d6ab4c5d8de",
		Size:   8759371,
	},
}

// httpClient is the client downloads go through. The default client honours
// HTTP(S)_PROXY; tests point assets at an httptest server instead.
var httpClient = http.DefaultClient

// platform is the libAssets key for this binary.
func platform() string { return runtime.GOOS + "/" + runtime.GOARCH }

// libFile is the library whose presence marks an installed directory.
func libFile() string {
	if runtime.GOOS == "darwin" {
		return "libllama.dylib"
	}
	return "libllama.so"
}

// ensureModelFile makes sure spec.File exists, downloading spec.URL into it
// (verified against spec.SHA256 and, when set, spec.Size) when it is absent;
// a download is logged to logger.
func ensureModelFile(ctx context.Context, spec ModelSpec, logger *slog.Logger) error {
	if _, err := os.Stat(spec.File); err == nil {
		return nil
	}
	if spec.URL == "" {
		return fmt.Errorf("model %q: %s does not exist and the spec has no URL to download it from", spec.Name, spec.File)
	}
	if err := os.MkdirAll(filepath.Dir(spec.File), 0o750); err != nil {
		return err
	}
	a := asset{URL: spec.URL, SHA256: spec.SHA256, Size: spec.Size}
	if err := fetchFile(ctx, "model "+spec.Name, a, spec.File, logger); err != nil {
		return fmt.Errorf("download %s: %w", spec.Name, err)
	}
	logger.Info("downloaded model", logComponent, slog.String("path", spec.File))
	return nil
}

// ensureLibs makes sure libDir holds the llama.cpp libraries. A managed
// directory (owned by this package) is filled from libAssets when empty, and
// the download and install are logged to logger; an operator directory is
// only checked, never written to.
func ensureLibs(ctx context.Context, libDir string, managed bool, logger *slog.Logger) error {
	return ensureLibsFor(ctx, platform(), libDir, managed, logger)
}

func ensureLibsFor(ctx context.Context, plat, libDir string, managed bool, logger *slog.Logger) error {
	if _, err := os.Stat(filepath.Join(libDir, libFile())); err == nil {
		if managed {
			return nil
		}
		return checkInstallRecord(libDir)
	}
	if !managed {
		return fmt.Errorf("no %s in %s: install llama.cpp build %s there with `yzma install --lib %s` (yzma %s installs it by default), or use a managed LibDir, which gets the pinned build downloaded", libFile(), libDir, LibTag, libDir, pinnedYzma)
	}
	a, ok := libAssets[plat]
	if !ok {
		return fmt.Errorf("no pinned llama.cpp build for %s: use an unmanaged LibDir populated with `yzma install`", plat)
	}
	parent := filepath.Dir(libDir)
	if err := os.MkdirAll(parent, 0o750); err != nil {
		return err
	}
	// Download and extract next to the final directory, then rename it into
	// place: an interrupted install never leaves a half-filled libDir, and a
	// half-filled one from an older crash (no libFile, see above) is replaced.
	staging, err := os.MkdirTemp(parent, ".partial-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(staging) }()
	archive := filepath.Join(staging, "libs.tar.gz")
	if err := fetchFile(ctx, "llama.cpp "+LibTag, a, archive, logger); err != nil {
		return fmt.Errorf("download llama.cpp %s: %w", LibTag, err)
	}
	libs := filepath.Join(staging, "libs")
	if err := os.Mkdir(libs, 0o750); err != nil {
		return err
	}
	if err := extractLibs(archive, libs); err != nil {
		return fmt.Errorf("extract llama.cpp %s: %w", LibTag, err)
	}
	if err := os.RemoveAll(libDir); err != nil {
		return err
	}
	if err := os.Rename(libs, libDir); err != nil {
		return err
	}
	logger.Info("installed llama.cpp "+LibTag, logComponent, slog.String("dir", libDir))
	return nil
}

// installRecord is the subset of `yzma install`'s record this package reads.
type installRecord struct {
	Tag         string `json:"tag"`
	UpstreamTag string `json:"upstream_tag"`
}

// checkInstallRecord refuses an operator directory whose `yzma install`
// record names a different llama.cpp build than LibTag. A directory without
// a record (manual install) passes here and is checked by version after
// load (see runtimeLoader.load).
func checkInstallRecord(libDir string) error {
	data, err := os.ReadFile(filepath.Join(libDir, installRecordFile)) //nolint:gosec // libDir is operator configuration and installRecordFile a fixed name; the content is only parsed as JSON
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var rec installRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return fmt.Errorf("parse %s in %s: %w", installRecordFile, libDir, err)
	}
	if rec.Tag == LibTag || rec.UpstreamTag == LibTag {
		return nil
	}
	return fmt.Errorf("%s holds llama.cpp %s (upstream %q) but semsearch/yzma is pinned to build %s: reinstall it there with yzma %s's default version, or use a managed LibDir", libDir, rec.Tag, rec.UpstreamTag, LibTag, pinnedYzma)
}

// fetchFile downloads a to dest, verifying size and SHA-256 before the file
// appears under its final name: the bytes go to a temp file in the same
// directory and are renamed into place only once they check out. Since a
// download can take minutes, it is logged (what names it) before it starts.
func fetchFile(ctx context.Context, what string, a asset, dest string, logger *slog.Logger) error {
	logger.Info("downloading "+what, logComponent, slog.String("url", a.URL), slog.Int64("size", a.Size), slog.String("sha256", a.SHA256), slog.String("dest", dest))
	tmp, sum, n, err := downloadTemp(ctx, a.URL, dest)
	if err != nil {
		return err
	}
	if a.Size != 0 && n != a.Size {
		_ = os.Remove(tmp)
		return fmt.Errorf("%s: got %d bytes, expected %d", a.URL, n, a.Size)
	}
	if sum != a.SHA256 {
		_ = os.Remove(tmp)
		return fmt.Errorf("%s: sha256 %s, expected %s", a.URL, sum, a.SHA256)
	}
	if err = os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// Download fetches url into dest with nothing to verify it against and
// returns the SHA-256 and size of what it wrote, so a new model can be pinned
// in the evaluation command's models file. Open never calls it: a ModelSpec
// with a URL must carry the digest.
func Download(ctx context.Context, url, dest string, logger *slog.Logger) (string, int64, error) {
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o750); err != nil {
		return "", 0, err
	}
	logger.Info("downloading unpinned file", logComponent, slog.String("url", url), slog.String("dest", dest))
	tmp, sum, n, err := downloadTemp(ctx, url, dest)
	if err != nil {
		return "", 0, err
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return "", 0, err
	}
	return sum, n, nil
}

// downloadTemp streams url into a closed temp file next to dest, capped at
// maxDownloadBytes, and returns the temp path with the SHA-256 and size of
// its content. On any error nothing is left behind.
func downloadTemp(ctx context.Context, url, dest string) (tmpPath, sum string, n int64, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", "", 0, err
	}
	req.Header.Set("User-Agent", "langai-semsearch")
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", "", 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return "", "", 0, fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dest), ".download-*")
	if err != nil {
		return "", "", 0, err
	}
	defer func() {
		_ = tmp.Close()
		if err != nil {
			_ = os.Remove(tmp.Name())
		}
	}()
	h := sha256.New()
	n, err = io.Copy(io.MultiWriter(tmp, h), io.LimitReader(resp.Body, maxDownloadBytes+1))
	if err != nil {
		return "", "", 0, err
	}
	if n > maxDownloadBytes {
		err = fmt.Errorf("%s exceeds the %d byte download cap", url, maxDownloadBytes)
		return "", "", 0, err
	}
	if err = tmp.Close(); err != nil {
		return "", "", 0, err
	}
	return tmp.Name(), hex.EncodeToString(h.Sum(nil)), n, nil
}

// wantLib reports whether an archive entry (by base name) is one of the
// runtime libraries: libggml*.so* (base, ggml, every CPU variant, any GPU
// backend) and libllama.so*. The tools' impl libraries, libmtmd and the RPC
// backend are skipped.
func wantLib(name string) bool {
	if strings.HasPrefix(name, "libggml-rpc") {
		return false
	}
	if strings.HasPrefix(name, "libggml") || strings.HasPrefix(name, "libllama.so") {
		return strings.Contains(name, ".so")
	}
	return false
}

// extractLibs copies the runtime libraries out of a llama.cpp release
// tarball into dir, flattened (release archives nest them under a
// version-named directory). Symlinks are recreated only when they point at a
// bare library name in the same directory; anything else is rejected rather
// than written.
func extractLibs(archive, dir string) error {
	f, err := os.Open(archive) //nolint:gosec // archive is the download this package just verified in its own staging directory, not a path taken from archive content
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer func() { _ = gz.Close() }()
	tr := tar.NewReader(gz)
	found := 0
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		name := filepath.Base(hdr.Name)
		if !wantLib(name) {
			continue
		}
		dest := filepath.Join(dir, name)
		switch hdr.Typeflag {
		case tar.TypeReg:
			if err := writeLibrary(dest, tr); err != nil {
				return err
			}
		case tar.TypeSymlink:
			target := hdr.Linkname
			if target != filepath.Base(target) || !wantLib(target) {
				return fmt.Errorf("refusing symlink %s -> %s", name, target)
			}
			if err := os.Symlink(target, dest); err != nil {
				return err
			}
		default:
			continue
		}
		found++
	}
	if found == 0 {
		return fmt.Errorf("no llama.cpp libraries found in %s", archive)
	}
	return nil
}

// writeLibrary writes one extracted library, bounded by maxDownloadBytes.
func writeLibrary(dest string, r io.Reader) (err error) {
	out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755) //nolint:gosec // dest is a wantLib-filtered base name inside this package's staging directory, writes are capped by maxDownloadBytes, and shared libraries must be executable
	if err != nil {
		return err
	}
	defer func() {
		if cerr := out.Close(); err == nil {
			err = cerr
		}
	}()
	n, err := io.Copy(out, io.LimitReader(r, maxDownloadBytes+1))
	if err != nil {
		return err
	}
	if n > maxDownloadBytes {
		return fmt.Errorf("%s exceeds the %d byte cap", dest, maxDownloadBytes)
	}
	return nil
}

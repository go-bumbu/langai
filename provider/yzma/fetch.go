package yzma

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

	"github.com/go-bumbu/sense/internal/fetch"
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
	// defaultCompleterFile is the GGUF behind DefaultCompleterModel inside
	// ModelsDir.
	defaultCompleterFile = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
	// installRecordFile is written by `yzma install`; an operator directory
	// that has one is checked against LibTag before anything is loaded.
	installRecordFile = "yzma-install.json"
	// maxDownloadBytes caps any single download or extracted library; the
	// largest evaluation candidate (bge-m3 Q8_0) is 635 MB. A download that
	// is pinned to a larger exact size (a chat model) is capped at that size
	// instead, see fetch.Downloader.
	maxDownloadBytes = 1 << 30
	// downloadTimeout bounds the first-use downloads as a whole when the
	// caller's ctx carries no deadline of its own.
	downloadTimeout = 15 * time.Minute
	// completerDownloadTimeout does the same for a chat model's first-use
	// downloads: the pinned one is 2.5 GB, which takes longer than downloadTimeout
	// on a slow link.
	completerDownloadTimeout = time.Hour
	// userAgent names this package's downloads.
	userAgent = "sense-semsearch"
	// cudaRuntimeMajor is the CUDA runtime the pinned CUDA build (the
	// "cuda-13.4" asset in libAssets) links against: its backend library
	// needs libcudart.so.13 and libcublas.so.13 from the host. internal/hw
	// only reports CUDA when the loader can find them, so a machine with the
	// NVIDIA driver and no such runtime gets the Vulkan build instead of a
	// CUDA build that cannot offload. Bump it with the asset.
	cudaRuntimeMajor = 13
)

// asset is one downloadable file with its expected digest.
type asset = fetch.Asset

// defaultModelAsset is unsloth's Q8_0 GGUF conversion of DefaultModel, which
// Google publishes under the Gemma Terms of Use.
var defaultModelAsset = asset{
	URL:    "https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf",
	SHA256: "a0f7b4e13c397a6e1b32c2de75b1f65a14c92ec524d5f674d94a4290a1c4969b",
	Size:   328577056,
}

// defaultCompleterAsset is unsloth's Q4_K_M GGUF conversion of
// DefaultCompleterModel, which Alibaba publishes under Apache-2.0. The digest
// and size are the LFS metadata Hugging Face reports for the file.
var defaultCompleterAsset = asset{
	URL:    "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
	SHA256: "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
	Size:   2497281120,
}

// libAssets are the builds of LibTag per GOOS/GOARCH and backend (see
// chooseBackend): the plain CPU build, and the builds that add one GPU
// backend library to it (Vulkan, CUDA 13.4 with the runtime expected on the
// host; the macOS build has Metal built in). linux/amd64 is the release
// target; the others are pinned because their digests are published with
// the same release. Other platforms install the libraries with the yzma CLI
// into an operator-supplied LibDir (see Runtime.ManagedLibDir). The digests
// are those of the ggml-org release assets, except linux/arm64/cpu, which
// comes from the manifest yzma verifies against.
var libAssets = map[string]asset{
	"linux/amd64/cpu": {
		URL:    "https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-ubuntu-x64.tar.gz",
		SHA256: "f6d25dde8f51133143d1453da4fd5f73b145127177612a283bf7995957af3392",
		Size:   17693462,
	},
	"linux/amd64/vulkan": {
		URL:    "https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-ubuntu-vulkan-x64.tar.gz",
		SHA256: "632c4e98feba2b94407a2130e3133e0c3aefb0ea1ab41337e926d8bfafdd0b74",
		Size:   31636673,
	},
	"linux/amd64/cuda": {
		URL:    "https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-ubuntu-cuda-13.4-x64.tar.gz",
		SHA256: "8082b7eaa74a714c9fecca19128f751c8e32da763ee8096b8ad1e824da7621d3",
		Size:   152519318,
	},
	"linux/arm64/cpu": {
		URL:    "https://github.com/hybridgroup/llama-cpp-builder/releases/download/v0.6.0/llama-v0.6.0-bin-ubuntu-cpu-arm64.tar.gz",
		SHA256: "296e1cac3c68912c8d77bbaea9a51b2c19aecf67173b7d2d1ff52d6ab4c5d8de",
		Size:   8759371,
	},
	"linux/arm64/vulkan": {
		URL:    "https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-ubuntu-vulkan-arm64.tar.gz",
		SHA256: "702d99c4219b4314cc6d3b10fb41ae96cbfc68226305c2681269dfb51487af34",
		Size:   24845090,
	},
	"darwin/arm64/metal": {
		URL:    "https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-macos-arm64.tar.gz",
		SHA256: "740288ec6887be94280a5dfa25b5e23a78285cab104519e6c7e218904ee82459",
		Size:   11971406,
	},
}

// libRecordFile is written into a managed LibDir with the tag and backend
// installed there, so a later request for a GPU build replaces a CPU
// install (the GPU builds are supersets, the other way round is a no-op),
// and a later version of this package, pinned to another tag, replaces
// either.
const libRecordFile = "sense-libs.json"

// libRecord is what libRecordFile holds.
type libRecord struct {
	Tag     string `json:"tag"`
	Backend string `json:"backend"`
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

// libsWant is what ensureLibs is asked for: a directory, the llama.cpp
// build it should hold, and what to do with it once it does.
type libsWant struct {
	// plat is the libAssets key of the platform, "linux/amd64".
	plat string
	// dir is the absolute library directory.
	dir string
	// managed marks dir as owned by this package (Runtime.ManagedLibDir).
	managed bool
	// build is the build dir should hold (see chooseBackend).
	build build
	// explicit says the build was asked for by name (Runtime.Accel) rather
	// than picked from the hardware.
	explicit bool
	// loadedDir is where this process loaded llama.cpp from, "" while it has
	// not; nil means it has not.
	loadedDir func() string
	// use loads the libraries from dir. It runs while a managed dir is
	// locked, so no other process replaces them halfway; nil does nothing.
	use func() error
}

// ensureLibs makes sure w.dir holds the llama.cpp libraries of w.build and
// then runs w.use on them. A managed directory is filled from libAssets when
// it is empty, when it holds this package's CPU build and a GPU build is
// wanted, or when its record names another llama.cpp tag than LibTag (see
// needsReinstall), under the directory's lock (see fetch.InstallDir), and the
// download and install are logged to logger. It is never filled over files
// this package did not install, and never while this process runs the
// libraries in it; a process that does run them, and wants nothing else,
// does not wait for the lock. An operator directory is only checked, never
// written to.
func ensureLibs(ctx context.Context, w libsWant, logger *slog.Logger) error {
	if !w.managed {
		if _, err := os.Stat(filepath.Join(w.dir, libFile())); err != nil {
			return fmt.Errorf("no %s in %s: install llama.cpp build %s there with `yzma install --lib %s` (yzma %s installs it by default), or use a managed LibDir, which gets the pinned build downloaded", libFile(), w.dir, LibTag, w.dir, pinnedYzma)
		}
		if err := checkInstallRecord(w.dir); err != nil {
			return err
		}
		return w.load()
	}
	rec, err := json.Marshal(libRecord{Tag: LibTag, Backend: w.build})
	if err != nil {
		return err
	}
	installed, err := downloader(logger).InstallDir(ctx, fetch.DirInstall{
		Dir: w.dir, What: "llama.cpp " + LibTag, Archive: libAssets[w.plat+"/"+w.build], Extract: extractLibs,
		RecordFile: libRecordFile, Record: rec,
		Ready: func() (bool, error) { return w.ready(logger) },
		Use:   w.load,
		// a process that runs the libraries from here does not queue behind
		// another process's download at its next open
		Loaded: func() bool { return w.loadedDir != nil && w.loadedDir() == w.dir },
	})
	if installed {
		logger.Info("installed llama.cpp "+LibTag, logComponent, slog.String("dir", w.dir), slog.String("build", w.build))
	}
	return err
}

// load runs w.use, which may be nil.
func (w libsWant) load() error {
	if w.use == nil {
		return nil
	}
	return w.use()
}

// ready reports whether the managed w.dir can be used as it is, and
// refuses, with an error, an install that cannot be made: of a build that
// is not pinned for the platform, or over libraries this process has
// already loaded. Those cannot be swapped under it; when the build was only
// picked from the hardware the process carries on with what it loaded.
func (w libsWant) ready(logger *slog.Logger) (bool, error) {
	_, err := os.Stat(filepath.Join(w.dir, libFile()))
	present := err == nil
	if present && !needsReinstall(w.dir, w.build) {
		return true, nil
	}
	if w.loadedDir != nil && w.loadedDir() == w.dir {
		if !w.explicit {
			logger.Warn("llama.cpp is already loaded from a build without the "+w.build+" backend; keeping it for this process", logComponent, slog.String("dir", w.dir))
			return true, nil
		}
		return false, fmt.Errorf("llama.cpp is already loaded from %s, which does not hold the %s build: libraries a process has loaded cannot be replaced under it; open with that Accel first, or restart", w.dir, w.build)
	}
	if _, ok := libAssets[w.plat+"/"+w.build]; !ok {
		return false, fmt.Errorf("no pinned llama.cpp %s build for %s: use an unmanaged LibDir populated with `yzma install`", w.build, w.plat)
	}
	if present {
		rec, _ := readLibRecord(w.dir)
		logger.Info("replacing the llama.cpp install with the "+w.build+" build of "+LibTag, logComponent,
			slog.String("dir", w.dir), slog.String("installed_tag", rec.Tag), slog.String("installed_build", rec.Backend))
	}
	return false, nil
}

// readLibRecord returns the record a managed install left in libDir; ok is
// false when there is none that can be read.
func readLibRecord(libDir string) (rec libRecord, ok bool) {
	data, err := os.ReadFile(filepath.Join(libDir, libRecordFile)) //nolint:gosec // libDir is this package's own directory and libRecordFile a fixed name; the content is only parsed as JSON
	if err != nil {
		return libRecord{}, false
	}
	if err := json.Unmarshal(data, &rec); err != nil {
		return libRecord{}, false
	}
	return rec, true
}

// needsReinstall reports whether a managed libDir that holds libraries must
// be replaced to serve build b. With a record it must when the record names
// another tag than LibTag, whatever the backend (the install of an older
// version of this package: its libraries fail the version check at load,
// and would for as long as the process lives), and when b is a GPU build
// the record does not name; a GPU build also serves the CPU, so that
// request keeps what is there. Without a record that can be read, a
// directory with no record file serves the CPU as it is (libraries someone
// put there, checked by version after load) and is no GPU build, and one
// whose record file is broken is replaced.
func needsReinstall(libDir string, b build) bool {
	rec, ok := readLibRecord(libDir)
	if !ok {
		if _, err := os.Lstat(filepath.Join(libDir, libRecordFile)); err == nil {
			return true
		}
		return b != "cpu"
	}
	if rec.Tag != LibTag {
		return true
	}
	return b != "cpu" && rec.Backend != b
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
	return fmt.Errorf("%s holds llama.cpp %s (upstream %q) but provider/yzma is pinned to build %s: reinstall it there with yzma %s's default version, or use a managed LibDir", libDir, rec.Tag, rec.UpstreamTag, LibTag, pinnedYzma)
}

// wantLib reports whether an archive entry (by base name) is one of the
// runtime libraries: libggml* (base, ggml, every CPU variant, any GPU
// backend), libllama and libmtmd (the multimodal projector runtime the
// Vision embedder drives), as .so or .dylib. The tools' impl libraries and
// the RPC backend are skipped.
func wantLib(name string) bool {
	if strings.HasPrefix(name, "libggml-rpc") {
		return false
	}
	if strings.HasPrefix(name, "libggml") || strings.HasPrefix(name, "libllama.") || strings.HasPrefix(name, "libmtmd.") {
		return strings.Contains(name, ".so") || strings.Contains(name, ".dylib")
	}
	return false
}

// extractLibs copies the runtime libraries out of a llama.cpp release
// tarball into dir, flattened (release archives nest them under a
// version-named directory), see fetch.ExtractTarGz.
func extractLibs(archive, dir string) error {
	found, err := fetch.ExtractTarGz(archive, dir, func(p string) bool { return wantLib(path.Base(p)) }, maxDownloadBytes)
	if err != nil {
		return err
	}
	if found == 0 {
		return fmt.Errorf("no llama.cpp libraries found in %s", archive)
	}
	return nil
}

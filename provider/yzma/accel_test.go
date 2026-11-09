package yzma

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/internal/hw"
)

// cpuOnly makes the hardware look like a plain CPU machine for the test, so
// a managed LibDir asks for the cpu build whatever the host has.
func cpuOnly(t *testing.T) {
	t.Helper()
	orig := detectHW
	detectHW = func() hw.Report { return hw.Report{Findings: []hw.Finding{{Accelerator: hw.CPU, Reason: "test"}}} }
	t.Cleanup(func() { detectHW = orig })
}

func report(accels ...hw.Accelerator) hw.Report {
	var r hw.Report
	for _, a := range accels {
		r.Findings = append(r.Findings, hw.Finding{Accelerator: a, Reason: "test"})
	}
	return r
}

func TestChooseBackend(t *testing.T) {
	cases := []struct {
		name   string
		accel  Accel
		report hw.Report
		plat   string
		want   string
		err    string
	}{
		{"auto, nothing found", AccelAuto, report(hw.CPU), "linux/amd64", "cpu", ""},
		{"auto, vulkan found", AccelAuto, report(hw.Vulkan, hw.CPU), "linux/amd64", "vulkan", ""},
		{"auto prefers cuda over vulkan", AccelAuto, report(hw.CUDA, hw.Vulkan, hw.CPU), "linux/amd64", "cuda", ""},
		{"auto skips rocm, which has no build", AccelAuto, report(hw.ROCm, hw.Vulkan, hw.CPU), "linux/amd64", "vulkan", ""},
		{"auto on arm64 has no cuda build", AccelAuto, report(hw.CUDA, hw.CPU), "linux/arm64", "cpu", ""},
		{"auto on macOS", AccelAuto, report(hw.Metal, hw.CPU), "darwin/arm64", "metal", ""},
		{"cpu on macOS is the metal build", AccelCPU, report(hw.Metal, hw.CPU), "darwin/arm64", "metal", ""},
		{"explicit vulkan", AccelVulkan, report(hw.CPU), "linux/amd64", "vulkan", ""},
		{"explicit cuda on arm64", AccelCUDA, report(hw.CPU), "linux/arm64", "", "no pinned llama.cpp build with cuda for linux/arm64"},
		{"explicit metal on linux", AccelMetal, report(hw.CPU), "linux/amd64", "", "no pinned llama.cpp build with metal"},
		{"unknown platform", AccelAuto, report(hw.CPU), "plan9/mips", "", "no pinned llama.cpp build for plan9/mips"},
	}
	for _, c := range cases {
		got, err := chooseBackend(c.accel, c.report, c.plat)
		if c.err != "" {
			if err == nil || !strings.Contains(err.Error(), c.err) {
				t.Errorf("%s: err = %v, want %q", c.name, err, c.err)
			}
			continue
		}
		if err != nil || got != c.want {
			t.Errorf("%s: got %q, %v; want %q", c.name, got, err, c.want)
		}
	}
}

func TestAccelNormalized(t *testing.T) {
	for _, a := range []Accel{AccelAuto, AccelCPU, AccelVulkan, AccelCUDA, AccelMetal} {
		if _, err := a.normalized(); err != nil {
			t.Errorf("%q: %v", a, err)
		}
	}
	if _, err := Accel("opencl").normalized(); err == nil || !strings.Contains(err.Error(), "opencl") {
		t.Errorf("opencl: %v", err)
	}
}

func TestResolveAccel(t *testing.T) {
	cpu := device{name: "CPU", backend: "CPU", description: "AMD Ryzen"}
	vulkan := device{name: "Vulkan0", backend: "Vulkan", description: "AMD Radeon 890M Graphics", gpu: true}
	llvmpipe := device{name: "Vulkan1", backend: "Vulkan", description: "llvmpipe (LLVM 21.1.8, 256 bits)", gpu: true}
	cuda := device{name: "CUDA0", backend: "CUDA", description: "NVIDIA RTX", gpu: true}
	cases := []struct {
		name    string
		accel   Accel
		layers  int
		devices []device
		want    int32
		used    string
		err     string
	}{
		{"auto, cpu only", AccelAuto, 0, []device{cpu}, 0, "cpu", ""},
		{"auto, no devices", AccelAuto, 0, nil, 0, "cpu", ""},
		{"auto, a gpu", AccelAuto, 0, []device{vulkan, cpu}, allLayers, "Vulkan0: AMD Radeon 890M Graphics", ""},
		{"auto skips llvmpipe", AccelAuto, 0, []device{llvmpipe, cpu}, 0, "cpu", ""},
		{"auto takes the first hardware gpu", AccelAuto, 0, []device{llvmpipe, cuda, vulkan, cpu}, allLayers, "CUDA0: NVIDIA RTX", ""},
		{"auto with a layer cap", AccelAuto, 12, []device{vulkan, cpu}, 12, "Vulkan0: AMD Radeon 890M Graphics", ""},
		{"cpu ignores the gpu", AccelCPU, 0, []device{vulkan, cpu}, 0, "cpu", ""},
		{"explicit vulkan", AccelVulkan, 0, []device{cuda, vulkan, cpu}, allLayers, "Vulkan0: AMD Radeon 890M Graphics", ""},
		{"explicit cuda missing", AccelCUDA, 0, []device{vulkan, cpu}, 0, "", `Accel "cuda": no CUDA device registered (devices: Vulkan0, CPU)`},
		{"explicit vulkan with only llvmpipe", AccelVulkan, 0, []device{llvmpipe, cpu}, 0, "", "no Vulkan device registered"},
	}
	for _, c := range cases {
		got, used, err := resolveAccel(c.accel, c.layers, c.devices)
		if c.err != "" {
			if err == nil || !strings.Contains(err.Error(), c.err) {
				t.Errorf("%s: err = %v, want %q", c.name, err, c.err)
			}
			continue
		}
		if err != nil || got != c.want || used != c.used {
			t.Errorf("%s: got %d, %q, %v; want %d, %q", c.name, got, used, err, c.want, c.used)
		}
	}
}

// TestOpenLlama_OffloadsToTheGPUItFinds: the layers resolveAccel picks reach
// loadModel, and an explicit accelerator without its device fails before
// the model is touched.
func TestOpenLlama_OffloadsToTheGPUItFinds(t *testing.T) {
	f := newFakeAPI()
	f.devs = []device{{name: "Vulkan0", backend: "Vulkan", description: "GPU", gpu: true}, {name: "CPU", backend: "CPU"}}
	b, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, 0, AccelAuto, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "loadModel"); args[1] != allLayers {
		t.Errorf("loadModel gpu layers = %v, want %d", args[1], allLayers)
	}
	if b.accel != "Vulkan0: GPU" {
		t.Errorf("accel = %q", b.accel)
	}
	f = newFakeAPI()
	f.devs = []device{{name: "CPU", backend: "CPU"}}
	if _, err := openLlama(f, &runtimeLoader{}, testLibDir, testModelPath, 2, PoolingMean, 0, AccelCUDA, 0, nil); err == nil || !strings.Contains(err.Error(), "no CUDA device") {
		t.Errorf("explicit cuda without a device: %v", err)
	}
	if f.count("loadModel") != 0 {
		t.Error("the model was loaded although the accelerator was missing")
	}
}

// TestOpenCompleter_OffloadsToo: the Completer goes through the same choice.
func TestOpenCompleter_OffloadsToo(t *testing.T) {
	f := newFakeChat()
	f.devs = []device{{name: "CUDA0", backend: "CUDA", description: "NVIDIA", gpu: true}}
	c, err := openCompleter(f, &runtimeLoader{}, testLibDir, testChatModelPath, testSpec(), 2, AccelAuto, 8, nil)
	if err != nil {
		t.Fatal(err)
	}
	if args := f.only(t, "loadModel"); args[1] != int32(8) {
		t.Errorf("loadModel gpu layers = %v, want 8", args[1])
	}
	if c.accel != "CUDA0: NVIDIA" {
		t.Errorf("accel = %q", c.accel)
	}
}

// TestEnsureLibsFor_ReplacesACPUInstallWithAGPUBuild: a managed directory
// holding the cpu build is replaced when a GPU build is wanted, and a GPU
// install serves a later cpu request as it is.
func TestEnsureLibsFor_ReplacesACPUInstallWithAGPUBuild(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the pinned archives hold Linux .so files")
	}
	_, cpuAsset := serve(t, tarGz(t, "llama-b11429", releaseLibs()))
	_, gpuAsset := serve(t, tarGz(t, "llama-b11429", append(releaseLibs(), tarEntry{name: "libggml-vulkan.so", body: "vulkan"})))
	libAssets["test/arch/cpu"], libAssets["test/arch/vulkan"] = cpuAsset, gpuAsset
	t.Cleanup(func() { delete(libAssets, "test/arch/cpu"); delete(libAssets, "test/arch/vulkan") })
	libDir := filepath.Join(t.TempDir(), "llama-cpp", LibTag)
	ctx := context.Background()

	if err := ensureLibsFor(ctx, "test/arch", libDir, true, "cpu", discardLogger()); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(libDir, libRecordFile))
	if err != nil {
		t.Fatal(err)
	}
	var rec libRecord
	if err := json.Unmarshal(data, &rec); err != nil || rec.Tag != LibTag || rec.Backend != "cpu" {
		t.Fatalf("record = %+v, %v", rec, err)
	}
	if _, err := os.Stat(filepath.Join(libDir, "libggml-vulkan.so")); err == nil {
		t.Fatal("the cpu build has a vulkan backend")
	}

	// the same request again is a no-op, a vulkan request replaces the directory
	if needsReinstall(libDir, "cpu") {
		t.Error("cpu over cpu wants a reinstall")
	}
	if err := ensureLibsFor(ctx, "test/arch", libDir, true, "vulkan", discardLogger()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(libDir, "libggml-vulkan.so")); err != nil {
		t.Fatal("the vulkan build was not installed")
	}
	// and a cpu request keeps the vulkan install, while another GPU build
	// would replace it
	if err := ensureLibsFor(ctx, "test/arch", libDir, true, "cpu", discardLogger()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(libDir, "libggml-vulkan.so")); err != nil {
		t.Fatal("the cpu request replaced the vulkan install")
	}
	if !needsReinstall(libDir, "cuda") {
		t.Error("cuda over vulkan does not want a reinstall")
	}
	// a directory without a record predates the record: a GPU request replaces it
	if err := os.Remove(filepath.Join(libDir, libRecordFile)); err != nil {
		t.Fatal(err)
	}
	if !needsReinstall(libDir, "vulkan") {
		t.Error("an unrecorded install is not replaced for a GPU build")
	}
	// an explicit accelerator whose build is not pinned fails
	err = ensureLibsFor(ctx, "test/arch", filepath.Join(t.TempDir(), "x"), true, "cuda", discardLogger())
	if err == nil || !strings.Contains(err.Error(), "no pinned llama.cpp cuda build for test/arch") {
		t.Errorf("unpinned build: %v", err)
	}
}

// TestRuntime_PrepareInstallsTheBuildTheHardwareNeeds: with AccelAuto the
// managed directory gets the build chooseBackend picks for the detected
// hardware, and an unmanaged directory is left alone.
func TestRuntime_PrepareInstallsTheBuildTheHardwareNeeds(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the pinned archives hold Linux .so files")
	}
	_, gpuAsset := serve(t, tarGz(t, "llama-b11429", append(releaseLibs(), tarEntry{name: "libggml-vulkan.so", body: "vulkan"})))
	key := platform() + "/vulkan"
	orig, had := libAssets[key]
	libAssets[key] = gpuAsset
	origDetect := detectHW
	detectHW = func() hw.Report { return report(hw.Vulkan, hw.CPU) }
	t.Cleanup(func() {
		if had {
			libAssets[key] = orig
		} else {
			delete(libAssets, key)
		}
		detectHW = origDetect
	})
	libDir := filepath.Join(t.TempDir(), "llama-cpp", LibTag)
	rt := Runtime{LibDir: libDir, ManagedLibDir: true}
	got, _, err := (&rt).prepare(context.Background(), newFakeAPI(), &runtimeLoader{})
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(got) || rt.Threads <= 0 {
		t.Errorf("prepare: dir %q, threads %d", got, rt.Threads)
	}
	if _, err := os.Stat(filepath.Join(libDir, "libggml-vulkan.so")); err != nil {
		t.Error("the vulkan build was not installed for the detected GPU")
	}

	bad := Runtime{LibDir: libDir, Accel: Accel("opencl")}
	if _, _, err := (&bad).prepare(context.Background(), newFakeAPI(), &runtimeLoader{}); err == nil || !strings.Contains(err.Error(), "opencl") {
		t.Errorf("bad accel: %v", err)
	}
	none := Runtime{}
	if _, _, err := (&none).prepare(context.Background(), newFakeAPI(), &runtimeLoader{}); err == nil {
		t.Error("no LibDir: want error")
	}
	// an unmanaged directory with libraries passes whatever the hardware says
	unmanaged := Runtime{LibDir: libDir}
	if _, _, err := (&unmanaged).prepare(context.Background(), newFakeAPI(), &runtimeLoader{}); err != nil {
		t.Errorf("unmanaged: %v", err)
	}
	// an unmanaged, empty directory still fails as before
	empty := Runtime{LibDir: t.TempDir()}
	if _, _, err := (&empty).prepare(context.Background(), newFakeAPI(), &runtimeLoader{}); err == nil || !errors.Is(err, err) {
		t.Errorf("unmanaged empty: %v", err)
	}
}

// TestRuntime_PrepareWarnsWhenTheGPUBuildFindsNoGPU: a GPU build that was
// installed because the hardware looked like it has a GPU, and then
// registers none, runs every model on the CPU. That must not be silent.
func TestRuntime_PrepareWarnsWhenTheGPUBuildFindsNoGPU(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the pinned archives hold Linux .so files")
	}
	_, gpuAsset := serve(t, tarGz(t, "llama-b11429", append(releaseLibs(), tarEntry{name: "libggml-vulkan.so", body: "vulkan"})))
	key := platform() + "/vulkan"
	orig, had := libAssets[key]
	libAssets[key] = gpuAsset
	origDetect := detectHW
	detectHW = func() hw.Report { return report(hw.Vulkan, hw.CPU) }
	offline(t)
	t.Cleanup(func() {
		if had {
			libAssets[key] = orig
		} else {
			delete(libAssets, key)
		}
		detectHW = origDetect
	})
	gpu := device{name: "Vulkan0", backend: "Vulkan", description: "GPU", gpu: true}
	for _, c := range []struct {
		name  string
		devs  []device
		accel Accel
		warns bool
	}{
		{"the GPU registered", []device{gpu, {name: "CPU", backend: "CPU"}}, AccelAuto, false},
		{"only the CPU registered", []device{{name: "CPU", backend: "CPU"}}, AccelAuto, true},
		{"only a software renderer registered", []device{{name: "Vulkan0", backend: "Vulkan", description: "llvmpipe (LLVM 21)", gpu: true}}, AccelAuto, true},
	} {
		var logs bytes.Buffer
		f := newFakeAPI()
		f.devs = c.devs
		rt := Runtime{LibDir: filepath.Join(t.TempDir(), "llama-cpp", LibTag), ManagedLibDir: true, Accel: c.accel, Logger: slog.New(slog.NewTextHandler(&logs, nil))}
		if _, _, err := (&rt).prepare(context.Background(), f, &runtimeLoader{}); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		warned := strings.Contains(logs.String(), "level=WARN") && strings.Contains(logs.String(), "registered no GPU device")
		if warned != c.warns {
			t.Errorf("%s: warned %v, want %v:\n%s", c.name, warned, c.warns, logs.String())
		}
	}
}

func TestExpectedWarning(t *testing.T) {
	if !expectedWarning("init: embeddings required but some input tokens were not marked as outputs -> overriding") {
		t.Error("the per-input embedding warning should be dropped")
	}
	if expectedWarning("load: control-looking token: 128247 '</s>' was not control-type") {
		t.Error("other warnings must get through")
	}
}

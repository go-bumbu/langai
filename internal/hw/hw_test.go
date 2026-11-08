package hw

import (
	"path/filepath"
	"strings"
	"testing"
)

// fake is a machine described by what exists on it.
type fake struct {
	os, arch string
	files    map[string]bool // paths that exist
	path     map[string]bool // programs on PATH
	env      map[string]string
	globs    map[string][]string // glob pattern → matches
}

func (f fake) goos() string           { return f.os }
func (f fake) goarch() string         { return f.arch }
func (f fake) exists(p string) bool   { return f.files[p] }
func (f fake) lookPath(n string) bool { return f.path[n] }
func (f fake) getenv(k string) string { return f.env[k] }
func (f fake) glob(p string) []string { return f.globs[p] }

func linux() fake {
	return fake{os: "linux", arch: "amd64", files: map[string]bool{}, path: map[string]bool{}, env: map[string]string{}, globs: map[string][]string{}}
}

func accels(r Report) string {
	parts := make([]string, len(r.Findings))
	for i, f := range r.Findings {
		parts[i] = string(f.Accelerator)
	}
	return strings.Join(parts, " ")
}

func TestDetect_NothingButCPU(t *testing.T) {
	r := detect(linux())
	if accels(r) != "cpu" || r.Best().Accelerator != CPU || r.Has(Vulkan) {
		t.Errorf("report = %s", r)
	}
	if r.String() != "cpu (always available)" {
		t.Errorf("String = %q", r)
	}
	if (Report{}).Best().Accelerator != CPU {
		t.Error("empty report: Best is not CPU")
	}
}

func TestDetect_OrderAndReasons(t *testing.T) {
	m := linux()
	m.files["/proc/driver/nvidia/version"] = true
	m.files["/opt/rocm"] = true
	m.globs[filepath.Join("/usr/share/vulkan/icd.d", "*.json")] = []string{"/usr/share/vulkan/icd.d/radeon_icd.json", "/usr/share/vulkan/icd.d/lvp_icd.json", "/usr/share/vulkan/icd.d/nvidia_icd.json"}
	r := detect(m)
	if accels(r) != "cuda rocm vulkan cpu" {
		t.Errorf("order = %s", accels(r))
	}
	if r.Best().Accelerator != CUDA || r.Best().Reason != "NVIDIA driver at /proc/driver/nvidia/version" {
		t.Errorf("best = %+v", r.Best())
	}
	if !r.Has(Vulkan) || !strings.Contains(r.String(), "vulkan (Vulkan ICD nvidia_icd, radeon_icd)") {
		t.Errorf("vulkan reason: %s", r)
	}
	if !strings.Contains(r.String(), "rocm (ROCm at /opt/rocm)") {
		t.Errorf("rocm reason: %s", r)
	}
}

func TestDetect_PathFallbacks(t *testing.T) {
	m := linux()
	m.path["nvidia-smi"] = true
	m.path["rocminfo"] = true
	r := detect(m)
	if accels(r) != "cuda rocm cpu" || r.Findings[0].Reason != "nvidia-smi on PATH" || r.Findings[1].Reason != "rocminfo on PATH" {
		t.Errorf("report = %s", r)
	}
}

func TestDetect_VulkanSoftwareOnlyAndEnv(t *testing.T) {
	m := linux()
	m.globs[filepath.Join("/usr/share/vulkan/icd.d", "*.json")] = []string{"/usr/share/vulkan/icd.d/lvp_icd.json", "/usr/share/vulkan/icd.d/gfxstream_vk_icd.json"}
	if r := detect(m); r.Has(Vulkan) {
		t.Errorf("software ICDs counted as hardware: %s", r)
	}
	m.env["VK_DRIVER_FILES"] = "/custom/icd.json"
	if r := detect(m); !r.Has(Vulkan) || r.Findings[0].Reason != "VK_DRIVER_FILES is set" {
		t.Errorf("env override: %s", r)
	}
	// XDG directories are searched too
	m = linux()
	m.env["XDG_DATA_DIRS"] = "/opt/share:/var/share"
	m.globs[filepath.Join("/var/share/vulkan/icd.d", "*.json")] = []string{"/var/share/vulkan/icd.d/intel_icd.json"}
	if r := detect(m); !r.Has(Vulkan) || !strings.Contains(r.String(), "intel_icd") {
		t.Errorf("XDG_DATA_DIRS: %s", r)
	}
}

func TestDetect_Darwin(t *testing.T) {
	m := fake{os: "darwin", arch: "arm64", files: map[string]bool{"/opt/rocm": true}, path: map[string]bool{"nvidia-smi": true}, env: map[string]string{"VK_DRIVER_FILES": "x"}, globs: map[string][]string{}}
	r := detect(m)
	if accels(r) != "metal cpu" || r.Best().Reason != "Apple silicon" {
		t.Errorf("darwin/arm64 = %s", r)
	}
	m.arch = "amd64"
	if r := detect(m); accels(r) != "cpu" {
		t.Errorf("darwin/amd64 = %s", r)
	}
}

func TestDetect_Windows(t *testing.T) {
	m := fake{os: "windows", arch: "amd64", files: map[string]bool{"/proc/driver/nvidia/version": true}, path: map[string]bool{"nvidia-smi": true}, env: map[string]string{}, globs: map[string][]string{}}
	if r := detect(m); accels(r) != "cuda cpu" || r.Best().Reason != "nvidia-smi on PATH" {
		t.Errorf("windows = %s", r)
	}
}

func TestDetect_RealMachineDoesNotPanic(t *testing.T) {
	r := Detect()
	if !r.Has(CPU) || r.Findings[len(r.Findings)-1].Accelerator != CPU {
		t.Errorf("CPU missing or not last: %s", r)
	}
}

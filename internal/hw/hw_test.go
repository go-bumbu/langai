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
	texts    map[string]string   // text files and their content
}

func (f fake) goos() string             { return f.os }
func (f fake) goarch() string           { return f.arch }
func (f fake) exists(p string) bool     { return f.files[p] }
func (f fake) lookPath(n string) bool   { return f.path[n] }
func (f fake) getenv(k string) string   { return f.env[k] }
func (f fake) glob(p string) []string   { return f.globs[p] }
func (f fake) readFile(p string) string { return f.texts[p] }

func linux() fake {
	return fake{os: "linux", arch: "amd64", files: map[string]bool{}, path: map[string]bool{}, env: map[string]string{}, globs: map[string][]string{}, texts: map[string]string{}}
}

// withCUDA13 puts the CUDA 13 runtime where the loader looks by default.
func withCUDA13(m fake) fake {
	m.files["/usr/lib/x86_64-linux-gnu/libcudart.so.13"] = true
	m.files["/usr/lib/x86_64-linux-gnu/libcublas.so.13"] = true
	return m
}

func accels(r Report) string {
	parts := make([]string, len(r.Findings))
	for i, f := range r.Findings {
		parts[i] = string(f.Accelerator)
	}
	return strings.Join(parts, " ")
}

func TestDetect_NothingButCPU(t *testing.T) {
	r := detect(linux(), Needs{})
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
	m := withCUDA13(linux())
	m.files["/proc/driver/nvidia/version"] = true
	m.files["/opt/rocm"] = true
	m.globs[filepath.Join("/usr/share/vulkan/icd.d", "*.json")] = []string{"/usr/share/vulkan/icd.d/radeon_icd.json", "/usr/share/vulkan/icd.d/lvp_icd.json", "/usr/share/vulkan/icd.d/nvidia_icd.json"}
	r := detect(m, Needs{CUDARuntime: 13})
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
	m := withCUDA13(linux())
	m.path["nvidia-smi"] = true
	m.path["rocminfo"] = true
	r := detect(m, Needs{CUDARuntime: 13})
	if accels(r) != "cuda rocm cpu" || r.Findings[0].Reason != "nvidia-smi on PATH" || r.Findings[1].Reason != "rocminfo on PATH" {
		t.Errorf("report = %s", r)
	}
}

// TestDetect_CUDANeedsItsRuntime: the NVIDIA driver alone runs no CUDA
// build. Without the runtime libraries the build needs, where the loader
// finds them, CUDA is not reported, so that the next accelerator (Vulkan,
// which the same card runs) is picked instead of a build that falls back to
// the CPU. The report still says what was seen.
func TestDetect_CUDANeedsItsRuntime(t *testing.T) {
	driver := func() fake {
		m := linux()
		m.files["/proc/driver/nvidia/version"] = true
		m.globs[filepath.Join("/usr/share/vulkan/icd.d", "*.json")] = []string{"/usr/share/vulkan/icd.d/nvidia_icd.json"}
		return m
	}
	cases := []struct {
		name  string
		setup func(m fake)
		needs Needs
		want  string // the accelerators reported
	}{
		{"driver only", func(fake) {}, Needs{CUDARuntime: 13}, "vulkan cpu"},
		{"the runtime in a default directory", func(m fake) {
			m.files["/usr/lib64/libcudart.so.13"] = true
			m.files["/usr/lib64/libcublas.so.13"] = true
		}, Needs{CUDARuntime: 13}, "cuda vulkan cpu"},
		{"another major version", func(m fake) {
			m.files["/usr/lib64/libcudart.so.12"] = true
			m.files["/usr/lib64/libcublas.so.12"] = true
		}, Needs{CUDARuntime: 13}, "vulkan cpu"},
		{"cudart without cublas", func(m fake) {
			m.files["/usr/lib64/libcudart.so.13"] = true
		}, Needs{CUDARuntime: 13}, "vulkan cpu"},
		{"on LD_LIBRARY_PATH", func(m fake) {
			m.env["LD_LIBRARY_PATH"] = "/opt/app/lib::/opt/cuda-13/lib64"
			m.files["/opt/cuda-13/lib64/libcudart.so.13"] = true
			m.files["/opt/cuda-13/lib64/libcublas.so.13"] = true
		}, Needs{CUDARuntime: 13}, "cuda vulkan cpu"},
		{"registered with the loader", func(m fake) {
			m.texts["/etc/ld.so.conf"] = "# the usual\ninclude /etc/ld.so.conf.d/*.conf\n"
			m.globs["/etc/ld.so.conf.d/*.conf"] = []string{"/etc/ld.so.conf.d/cuda-13-4.conf", "/etc/ld.so.conf.d/x86_64.conf"}
			m.texts["/etc/ld.so.conf.d/cuda-13-4.conf"] = "/usr/local/cuda-13.4/targets/x86_64-linux/lib # toolkit\n"
			m.texts["/etc/ld.so.conf.d/x86_64.conf"] = "include nested/*.conf\nhwcap 1 nosegneg\n"
			m.files["/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcudart.so.13"] = true
			m.files["/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcublas.so.13"] = true
		}, Needs{CUDARuntime: 13}, "cuda vulkan cpu"},
		{"installed but not on the loader's path", func(m fake) {
			m.files["/usr/local/cuda-13.4/lib64/libcudart.so.13"] = true
			m.files["/usr/local/cuda-13.4/lib64/libcublas.so.13"] = true
		}, Needs{CUDARuntime: 13}, "vulkan cpu"},
		{"any version will do", func(m fake) {
			m.globs["/usr/lib64/libcudart.so.*"] = []string{"/usr/lib64/libcudart.so.12"}
			m.globs["/usr/lib64/libcublas.so.*"] = []string{"/usr/lib64/libcublas.so.12"}
		}, Needs{}, "cuda vulkan cpu"},
		{"any version, none there", func(fake) {}, Needs{}, "vulkan cpu"},
	}
	for _, c := range cases {
		m := driver()
		c.setup(m)
		r := detect(m, c.needs)
		if accels(r) != c.want {
			t.Errorf("%s: found %s, want %s", c.name, accels(r), c.want)
		}
		if r.Has(CUDA) != (len(r.Unusable) == 0) {
			t.Errorf("%s: cuda found %v with %d unusable", c.name, r.Has(CUDA), len(r.Unusable))
		}
	}
	r := detect(driver(), Needs{CUDARuntime: 13})
	want := "no cuda (NVIDIA driver at /proc/driver/nvidia/version, but no libcudart.so.13, libcublas.so.13 on the loader's path (the CUDA runtime is not installed))"
	if !strings.HasSuffix(r.String(), want) || r.Best().Accelerator != Vulkan {
		t.Errorf("report = %s\nwant it to end with %s", r, want)
	}
}

func TestDetect_VulkanSoftwareOnlyAndEnv(t *testing.T) {
	m := linux()
	m.globs[filepath.Join("/usr/share/vulkan/icd.d", "*.json")] = []string{"/usr/share/vulkan/icd.d/lvp_icd.json", "/usr/share/vulkan/icd.d/gfxstream_vk_icd.json"}
	if r := detect(m, Needs{}); r.Has(Vulkan) {
		t.Errorf("software ICDs counted as hardware: %s", r)
	}
	m.env["VK_DRIVER_FILES"] = "/custom/icd.json"
	if r := detect(m, Needs{}); !r.Has(Vulkan) || r.Findings[0].Reason != "VK_DRIVER_FILES is set" {
		t.Errorf("env override: %s", r)
	}
	// XDG directories are searched too
	m = linux()
	m.env["XDG_DATA_DIRS"] = "/opt/share:/var/share"
	m.globs[filepath.Join("/var/share/vulkan/icd.d", "*.json")] = []string{"/var/share/vulkan/icd.d/intel_icd.json"}
	if r := detect(m, Needs{}); !r.Has(Vulkan) || !strings.Contains(r.String(), "intel_icd") {
		t.Errorf("XDG_DATA_DIRS: %s", r)
	}
}

func TestDetect_Darwin(t *testing.T) {
	m := fake{os: "darwin", arch: "arm64", files: map[string]bool{"/opt/rocm": true}, path: map[string]bool{"nvidia-smi": true}, env: map[string]string{"VK_DRIVER_FILES": "x"}, globs: map[string][]string{}}
	r := detect(m, Needs{})
	if accels(r) != "metal cpu" || r.Best().Reason != "Apple silicon" {
		t.Errorf("darwin/arm64 = %s", r)
	}
	m.arch = "amd64"
	if r := detect(m, Needs{}); accels(r) != "cpu" {
		t.Errorf("darwin/amd64 = %s", r)
	}
}

func TestDetect_Windows(t *testing.T) {
	m := fake{os: "windows", arch: "amd64", files: map[string]bool{"/proc/driver/nvidia/version": true}, path: map[string]bool{"nvidia-smi": true}, env: map[string]string{}, globs: map[string][]string{}}
	if r := detect(m, Needs{}); accels(r) != "cuda cpu" || r.Best().Reason != "nvidia-smi on PATH" {
		t.Errorf("windows = %s", r)
	}
}

func TestDetect_RealMachineDoesNotPanic(t *testing.T) {
	r := Detect(Needs{})
	if !r.Has(CPU) || r.Findings[len(r.Findings)-1].Accelerator != CPU {
		t.Errorf("CPU missing or not last: %s", r)
	}
}

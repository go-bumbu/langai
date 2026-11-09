// Package hw reports which accelerators a machine appears to have, so a
// provider can pick the matching runtime build and say why. It looks for
// the driver side (an NVIDIA kernel driver, a ROCm install, a Vulkan
// installable client driver, Apple silicon) and, for CUDA on Linux, for the
// runtime libraries a build links against, since a driver without them
// cannot run it: whether a backend then really registers a device is for
// the runtime to find out after loading.
package hw

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
)

// Accelerator is one way of running a model.
type Accelerator string

const (
	CUDA   Accelerator = "cuda"
	ROCm   Accelerator = "rocm"
	Vulkan Accelerator = "vulkan"
	Metal  Accelerator = "metal"
	// CPU is always available and always last.
	CPU Accelerator = "cpu"
)

// Finding is one accelerator the machine has and the evidence for it.
type Finding struct {
	Accelerator Accelerator
	Reason      string
}

// Report lists what was found, best first; CPU closes the list.
type Report struct {
	Findings []Finding
	// Unusable lists accelerators the machine has a driver for but cannot
	// run a build on, with what is missing: an NVIDIA driver without the
	// CUDA runtime, say. They are not among the Findings.
	Unusable []Finding
}

// Needs is what a provider's builds require of the machine beyond a
// driver.
type Needs struct {
	// CUDARuntime is the major version of the CUDA runtime the provider's
	// CUDA build links against (13 for libcudart.so.13 and libcublas.so.13).
	// CUDA is only reported when the dynamic loader can find those
	// libraries; 0 takes a runtime of any version.
	CUDARuntime int
}

// Best is the first finding, CPU when nothing else was found.
func (r Report) Best() Finding {
	if len(r.Findings) == 0 {
		return Finding{Accelerator: CPU, Reason: "nothing detected"}
	}
	return r.Findings[0]
}

// Has reports whether a was found.
func (r Report) Has(a Accelerator) bool {
	return slices.ContainsFunc(r.Findings, func(f Finding) bool { return f.Accelerator == a })
}

// String lists the findings with their reasons, then what was found but
// cannot be used, for a log line.
func (r Report) String() string {
	parts := make([]string, 0, len(r.Findings)+len(r.Unusable))
	for _, f := range r.Findings {
		parts = append(parts, string(f.Accelerator)+" ("+f.Reason+")")
	}
	for _, f := range r.Unusable {
		parts = append(parts, "no "+string(f.Accelerator)+" ("+f.Reason+")")
	}
	return strings.Join(parts, ", ")
}

// Detect looks at this machine for what a provider with the given needs can
// use.
func Detect(needs Needs) Report {
	return detect(system{}, needs)
}

// probes is what detect reads from the machine; tests substitute a fake.
type probes interface {
	goos() string
	goarch() string
	exists(path string) bool
	lookPath(name string) bool
	getenv(key string) string
	glob(pattern string) []string
	// readFile is the content of a text file, "" when it cannot be read.
	readFile(path string) string
}

// system is the real machine.
type system struct{}

func (system) goos() string            { return runtime.GOOS }
func (system) goarch() string          { return runtime.GOARCH }
func (system) exists(path string) bool { _, err := os.Stat(path); return err == nil }
func (system) lookPath(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}
func (system) getenv(key string) string { return os.Getenv(key) }
func (system) glob(pattern string) []string {
	matches, _ := filepath.Glob(pattern)
	return matches
}
func (system) readFile(path string) string {
	data, err := os.ReadFile(path) //nolint:gosec // the loader's own configuration files, read to learn its search path
	if err != nil {
		return ""
	}
	return string(data)
}

// detect runs the probes in order of preference and appends CPU.
func detect(p probes, needs Needs) Report {
	var r Report
	add := func(a Accelerator, reason string) {
		if reason != "" {
			r.Findings = append(r.Findings, Finding{Accelerator: a, Reason: reason})
		}
	}
	driver, missing := cuda(p, needs)
	if missing != "" {
		r.Unusable = append(r.Unusable, Finding{Accelerator: CUDA, Reason: driver + ", but " + missing})
	} else {
		add(CUDA, driver)
	}
	add(ROCm, rocm(p))
	add(Vulkan, vulkan(p))
	add(Metal, metal(p))
	add(CPU, "always available")
	return r
}

// cuda looks for the NVIDIA driver: the kernel module's version file on
// Linux, or nvidia-smi on PATH anywhere but macOS, which has no CUDA. On
// Linux the driver alone is not enough to run a CUDA build: the CUDA
// runtime is a separate install, and without it the build's backend fails
// to load and the machine silently stays on its CPU although another
// accelerator (Vulkan runs on the same card) would have worked. So the
// runtime's libraries must be where the dynamic loader finds them; missing
// says which are not. Elsewhere only the driver is looked at.
func cuda(p probes, needs Needs) (driver, missing string) {
	switch {
	case p.goos() == "darwin":
		return "", ""
	case p.goos() == "linux" && p.exists("/proc/driver/nvidia/version"):
		driver = "NVIDIA driver at /proc/driver/nvidia/version"
	case p.lookPath("nvidia-smi"):
		driver = "nvidia-smi on PATH"
	default:
		return "", ""
	}
	if p.goos() != "linux" {
		return driver, ""
	}
	dirs := loaderDirs(p)
	var absent []string
	for _, lib := range cudaRuntimeLibs(needs.CUDARuntime) {
		if !findLib(p, dirs, lib) {
			absent = append(absent, lib)
		}
	}
	if len(absent) > 0 {
		return driver, "no " + strings.Join(absent, ", ") + " on the loader's path (the CUDA runtime is not installed)"
	}
	return driver, ""
}

// cudaRuntimeLibs names the CUDA runtime libraries a CUDA build needs to
// load, as glob patterns: the runtime itself and cuBLAS, of the given major
// version or, for 0, of any.
func cudaRuntimeLibs(major int) []string {
	version := "*"
	if major > 0 {
		version = fmt.Sprint(major)
	}
	return []string{"libcudart.so." + version, "libcublas.so." + version}
}

// findLib reports whether a file matching the pattern is in one of dirs.
func findLib(p probes, dirs []string, pattern string) bool {
	for _, dir := range dirs {
		if strings.ContainsAny(pattern, "*?[") {
			if len(p.glob(filepath.Join(dir, pattern))) > 0 {
				return true
			}
		} else if p.exists(filepath.Join(dir, pattern)) {
			return true
		}
	}
	return false
}

// loaderDirs lists where the Linux dynamic loader looks for a library a
// shared object needs: LD_LIBRARY_PATH, the directories of /etc/ld.so.conf
// and the files it includes (where the CUDA packages register theirs), and
// the built-in ones.
func loaderDirs(p probes) []string {
	var dirs []string
	for _, d := range strings.Split(p.getenv("LD_LIBRARY_PATH"), ":") {
		if d != "" {
			dirs = append(dirs, d)
		}
	}
	dirs = append(dirs, confDirs(p, "/etc/ld.so.conf", 0)...)
	dirs = append(dirs, "/lib", "/usr/lib", "/lib64", "/usr/lib64", "/usr/local/lib")
	switch p.goarch() {
	case "amd64":
		dirs = append(dirs, "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu")
	case "arm64":
		dirs = append(dirs, "/lib/aarch64-linux-gnu", "/usr/lib/aarch64-linux-gnu")
	}
	return dirs
}

// confDirs reads the directories a loader configuration file lists,
// following its include lines (glob patterns, relative to the file's
// directory) a few levels deep.
func confDirs(p probes, file string, depth int) []string {
	if depth > 3 {
		return nil
	}
	var dirs []string
	for _, line := range strings.Split(p.readFile(file), "\n") {
		line, _, _ = strings.Cut(line, "#")
		fields := strings.Fields(line)
		switch {
		case len(fields) == 2 && fields[0] == "include":
			pattern := fields[1]
			if !filepath.IsAbs(pattern) {
				pattern = filepath.Join(filepath.Dir(file), pattern)
			}
			for _, included := range p.glob(pattern) {
				dirs = append(dirs, confDirs(p, included, depth+1)...)
			}
		case len(fields) == 1 && filepath.IsAbs(fields[0]):
			dirs = append(dirs, fields[0])
		}
	}
	return dirs
}

// rocm looks for a ROCm install on Linux.
func rocm(p probes) string {
	if p.goos() != "linux" {
		return ""
	}
	if p.exists("/opt/rocm") {
		return "ROCm at /opt/rocm"
	}
	if p.lookPath("rocminfo") {
		return "rocminfo on PATH"
	}
	return ""
}

// softwareICDs are Vulkan drivers that render on the CPU; they make Vulkan
// available without making anything faster.
var softwareICDs = []string{"lvp_icd", "gfxstream"}

// vulkan looks for a hardware Vulkan installable client driver on Linux:
// the loader's environment overrides, or an ICD manifest in the directories
// the loader searches. Software drivers alone (llvmpipe) do not count.
func vulkan(p probes) string {
	if p.goos() != "linux" {
		return ""
	}
	for _, key := range []string{"VK_DRIVER_FILES", "VK_ICD_FILENAMES"} {
		if v := p.getenv(key); v != "" {
			return key + " is set"
		}
	}
	dirs := []string{"/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d", "/usr/local/etc/vulkan/icd.d"}
	for _, key := range []string{"XDG_DATA_DIRS", "XDG_CONFIG_DIRS"} {
		for _, d := range strings.Split(p.getenv(key), ":") {
			if d != "" {
				dirs = append(dirs, filepath.Join(d, "vulkan", "icd.d"))
			}
		}
	}
	var hardware []string
	for _, d := range dirs {
		for _, m := range p.glob(filepath.Join(d, "*.json")) {
			name := strings.TrimSuffix(filepath.Base(m), ".json")
			if !slices.ContainsFunc(softwareICDs, func(s string) bool { return strings.HasPrefix(name, s) }) {
				hardware = append(hardware, name)
			}
		}
	}
	if len(hardware) == 0 {
		return ""
	}
	slices.Sort(hardware)
	return fmt.Sprintf("Vulkan ICD %s", strings.Join(slices.Compact(hardware), ", "))
}

// metal is Apple silicon.
func metal(p probes) string {
	if p.goos() == "darwin" && p.goarch() == "arm64" {
		return "Apple silicon"
	}
	return ""
}

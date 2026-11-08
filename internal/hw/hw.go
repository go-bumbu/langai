// Package hw reports which accelerators a machine appears to have, so a
// provider can pick the matching runtime build and say why. It only looks
// for the driver side (an NVIDIA kernel driver, a ROCm install, a Vulkan
// installable client driver, Apple silicon): whether a backend then really
// registers a device is for the runtime to find out after loading.
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

// String lists the findings with their reasons, for a log line.
func (r Report) String() string {
	parts := make([]string, len(r.Findings))
	for i, f := range r.Findings {
		parts[i] = string(f.Accelerator) + " (" + f.Reason + ")"
	}
	return strings.Join(parts, ", ")
}

// Detect looks at this machine.
func Detect() Report {
	return detect(system{})
}

// probes is what detect reads from the machine; tests substitute a fake.
type probes interface {
	goos() string
	goarch() string
	exists(path string) bool
	lookPath(name string) bool
	getenv(key string) string
	glob(pattern string) []string
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

// detect runs the probes in order of preference and appends CPU.
func detect(p probes) Report {
	var r Report
	add := func(a Accelerator, reason string) {
		if reason != "" {
			r.Findings = append(r.Findings, Finding{Accelerator: a, Reason: reason})
		}
	}
	add(CUDA, cuda(p))
	add(ROCm, rocm(p))
	add(Vulkan, vulkan(p))
	add(Metal, metal(p))
	add(CPU, "always available")
	return r
}

// cuda looks for the NVIDIA driver: the kernel module's version file on
// Linux, or nvidia-smi on PATH anywhere but macOS, which has no CUDA.
func cuda(p probes) string {
	if p.goos() == "darwin" {
		return ""
	}
	if p.goos() == "linux" && p.exists("/proc/driver/nvidia/version") {
		return "NVIDIA driver at /proc/driver/nvidia/version"
	}
	if p.lookPath("nvidia-smi") {
		return "nvidia-smi on PATH"
	}
	return ""
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

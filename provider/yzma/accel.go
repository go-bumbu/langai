package yzma

import (
	"fmt"
	"slices"
	"strings"

	"github.com/go-bumbu/langai/internal/hw"
)

// Accel is how a model is run: on the CPU, or offloaded to a GPU through one
// of llama.cpp's backends. llama.cpp is one CPU library plus one shared
// library per GPU backend, and loading registers every backend whose driver
// answers, so a managed LibDir gets the build that carries the backend the
// machine can use (see internal/hw) and, after loading, the model goes to
// the GPU device that registered. Nothing else changes: the same model file,
// the same vectors.
type Accel string

const (
	// AccelAuto picks from the hardware: the best accelerator internal/hw
	// finds that has a pinned build, the CPU when none has. CUDA counts only
	// with the CUDA 13 runtime where the loader finds it: an NVIDIA driver
	// without it gets the Vulkan build. When the build that was picked
	// registers no GPU after all, the models run on the CPU and a warning
	// says so.
	AccelAuto Accel = ""
	// AccelCPU never offloads and never installs a GPU build.
	AccelCPU Accel = "cpu"
	// AccelVulkan requires the Vulkan backend: any GPU with a Vulkan driver.
	AccelVulkan Accel = "vulkan"
	// AccelCUDA requires the CUDA backend: an NVIDIA GPU with the CUDA 13
	// runtime installed on the host.
	AccelCUDA Accel = "cuda"
	// AccelMetal requires the Metal backend: Apple silicon.
	AccelMetal Accel = "metal"
)

// accels lists the accepted values, for error messages.
var accels = []Accel{AccelAuto, AccelCPU, AccelVulkan, AccelCUDA, AccelMetal}

// normalized returns a, or an error naming the accepted values.
func (a Accel) normalized() (Accel, error) {
	if slices.Contains(accels, a) {
		return a, nil
	}
	return "", fmt.Errorf("unknown Accel %q: use one of auto (\"\"), %s, %s, %s, %s", a, AccelCPU, AccelVulkan, AccelCUDA, AccelMetal)
}

// build is the name of a llama.cpp build in libAssets: "cpu", "vulkan",
// "cuda" or "metal" (the macOS build, which carries Metal and runs the CPU
// too).
type build = string

// buildsFor maps an accelerator to the builds that serve it, best first.
// The CPU is served by the plain build, or on macOS by the only build there
// is; a GPU accelerator only by its own build.
func buildsFor(a hw.Accelerator) []build {
	switch a {
	case hw.CUDA:
		return []build{"cuda"}
	case hw.Vulkan:
		return []build{"vulkan"}
	case hw.Metal:
		return []build{"metal"}
	case hw.CPU:
		return []build{"cpu", "metal"}
	}
	return nil // ROCm has no pinned build
}

// chooseBackend picks the build a managed LibDir gets on plat ("linux/amd64")
// for accel: with AccelAuto the first finding of report that has a pinned
// build; with an explicit accelerator its build, which must be pinned.
func chooseBackend(accel Accel, report hw.Report, plat string) (build, error) {
	pinned := func(b build) bool {
		_, ok := libAssets[plat+"/"+b]
		return ok
	}
	if accel == AccelAuto {
		for _, f := range report.Findings {
			for _, b := range buildsFor(f.Accelerator) {
				if pinned(b) {
					return b, nil
				}
			}
		}
		return "", fmt.Errorf("no pinned llama.cpp build for %s: use an unmanaged LibDir populated with `yzma install`", plat)
	}
	for _, b := range buildsFor(hw.Accelerator(accel)) {
		if pinned(b) {
			return b, nil
		}
	}
	return "", fmt.Errorf("no pinned llama.cpp build with %s for %s: use an unmanaged LibDir populated with `yzma install`, or Accel %q", accel, plat, AccelCPU)
}

// device is a ggml device as the loaded libraries report it.
type device struct {
	name        string // "Vulkan0", "CUDA0", "CPU"
	backend     string // the backend's registered name: "Vulkan", "CUDA", "Metal", "CPU"
	description string // "AMD Radeon 890M Graphics"
	gpu         bool   // a GPU (discrete or integrated), not the CPU or an accelerator
}

// software marks Vulkan devices that are the CPU in disguise: offloading to
// them is slower than not offloading.
func (d device) software() bool {
	desc := strings.ToLower(d.description)
	return strings.Contains(desc, "llvmpipe") || strings.Contains(desc, "lavapipe") || strings.Contains(desc, "swiftshader")
}

// backendNames maps an Accel to the name its ggml backend registers under.
var backendNames = map[Accel]string{AccelVulkan: "Vulkan", AccelCUDA: "CUDA", AccelMetal: "Metal"}

// allLayers is the NGpuLayers that offloads every layer: llama.cpp caps it
// at the model's count.
const allLayers int32 = 999

// resolveAccel decides, from the devices the loaded libraries registered,
// how many layers go to a GPU and which device that is: gpuLayers 0 means
// the CPU. With AccelAuto the first hardware GPU wins; an explicit
// accelerator needs a device of its backend, and fails when none
// registered. layers > 0 limits the offload for auto and explicit
// accelerators. used is the text for the log line.
func resolveAccel(accel Accel, layers int, devices []device) (gpuLayers int32, used string, err error) {
	if accel == AccelCPU {
		return 0, "cpu", nil
	}
	var pick *device
	for i := range devices {
		d := &devices[i]
		if !d.gpu || d.software() {
			continue
		}
		if accel != AccelAuto && d.backend != backendNames[accel] {
			continue
		}
		pick = d
		break
	}
	if pick == nil {
		if accel == AccelAuto {
			return 0, "cpu", nil
		}
		return 0, "", fmt.Errorf("Accel %q: no %s device registered (devices: %s): install the driver, or use a managed LibDir to get the %s build", accel, backendNames[accel], deviceList(devices), accel)
	}
	gpuLayers = allLayers
	if layers > 0 {
		gpuLayers = int32(min(layers, int(allLayers)))
	}
	return gpuLayers, pick.name + ": " + pick.description, nil
}

// deviceList names the devices for an error message.
func deviceList(devices []device) string {
	if len(devices) == 0 {
		return "none"
	}
	names := make([]string, len(devices))
	for i, d := range devices {
		names[i] = d.name
	}
	return strings.Join(names, ", ")
}

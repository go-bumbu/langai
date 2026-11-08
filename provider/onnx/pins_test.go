package onnx

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// bindingModule is the module whose release LibVersion in fetch.go belongs to.
const bindingModule = "github.com/yalue/onnxruntime_go"

// TestPinsMatchBindingVersion is the compat lock: onnxruntime_go's headers
// target one ONNX Runtime release, so the pinned build in fetch.go must be
// refreshed together with the binding version in go.mod (take the release
// its README names, then update LibVersion, libMajorMinor, libAssets and
// pinnedBinding). It reads go.mod itself: test binaries built by Go 1.26
// carry no module dependencies in debug.ReadBuildInfo.
func TestPinsMatchBindingVersion(t *testing.T) {
	// go test runs in the package directory, two levels below the module root
	gomod := filepath.Join("..", "..", "go.mod")
	data, err := os.ReadFile(gomod)
	if err != nil {
		t.Fatalf("read the module's go.mod: %v", err)
	}
	version, ok := requiredVersion(string(data), bindingModule)
	if !ok {
		t.Fatalf("%s does not require %s", gomod, bindingModule)
	}
	if version != pinnedBinding {
		t.Fatalf("go.mod has onnxruntime_go %s but the pins in fetch.go were taken from %s: refresh LibVersion, libMajorMinor, libAssets and pinnedBinding together", version, pinnedBinding)
	}
	if !strings.HasPrefix(LibVersion, libMajorMinor+".") {
		t.Fatalf("LibVersion %s is not a patch of libMajorMinor %s", LibVersion, libMajorMinor)
	}
}

// requiredVersion returns the version gomod requires for the module path,
// from a single-line `require path version` directive or from a line inside
// a `require ( ... )` block; ok is false when gomod does not require it.
func requiredVersion(gomod, path string) (version string, ok bool) {
	inRequire := false
	for _, line := range strings.Split(gomod, "\n") {
		line, _, _ = strings.Cut(line, "//") // drops "// indirect"
		f := strings.Fields(line)
		switch {
		case len(f) == 0:
		case inRequire && f[0] == ")":
			inRequire = false
		case inRequire && len(f) == 2 && f[0] == path:
			return f[1], true
		case len(f) == 2 && f[0] == "require" && f[1] == "(":
			inRequire = true
		case len(f) == 3 && f[0] == "require" && f[1] == path:
			return f[2], true
		}
	}
	return "", false
}

package yzma

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// yzmaModule is the module whose release the pins in fetch.go belong to.
const yzmaModule = "github.com/hybridgroup/yzma"

// TestPinsMatchYzmaVersion is the compat lock: yzma's Go bindings mirror one
// llama.cpp ABI, so the pinned build in fetch.go must be refreshed together
// with the yzma version in go.mod (take the new values from that release's
// download.DefaultVersion manifest, then update pinnedYzma). It reads go.mod
// itself: test binaries built by Go 1.26 carry no module dependencies in
// debug.ReadBuildInfo, which would leave the lock nothing to compare.
func TestPinsMatchYzmaVersion(t *testing.T) {
	// go test runs in the package directory, two levels below the module root
	gomod := filepath.Join("..", "..", "go.mod")
	data, err := os.ReadFile(gomod)
	if err != nil {
		t.Fatalf("read the module's go.mod: %v", err)
	}
	version, ok := requiredVersion(string(data), yzmaModule)
	if !ok {
		t.Fatalf("%s does not require %s", gomod, yzmaModule)
	}
	if version != pinnedYzma {
		t.Fatalf("go.mod has yzma %s but the pins in fetch.go were taken from %s: refresh LibTag, libVersion, libAssets and pinnedYzma together", version, pinnedYzma)
	}
}

// requiredVersion returns the version gomod requires for the module path,
// from a single-line `require path version` directive or from a line inside
// a `require ( ... )` block; ok is false when gomod does not require it.
// Other blocks (replace, exclude) are not requirements and are skipped.
func requiredVersion(gomod, path string) (version string, ok bool) {
	inRequire := false
	for _, line := range strings.Split(gomod, "\n") {
		line, _, _ = strings.Cut(line, "//") // drops "// indirect"
		f := strings.Fields(line)
		switch {
		case len(f) == 0: // blank or comment-only line
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

// TestRequiredVersion covers both forms go.mod writes a requirement in; the
// repository's go.mod only exercises the block form.
func TestRequiredVersion(t *testing.T) {
	cases := []struct {
		name, gomod, want string
		ok                bool
	}{
		{"block", "require (\n\tgithub.com/a/b v1.0.0\n\tgithub.com/hybridgroup/yzma v1.29.1\n)\n", "v1.29.1", true},
		{"single line", "go 1.26.0\n\nrequire github.com/hybridgroup/yzma v1.29.1\n", "v1.29.1", true},
		{"indirect", "require (\n\tgithub.com/hybridgroup/yzma v1.29.1 // indirect\n)\n", "v1.29.1", true},
		{"other module", "require (\n\tgithub.com/hybridgroup/yzma-extra v1.0.0\n)\n", "", false},
		{"replace and exclude only", "replace (\n\tgithub.com/hybridgroup/yzma v1.28.0 => ../yzma\n)\n\nexclude (\n\tgithub.com/hybridgroup/yzma v1.27.0\n)\n\nexclude github.com/hybridgroup/yzma v1.26.0\n", "", false},
	}
	for _, c := range cases {
		got, ok := requiredVersion(c.gomod, yzmaModule)
		if got != c.want || ok != c.ok {
			t.Errorf("%s: requiredVersion = %q, %v; want %q, %v", c.name, got, ok, c.want, c.ok)
		}
	}
}

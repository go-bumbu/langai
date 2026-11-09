//go:build !unix

package fetch

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestPartialPath_IsThisProcessesOwn: without file locks two processes
// after one file would write the same partial file through each other, so
// each writes one named after its process id.
func TestPartialPath_IsThisProcessesOwn(t *testing.T) {
	dest := filepath.Join("models", "m.gguf")
	got := partialPath(dest)
	if want := dest + partialSuffix + "-" + strconv.Itoa(os.Getpid()); got != want {
		t.Errorf("partialPath = %q, want %q", got, want)
	}
	if !strings.HasPrefix(got, dest+partialSuffix) || got == dest+partialSuffix {
		t.Errorf("partialPath = %q, want a name of this process's own under %q", got, dest+partialSuffix)
	}
}

// TestResumable_NeverWithoutFileLocks: a partial file no other process will
// ever pick up is not kept after a failure.
func TestResumable_NeverWithoutFileLocks(t *testing.T) {
	part, err := os.CreateTemp(t.TempDir(), "part")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = part.Close() }()
	if _, err := part.WriteString("half"); err != nil {
		t.Fatal(err)
	}
	if resumable(part, Asset{SHA256: strings.Repeat("0", 64)}, errors.New("connection reset")) {
		t.Error("a partial file was kept for a later process that cannot find it")
	}
}

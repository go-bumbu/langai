//go:build unix

package yzma

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// lockAsAnotherProcess takes the install lock of libDir the way another
// process holds it while it downloads a build: an exclusive flock on
// "<libDir>.lock" through an open file of its own. It is released when the
// test ends.
func lockAsAnotherProcess(t *testing.T, libDir string) {
	t.Helper()
	f, err := os.OpenFile(libDir+".lock", os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Close() })
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		t.Fatal(err)
	}
}

// TestEnsureLibs_AProcessThatLoadedDoesNotWaitForTheLock: an open in a
// process that already runs llama.cpp from the managed directory returns at
// once while another process holds the directory's lock; a process that has
// not loaded waits for it, and gives up with its context's error.
func TestEnsureLibs_AProcessThatLoadedDoesNotWaitForTheLock(t *testing.T) {
	for _, tc := range []struct {
		name    string
		loaded  bool
		wantErr error
	}{
		{"already loaded from the directory", true, nil},
		{"nothing loaded yet", false, context.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			libDir := filepath.Join(t.TempDir(), "llama-cpp", LibTag)
			if err := os.MkdirAll(libDir, 0o750); err != nil {
				t.Fatal(err)
			}
			for name, content := range map[string]string{libFile(): "llama", libRecordFile: `{"tag":"` + LibTag + `","backend":"cpu"}`} {
				if err := os.WriteFile(filepath.Join(libDir, name), []byte(content), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			loader, f := &runtimeLoader{}, newFakeAPI()
			if tc.loaded {
				if err := loader.load(f, libDir, nil); err != nil {
					t.Fatal(err)
				}
			}
			lockAsAnotherProcess(t, libDir)
			ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
			defer cancel()
			start := time.Now()
			err := ensureLibs(ctx, libsWant{plat: "test/arch", dir: libDir, managed: true, build: "cpu", explicit: true,
				loadedDir: loader.loadedDir,
				use:       func() error { return loader.load(f, libDir, nil) }}, discardLogger())
			if !errors.Is(err, tc.wantErr) || (tc.wantErr == nil && err != nil) {
				t.Errorf("ensureLibs = %v, want %v", err, tc.wantErr)
			}
			if tc.loaded && time.Since(start) > 200*time.Millisecond {
				t.Errorf("the open took %v: it waited for the lock", time.Since(start))
			}
			if tc.loaded && f.count("load") != 1 {
				t.Errorf("the libraries were loaded %d times", f.count("load"))
			}
		})
	}
}

//go:build unix

package onnx

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// TestEnsureLibs_AProcessThatLoadedDoesNotWaitForTheLock: an open in a
// process that already runs ONNX Runtime from the managed directory returns
// at once while another process holds the directory's lock (an exclusive
// flock on "<LibDir>.lock" through an open file of its own, as during a
// download); a process that has not loaded waits for it, and gives up with
// its context's error.
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
			libDir := filepath.Join(t.TempDir(), "onnxruntime", LibVersion)
			if err := os.MkdirAll(libDir, 0o750); err != nil {
				t.Fatal(err)
			}
			libPath := filepath.Join(libDir, libFile())
			if err := os.WriteFile(libPath, []byte("not really"), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := writeInstallRecord(libDir, variantCPU); err != nil {
				t.Fatal(err)
			}
			loader, f := &runtimeLoader{}, &fakeAPI{}
			if tc.loaded {
				if err := loader.load(f, libPath, discardLogger()); err != nil {
					t.Fatal(err)
				}
			}
			other, err := os.OpenFile(libDir+".lock", os.O_RDWR|os.O_CREATE, 0o600)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = other.Close() }()
			if err := syscall.Flock(int(other.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
			defer cancel()
			start := time.Now()
			err = ensureLibs(ctx, libsWant{plat: "test/arch", dir: libDir, managed: true, variant: variantCPU,
				loadedDir: loader.loadedDir,
				use:       func() error { return loader.load(f, libPath, discardLogger()) }}, discardLogger())
			if !errors.Is(err, tc.wantErr) || (tc.wantErr == nil && err != nil) {
				t.Errorf("ensureLibs = %v, want %v", err, tc.wantErr)
			}
			if tc.loaded && time.Since(start) > 200*time.Millisecond {
				t.Errorf("the open took %v: it waited for the lock", time.Since(start))
			}
			if tc.loaded && len(f.loads) != 1 {
				t.Errorf("the library was loaded %d times", len(f.loads))
			}
		})
	}
}

//go:build unix

package fetch

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestInstallDir_AProcessThatLoadedDoesNotWait: while another process holds
// the directory's lock, downloading a build into it, a process that already
// runs the libraries from that directory must not queue behind the download
// at its next open: it has nothing to install and nothing to load. Whoever
// has not loaded waits its turn, and gives up with its context's error.
func TestInstallDir_AProcessThatLoadedDoesNotWait(t *testing.T) {
	for _, tc := range []struct {
		name string
		// queued says that another goroutine of this process waits for the
		// lock too, holding the directory's turn inside the process
		queued  bool
		loaded  func() bool
		ready   bool
		wantErr error
		used    int
	}{
		{name: "loaded and ready", loaded: func() bool { return true }, ready: true, used: 1},
		{name: "loaded and ready, behind a goroutine that waits", queued: true, loaded: func() bool { return true }, ready: true, used: 1},
		{name: "loaded, but the directory is not ready", loaded: func() bool { return true }, wantErr: context.DeadlineExceeded},
		{name: "ready, but not loaded", loaded: func() bool { return false }, ready: true, wantErr: context.DeadlineExceeded},
		{name: "ready, not loaded, behind a goroutine that waits", queued: true, loaded: func() bool { return false }, ready: true, wantErr: context.DeadlineExceeded},
		{name: "ready, and nobody says whether it is loaded", ready: true, wantErr: context.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := filepath.Join(realTemp(t), "libs")
			// another process: the file lock, held through another open file
			other, err := openLocked(context.Background(), dir+lockSuffix)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = other.Close() }()
			if tc.queued {
				release, err := inProcess.acquire(context.Background(), dir)
				if err != nil {
					t.Fatal(err)
				}
				defer release()
			}

			used := 0
			in := DirInstall{
				Dir: dir, What: "libx 1.0", RecordFile: testRecord,
				Archive: Asset{URL: "http://127.0.0.1:1/unreachable"},
				Loaded:  tc.loaded,
				Ready:   func() (bool, error) { return tc.ready, nil },
				Use:     func() error { used++; return nil },
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*lockPoll)
			defer cancel()
			start := time.Now()
			installed, err := dl(cap).InstallDir(ctx, in)
			if installed || !errors.Is(err, tc.wantErr) || (tc.wantErr == nil && err != nil) {
				t.Errorf("InstallDir = %v, %v; want no install and %v", installed, err, tc.wantErr)
			}
			if used != tc.used {
				t.Errorf("Use ran %d times, want %d", used, tc.used)
			}
			if tc.wantErr == nil && time.Since(start) > 2*lockPoll {
				t.Errorf("the call took %v: it waited for the lock", time.Since(start))
			}
		})
	}
}

// TestTryLock_ExcludesAnotherOpenFile: a flock belongs to the open file, so
// a second open of the same path, which is what another process has, is
// refused until the first is closed.
func TestTryLock_ExcludesAnotherOpenFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dir.lock")
	first, err := openLocked(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	second, err := os.OpenFile(path, os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = second.Close() }()
	if ok, err := tryLock(second); ok || err != nil {
		t.Fatalf("a held lock was granted again: %v, %v", ok, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*lockPoll)
	defer cancel()
	if _, err := openLocked(ctx, path); err == nil {
		t.Fatal("openLocked got a held lock")
	}
	_ = first.Close()
	if ok, err := tryLock(second); !ok || err != nil {
		t.Fatalf("a released lock was refused: %v, %v", ok, err)
	}
}

// TestOpenLocked_FollowsThePath: a download renames its locked partial file
// into place when it is done. Whoever waited for that lock must not be
// handed the renamed file, which is now the finished download, but the file
// at the path it asked for.
func TestOpenLocked_FollowsThePath(t *testing.T) {
	dir := t.TempDir()
	partial, dest := filepath.Join(dir, "model.partial"), filepath.Join(dir, "model")
	holder, err := openLocked(context.Background(), partial)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := holder.WriteString("finished"); err != nil {
		t.Fatal(err)
	}
	got := make(chan *os.File, 1)
	go func() {
		f, err := openLocked(context.Background(), partial)
		if err != nil {
			t.Error(err)
		}
		got <- f
	}()
	time.Sleep(2 * lockPoll) // the waiter has opened the file and polls for its lock
	if err := os.Rename(partial, dest); err != nil {
		t.Fatal(err)
	}
	_ = holder.Close()
	select {
	case f := <-got:
		if f == nil {
			return
		}
		defer func() { _ = f.Close() }()
		info, err := f.Stat()
		if err != nil || info.Size() != 0 {
			t.Errorf("the waiter got a file of %d bytes (%v), want the new, empty one at the path", info.Size(), err)
		}
		at, err := os.Stat(partial)
		if err != nil || !os.SameFile(info, at) {
			t.Errorf("the waiter's file is not the one at %s: %v", partial, err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the waiter never got the lock")
	}
	if b, _ := os.ReadFile(dest); string(b) != "finished" {
		t.Errorf("the renamed file holds %q", b)
	}
}

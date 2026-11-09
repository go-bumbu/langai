//go:build unix

package fetch

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

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

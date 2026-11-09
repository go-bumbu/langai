package fetch

import (
	"context"
	"os"
	"sync"
	"time"
)

// lockPoll is how often a waiter asks again for a lock another process
// holds.
const lockPoll = 100 * time.Millisecond

// keyedLock serializes work per key inside this process. The file locks do
// the same between processes, and on a platform without them (see
// lock_other.go) this is what keeps two goroutines apart.
type keyedLock struct {
	mu   sync.Mutex
	held map[string]chan struct{}
}

// inProcess holds one key per download destination or install directory.
var inProcess = keyedLock{held: map[string]chan struct{}{}}

// acquire waits for key, until ctx ends, and returns how to give it back.
func (k *keyedLock) acquire(ctx context.Context, key string) (release func(), err error) {
	for {
		k.mu.Lock()
		busy, taken := k.held[key]
		if !taken {
			done := make(chan struct{})
			k.held[key] = done
			k.mu.Unlock()
			return func() {
				k.mu.Lock()
				delete(k.held, key)
				k.mu.Unlock()
				close(done)
			}, nil
		}
		k.mu.Unlock()
		select {
		case <-busy:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// openLocked opens the file at path, creating it when missing, and takes an
// exclusive lock on it that other processes honour, waiting until ctx ends.
// Closing the file releases the lock. The file returned is the one at path:
// one that was renamed away or removed while this call waited (its holder
// finished) is given up for the file now there.
func openLocked(ctx context.Context, path string) (*os.File, error) {
	for {
		f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600) //nolint:gosec // path is a lock or partial file this package names next to the caller's own destination
		if err != nil {
			return nil, err
		}
		if err := waitLock(ctx, f); err != nil {
			_ = f.Close()
			return nil, err
		}
		held, heldErr := f.Stat()
		now, nowErr := os.Stat(path)
		if heldErr == nil && nowErr == nil && os.SameFile(held, now) {
			return f, nil
		}
		_ = f.Close()
	}
}

// waitLock polls for the lock on f until it is granted or ctx ends.
func waitLock(ctx context.Context, f *os.File) error {
	for {
		ok, err := tryLock(f)
		if err != nil {
			return err
		}
		if ok {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(lockPoll):
		}
	}
}

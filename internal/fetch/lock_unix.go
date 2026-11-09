//go:build unix

package fetch

import (
	"errors"
	"os"
	"syscall"
)

// crossProcessLocks says whether tryLock keeps processes apart: here it
// does, by flock.
const crossProcessLocks = true

// partialPath names the file a download of dest is written to: one name
// for every process, since the lock on it makes them take turns, and that
// is what lets the next process pick up what a killed one left.
func partialPath(dest string) string { return dest + partialSuffix }

// tryLock asks for an exclusive flock on f without waiting; false means
// another open file, in this process or another, holds it.
func tryLock(f *os.File) (bool, error) {
	err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	switch {
	case err == nil:
		return true, nil
	case errors.Is(err, syscall.EWOULDBLOCK), errors.Is(err, syscall.EINTR):
		return false, nil
	}
	return false, err
}

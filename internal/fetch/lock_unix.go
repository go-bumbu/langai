//go:build unix

package fetch

import (
	"errors"
	"os"
	"syscall"
)

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

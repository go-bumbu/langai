//go:build !unix

package fetch

import "os"

// tryLock grants every lock where there is no flock: goroutines of one
// process are kept apart by keyedLock, processes are not.
func tryLock(*os.File) (bool, error) { return true, nil }

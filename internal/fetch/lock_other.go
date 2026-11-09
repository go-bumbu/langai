//go:build !unix

package fetch

import (
	"os"
	"strconv"
)

// crossProcessLocks says whether tryLock keeps processes apart. Here it
// does not: there is no flock, and no pinned build runs on such a system
// today, so what follows is what keeps two processes from damaging each
// other's work, not a full replacement.
const crossProcessLocks = false

// tryLock grants every lock where there is no flock: goroutines of one
// process are kept apart by keyedLock, processes are not.
func tryLock(*os.File) (bool, error) { return true, nil }

// partialPath names the file a download of dest is written to. Without a
// lock between processes it is this process's own, named after its process
// id: two processes after one file each write a whole copy and the last
// rename wins, instead of both writing one file through each other. The
// price is that no process can pick up what another one left, so a failed
// download keeps nothing here (see resumable), and the partial file of a
// process that was killed stays until someone removes it. An install into
// a directory (InstallDir) is not serialized between processes either: two
// that start together on an empty directory can fail each other, and the
// next open finds it installed or installs it.
func partialPath(dest string) string {
	return dest + partialSuffix + "-" + strconv.Itoa(os.Getpid())
}

package fetch

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const (
	// lockSuffix, appended to an install directory, names the file whose lock
	// serializes installs into it. The file is never removed: a lock that is
	// deleted while someone waits for it protects nothing.
	lockSuffix = ".lock"
	// archiveSuffix names the downloaded tarball, stagingSuffix the directory
	// it is unpacked into, and retiredSuffix where the directory being
	// replaced waits to be deleted. All sit next to the install directory, on
	// its file system, so the last step is a rename. The directory's parent
	// is the caller's and may hold anything (with a library directory of
	// "~/llama" it is the home directory), so the names say whose they are:
	// nobody keeps a backup called "llama.langai-retired". A name alone is
	// still no licence to delete, see clearLeftover.
	archiveSuffix = ".langai-download"
	stagingSuffix = ".langai-staging"
	retiredSuffix = ".langai-retired"
	// markerFile is the first thing written into a staging directory and the
	// last thing removed from it before it gets its final name: a directory
	// with it in is one an install of this package was filling.
	markerFile = ".langai-install"
)

// DirInstall describes the install of a release tarball's libraries into a
// directory this package's caller owns. What differs between providers is
// here: which archive, how its layout is unpacked, what the record of the
// install says, and when the directory is good as it is.
type DirInstall struct {
	// Dir is the directory to fill.
	Dir string
	// What names the install in the download's log line and in errors
	// ("llama.cpp b11429").
	What string
	// Archive is the gzipped release tarball.
	Archive Asset
	// Extract unpacks the verified archive into dir, a new directory that
	// holds nothing but this package's marker file (".langai-install").
	Extract func(archive, dir string) error
	// RecordFile is the name of the file written next to the libraries with
	// Record as its content, so a later run can tell what Dir holds. Its
	// presence is also what marks a directory as installed by the caller: a
	// directory with other content and no such file is never replaced.
	RecordFile string
	Record     []byte
	// Ready reports whether Dir already holds what is wanted. It is asked
	// once the directory's lock is held, so an install that another
	// goroutine or process finished in the meantime is found instead of
	// repeated. An error stops everything: the caller refusing to replace a
	// directory it is using, say. Nil means never ready.
	Ready func() (bool, error)
	// Use, when set, runs once Dir holds what is wanted, with the lock still
	// held: loading the libraries there keeps another process from replacing
	// them halfway through the load.
	Use func() error
}

// InstallDir makes in.Dir hold what in describes and runs in.Use on it.
// Installs into one directory are serialized, between goroutines and
// (where the platform has file locks) between processes, by a lock file
// next to the directory, and whether an install is needed is decided under
// that lock. The download and the extraction happen next to the final
// directory, which is renamed into place at the end: an interrupted install
// never leaves a half-filled in.Dir. A directory that holds anything this
// package's caller did not install is not touched; that is an error.
//
// Nothing is ever removed for its name alone. A call that finds the
// directory ready writes nothing but the lock file. One that installs works
// under "<Dir>.langai-download", "<Dir>.langai-staging" and
// "<Dir>.langai-retired", and clears what an install that was killed left
// under the last two only when it is recognizably that (see clearLeftover);
// anything else under those names stops the install with an error.
//
// in.Dir is resolved first (see ResolveDir), so a directory that is a
// symbolic link is installed where the link points, next to which the lock
// and the staging then are, and the link stays a link. A caller that names
// the directory elsewhere (in Ready, in Use, to compare it with the one it
// loaded from) resolves it the same way.
//
// installed reports whether the directory was filled by this call.
func (d Downloader) InstallDir(ctx context.Context, in DirInstall) (installed bool, err error) {
	if in.Dir, err = ResolveDir(in.Dir); err != nil {
		return false, err
	}
	release, err := inProcess.acquire(ctx, in.Dir)
	if err != nil {
		return false, err
	}
	defer release()
	lock, lockErr := lockDir(ctx, in.Dir)
	if lockErr == nil {
		defer func() { _ = lock.Close() }()
	}
	ready, err := in.ready()
	if err != nil {
		return false, err
	}
	switch {
	case ready:
		// a directory that cannot be locked (a read-only image, say) is fine
		// as long as there is nothing to do to it
	case lockErr != nil:
		return false, lockErr
	default:
		if err := d.install(ctx, in); err != nil {
			return false, err
		}
		installed = true
	}
	if in.Use != nil {
		if err := in.Use(); err != nil {
			return installed, err
		}
	}
	return installed, nil
}

// maxLinkHops is how many symbolic links ResolveDir follows by hand, one
// pointing at the next, before it takes them for a loop.
const maxLinkHops = 40

// ResolveDir returns dir as an absolute path with its symbolic links
// resolved, as far as the path exists: the part that is not there yet is
// kept as it is, behind the real path of the part that is, and a link whose
// target is not there yet is followed to that target. It is the name to
// install into and to compare directories by. A managed directory that is
// a link (to another disk, say) must be filled where it points: the renames
// of an install would otherwise replace the link by a directory and leave
// the old files behind it; and one directory reached under two names must
// compare equal to itself.
func ResolveDir(dir string) (string, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	return resolve(abs, 0)
}

// resolve is ResolveDir for an absolute path; hops counts the links
// followed by hand so far.
func resolve(path string, hops int) (string, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err == nil {
		return resolved, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	parent := filepath.Dir(path)
	if parent == path {
		return path, nil
	}
	realParent, err := resolve(parent, hops)
	if err != nil {
		return "", err
	}
	here := filepath.Join(realParent, filepath.Base(path))
	info, err := os.Lstat(here)
	if err != nil || info.Mode()&os.ModeSymlink == 0 {
		return here, nil // not there, which is fine: the rest of the path is kept
	}
	// a link whose target does not exist
	if hops == maxLinkHops {
		return "", fmt.Errorf("%s: too many levels of symbolic links", path)
	}
	target, err := os.Readlink(here)
	if err != nil {
		return "", err
	}
	if !filepath.IsAbs(target) {
		target = filepath.Join(realParent, target)
	}
	return resolve(target, hops+1)
}

// ready asks in.Ready, which may be nil.
func (in DirInstall) ready() (bool, error) {
	if in.Ready == nil {
		return false, nil
	}
	return in.Ready()
}

// lockDir takes the lock that serializes installs into dir, creating dir's
// parent and the lock file as needed. It removes nothing.
func lockDir(ctx context.Context, dir string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(dir), 0o750); err != nil {
		return nil, err
	}
	return openLocked(ctx, dir+lockSuffix)
}

// install fills in.Dir from in.Archive. The caller holds the directory's
// lock.
func (d Downloader) install(ctx context.Context, in DirInstall) error {
	// before the download, not after it: a refusal should not cost one
	if err := replaceable(in.Dir, in.RecordFile); err != nil {
		return err
	}
	staging, retired := in.Dir+stagingSuffix, in.Dir+retiredSuffix
	for _, left := range []string{staging, retired} {
		if err := clearLeftover(left, in.RecordFile); err != nil {
			return err
		}
	}
	archive := in.Dir + archiveSuffix
	if err := d.File(ctx, in.What, in.Archive, archive); err != nil {
		return fmt.Errorf("download %s: %w", in.What, err)
	}
	defer func() { _ = os.Remove(archive) }()
	if err := os.Mkdir(staging, 0o750); err != nil {
		return err
	}
	// from here on staging is this call's: made by the Mkdir above, which
	// fails on a directory that is already there
	defer func() { _ = os.RemoveAll(staging) }()
	marker := filepath.Join(staging, markerFile)
	if err := writeSynced(marker, nil); err != nil {
		return err
	}
	if err := in.Extract(archive, staging); err != nil {
		return fmt.Errorf("extract %s: %w", in.What, err)
	}
	if err := writeSynced(filepath.Join(staging, in.RecordFile), in.Record); err != nil {
		return err
	}
	// the record marks the directory from now on, so the marker can go
	// before the directory gets its final name
	if err := os.Remove(marker); err != nil {
		return err
	}
	syncDir(staging)
	return swap(staging, retired, in.Dir)
}

// clearLeftover removes what an install that was killed left at path, the
// staging or the retired sibling of an install directory, and returns nil
// when nothing is there afterwards. It removes only what is recognizably
// such a leftover: a directory that carries the marker (a staging directory
// being filled) or the install's record (a filled staging directory, or the
// installed directory that was being replaced), or an empty one. Anything
// else at that name belongs to someone else, and is an error that leaves it
// as it is.
func clearLeftover(path, record string) error {
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.IsDir() {
		for _, sign := range []string{markerFile, record} {
			if s, err := os.Lstat(filepath.Join(path, sign)); err == nil && s.Mode().IsRegular() {
				return os.RemoveAll(path)
			}
		}
		// os.Remove takes an empty directory and none with content
		if err := os.Remove(path); err == nil {
			return nil
		}
	}
	return fmt.Errorf("%s is in the way of the install and was not left by one (it is no directory holding %s or %s): move it away", path, markerFile, record)
}

// replaceable reports, as an error when not, whether dir may be replaced
// by an install: it is missing, empty, or carries the record of an earlier
// install. Anything else was put there by someone else, a LibDir pointed at
// a directory with other content, and is left alone.
func replaceable(dir, record string) error {
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if len(entries) == 0 {
		return nil
	}
	if _, err := os.Lstat(filepath.Join(dir, record)); err == nil {
		return nil
	}
	return fmt.Errorf("%s holds files that were not installed here (it has no %s): refusing to replace it; point the managed directory at an empty or new one, or use this one unmanaged", dir, record)
}

// writeSynced writes data to path and flushes it to disk.
func writeSynced(path string, data []byte) (err error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600) //nolint:gosec // path is the caller's fixed record name inside the staging directory this package just made
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()
	if _, err := f.Write(data); err != nil {
		return err
	}
	return f.Sync()
}

// swap puts staging in dir's place. The directory being replaced is moved
// aside to retired first and deleted afterwards, so dir is missing only
// between two renames, and is put back when the second one fails. Nothing
// is at retired when it is called (see clearLeftover), and what is deleted
// there afterwards is the directory this call moved.
func swap(staging, retired, dir string) error {
	err := os.Rename(dir, retired)
	had := err == nil
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.Rename(staging, dir); err != nil {
		if had {
			_ = os.Rename(retired, dir)
		}
		return err
	}
	if had {
		_ = os.RemoveAll(retired)
	}
	syncDir(filepath.Dir(dir))
	return nil
}

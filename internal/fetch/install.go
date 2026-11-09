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
	// its file system, so the last step is a rename.
	archiveSuffix = ".download"
	stagingSuffix = ".staging"
	retiredSuffix = ".old"
	// staleStagingPrefix is how staging directories were named before
	// installs took a lock: a random name per attempt, which a killed
	// process left behind for good.
	staleStagingPrefix = ".partial-"
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
	// Extract unpacks the verified archive into dir, an empty directory.
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
// installed reports whether the directory was filled by this call.
func (d Downloader) InstallDir(ctx context.Context, in DirInstall) (installed bool, err error) {
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

// ready asks in.Ready, which may be nil.
func (in DirInstall) ready() (bool, error) {
	if in.Ready == nil {
		return false, nil
	}
	return in.Ready()
}

// lockDir takes the lock that serializes installs into dir, creating dir's
// parent and the lock file as needed, and clears what an install that was
// killed between its last steps left next to dir.
func lockDir(ctx context.Context, dir string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(dir), 0o750); err != nil {
		return nil, err
	}
	lock, err := openLocked(ctx, dir+lockSuffix)
	if err != nil {
		return nil, err
	}
	_ = os.RemoveAll(dir + stagingSuffix)
	_ = os.RemoveAll(dir + retiredSuffix)
	sweepStale(filepath.Dir(dir), staleStagingPrefix)
	return lock, nil
}

// install fills in.Dir from in.Archive. The caller holds the directory's
// lock.
func (d Downloader) install(ctx context.Context, in DirInstall) error {
	// before the download, not after it: a refusal should not cost one
	if err := replaceable(in.Dir, in.RecordFile); err != nil {
		return err
	}
	archive := in.Dir + archiveSuffix
	if err := d.File(ctx, in.What, in.Archive, archive); err != nil {
		return fmt.Errorf("download %s: %w", in.What, err)
	}
	defer func() { _ = os.Remove(archive) }()
	staging := in.Dir + stagingSuffix
	if err := os.Mkdir(staging, 0o750); err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(staging) }()
	if err := in.Extract(archive, staging); err != nil {
		return fmt.Errorf("extract %s: %w", in.What, err)
	}
	if err := writeSynced(filepath.Join(staging, in.RecordFile), in.Record); err != nil {
		return err
	}
	syncDir(staging)
	return swap(staging, in.Dir)
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
// aside first and deleted afterwards, so dir is missing only between two
// renames, and is put back when the second one fails.
func swap(staging, dir string) error {
	retired := dir + retiredSuffix
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

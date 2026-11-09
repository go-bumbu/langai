package fetch

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
)

// ExtractTarGz copies the entries of the gzipped tarball at archive that
// want accepts into dir, flattened to their base names (release archives
// nest them under a version-named directory), and returns how many it
// wrote. want sees each entry's slash-separated path inside the archive. A
// symlink is recreated only when it points at a bare name in the same
// directory that want accepts too; anything else is rejected rather than
// written. Each file is capped at limit bytes.
func ExtractTarGz(archive, dir string, want func(path string) bool, limit int64) (int, error) {
	f, err := os.Open(archive) //nolint:gosec // archive is a download the caller just verified in its own staging directory, not a path taken from archive content
	if err != nil {
		return 0, err
	}
	defer func() { _ = f.Close() }()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return 0, err
	}
	defer func() { _ = gz.Close() }()
	tr := tar.NewReader(gz)
	found := 0
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return found, nil
		}
		if err != nil {
			return found, err
		}
		if !want(hdr.Name) {
			continue
		}
		dest := filepath.Join(dir, path.Base(hdr.Name))
		switch hdr.Typeflag {
		case tar.TypeReg:
			if err := writeFile(dest, tr, limit); err != nil {
				return found, err
			}
		case tar.TypeSymlink:
			target := hdr.Linkname
			if target != path.Base(target) || !want(path.Join(path.Dir(hdr.Name), target)) { //nolint:gosec // the joined path is only shown to the filter, never opened: dest is always a base name inside dir, and target a bare name

				return found, fmt.Errorf("refusing symlink %s -> %s", path.Base(hdr.Name), target)
			}
			if err := os.Symlink(target, dest); err != nil {
				return found, err
			}
		default:
			continue
		}
		found++
	}
}

// writeFile writes one extracted file, executable (shared libraries must
// be), bounded by limit bytes.
func writeFile(dest string, r io.Reader, limit int64) (err error) {
	out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755) //nolint:gosec // dest is a base name the caller's filter accepted inside its staging directory, writes are capped, and shared libraries must be executable
	if err != nil {
		return err
	}
	defer func() {
		if cerr := out.Close(); err == nil {
			err = cerr
		}
	}()
	n, err := io.Copy(out, io.LimitReader(r, limit+1))
	if err != nil {
		return err
	}
	if n > limit {
		return fmt.Errorf("%s exceeds the %d byte cap", dest, limit)
	}
	return nil
}

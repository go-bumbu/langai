// Package fetch is what the providers share to get their native libraries
// and models onto disk: a download verified by size and SHA-256 before it
// appears under its final name, a file that is downloaded only when it is
// missing, an unverified download that reports the digest a new pin needs,
// the extraction of the libraries out of a release tarball, and the staged
// install of such a tarball into a directory (install.go). It knows nothing
// about which libraries or models those are.
package fetch

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path"
	"path/filepath"
)

// Asset is one downloadable file with its expected digest.
type Asset struct {
	URL    string
	SHA256 string
	Size   int64 // exact size when known, 0 to skip the check
}

// Downloader is how one provider's downloads are made. The zero value works:
// http.DefaultClient, no user agent, no cap beyond an asset's own size, and
// no log lines.
type Downloader struct {
	// Client makes the requests; nil means http.DefaultClient.
	Client *http.Client
	// UserAgent names the downloads to the server.
	UserAgent string
	// Limit caps a single download in bytes; an asset pinned to a larger
	// exact size is capped at that size instead.
	Limit int64
	// Logger gets a line before each download starts (a download can take
	// minutes) and one when a missing file was fetched; nil discards them.
	Logger *slog.Logger
}

// logger is d.Logger, or one that discards.
func (d Downloader) logger() *slog.Logger {
	if d.Logger == nil {
		return slog.New(slog.DiscardHandler)
	}
	return d.Logger
}

// File downloads a to dest, verifying size and SHA-256 before the file
// appears under its final name: the bytes go to a temp file in the same
// directory and are renamed into place only once they check out. what names
// the download in the log line written before it starts.
func (d Downloader) File(ctx context.Context, what string, a Asset, dest string) error {
	d.logger().Info("downloading "+what, slog.String("url", a.URL), slog.Int64("size", a.Size), slog.String("sha256", a.SHA256), slog.String("dest", dest))
	tmp, sum, n, err := d.downloadTemp(ctx, a.URL, dest, max(d.Limit, a.Size))
	if err != nil {
		return err
	}
	if a.Size != 0 && n != a.Size {
		_ = os.Remove(tmp)
		return fmt.Errorf("%s: got %d bytes, expected %d", a.URL, n, a.Size)
	}
	if sum != a.SHA256 {
		_ = os.Remove(tmp)
		return fmt.Errorf("%s: sha256 %s, expected %s", a.URL, sum, a.SHA256)
	}
	if err = os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// EnsureFile makes sure dest exists, downloading a into it (see File) when
// it is absent; a download is logged. what names the file in log lines and
// errors ("model org/name"). An asset without a URL cannot be fetched, so a
// missing file is then an error naming it.
func (d Downloader) EnsureFile(ctx context.Context, what string, a Asset, dest string) error {
	if _, err := os.Stat(dest); err == nil {
		return nil
	}
	if a.URL == "" {
		return fmt.Errorf("%s: %s does not exist and the spec has no URL to download it from", what, dest)
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o750); err != nil {
		return err
	}
	if err := d.File(ctx, what, a, dest); err != nil {
		return fmt.Errorf("download %s: %w", what, err)
	}
	d.logger().Info("downloaded "+what, slog.String("path", dest))
	return nil
}

// Unpinned fetches url into dest with nothing to verify it against and
// returns the SHA-256 and size of what it wrote, so a new file can be
// pinned.
func (d Downloader) Unpinned(ctx context.Context, url, dest string) (string, int64, error) {
	d.logger().Info("downloading unpinned file", slog.String("url", url), slog.String("dest", dest))
	if err := os.MkdirAll(filepath.Dir(dest), 0o750); err != nil {
		return "", 0, err
	}
	tmp, sum, n, err := d.downloadTemp(ctx, url, dest, d.Limit)
	if err != nil {
		return "", 0, err
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return "", 0, err
	}
	return sum, n, nil
}

// downloadTemp streams url into a closed temp file next to dest, capped at
// limit bytes, and returns the temp path with the SHA-256 and size of its
// content. On any error nothing is left behind.
func (d Downloader) downloadTemp(ctx context.Context, url, dest string, limit int64) (tmpPath, sum string, n int64, err error) {
	client := d.Client
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", "", 0, err
	}
	req.Header.Set("User-Agent", d.UserAgent)
	resp, err := client.Do(req)
	if err != nil {
		return "", "", 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return "", "", 0, fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dest), ".download-*")
	if err != nil {
		return "", "", 0, err
	}
	defer func() {
		_ = tmp.Close()
		if err != nil {
			_ = os.Remove(tmp.Name())
		}
	}()
	h := sha256.New()
	n, err = io.Copy(io.MultiWriter(tmp, h), io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return "", "", 0, err
	}
	if n > limit {
		err = fmt.Errorf("%s exceeds the %d byte download cap", url, limit)
		return "", "", 0, err
	}
	if err = tmp.Close(); err != nil {
		return "", "", 0, err
	}
	return tmp.Name(), hex.EncodeToString(h.Sum(nil)), n, nil
}

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

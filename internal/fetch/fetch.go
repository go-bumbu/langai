// Package fetch is what the providers share to get their native libraries
// and models onto disk: a download verified by size and SHA-256 before it
// appears under its final name, a file that is downloaded only when it is
// missing or not the pinned size, an unverified download that reports the
// digest a new pin needs, the extraction of the libraries out of a release
// tarball (extract.go), and the staged install of such a tarball into a
// directory (install.go). It knows nothing about which libraries or models
// those are.
//
// A download is written to "<dest>.partial" next to its destination and
// renamed into place once it checks out. The partial file is held under a
// lock (lock.go), so goroutines and processes after the same file take
// turns, and it survives an interrupted run: the next one asks the server
// for the rest with a Range request and verifies the whole against the
// pinned digest. That is so where the system has file locks (unix). Where
// it has none, processes are not kept apart: each writes a partial file of
// its own ("<dest>.partial-<pid>", see lock_other.go), so none writes
// through another, and nothing is resumed.
//
// The directories all this happens in are the caller's, and may hold
// anything. The package writes only under names derived from the
// destination ("<dest>.partial", "<dir>.lock", "<dir>.langai-download",
// "<dir>.langai-staging", "<dir>.langai-retired") and removes only what it
// wrote or what carries its marker: nothing is swept for its name or age.
package fetch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// partialSuffix is appended to a destination to name the file its download
// is written to (see partialPath).
const partialSuffix = ".partial"

// Asset is one downloadable file with its expected digest.
type Asset struct {
	URL string
	// SHA256 is the digest in hex, in either case.
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

// WithDefaultTimeout bounds ctx by d unless the caller already gave it a
// deadline, which then stands: a default is for callers that set none, and
// must not cut short the time a caller allowed for a large download.
func WithDefaultTimeout(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, d)
}

// File downloads a to dest, verifying size and SHA-256 before the file
// appears under its final name, and replaces whatever was there. The bytes
// of an earlier, interrupted attempt are kept and only the rest is asked
// for. what names the download in the log line written before it starts.
func (d Downloader) File(ctx context.Context, what string, a Asset, dest string) error {
	_, _, _, err := d.download(ctx, what, a, dest, nil)
	return err
}

// EnsureFile makes sure dest holds a, downloading it (see File) when dest
// is absent or, for an asset with a known size, has another size: a file
// cut short by a crash or a full disk is fetched again instead of being
// handed to a model loader. A file of the right size is taken as it is,
// without reading it. what names the file in log lines and errors ("model
// org/name"). An asset without a URL cannot be fetched, so a missing or
// wrong file is then an error naming it.
func (d Downloader) EnsureFile(ctx context.Context, what string, a Asset, dest string) error {
	ok, got, err := onDisk(dest, a.Size)
	if err != nil {
		return err
	}
	if ok {
		return nil
	}
	if a.URL == "" {
		if got >= 0 {
			return fmt.Errorf("%s: %s has %d bytes where the spec says %d, and the spec has no URL to download it from", what, dest, got, a.Size)
		}
		return fmt.Errorf("%s: %s does not exist and the spec has no URL to download it from", what, dest)
	}
	if got >= 0 {
		d.logger().Warn("replacing "+what+": the file is not the pinned size", slog.String("path", dest), slog.Int64("size", got), slog.Int64("pinned", a.Size))
	}
	_, _, fetched, err := d.download(ctx, what, a, dest, func() bool {
		ok, _, err := onDisk(dest, a.Size)
		return err == nil && ok
	})
	if err != nil {
		return fmt.Errorf("download %s: %w", what, err)
	}
	if fetched {
		d.logger().Info("downloaded "+what, slog.String("path", dest))
	}
	return nil
}

// Unpinned fetches url into dest with nothing to verify it against and
// returns the SHA-256 and size of what it wrote, so a new file can be
// pinned. Without a digest a resumed download could not be checked, so this
// one always starts from the first byte.
func (d Downloader) Unpinned(ctx context.Context, url, dest string) (string, int64, error) {
	sum, n, _, err := d.download(ctx, "unpinned file", Asset{URL: url}, dest, nil)
	return sum, n, err
}

// onDisk reports whether dest holds an asset of the given size as far as
// can be told without reading it: it exists and, when the size is known
// (not 0), has that size. got is the size found, -1 when there is no file.
func onDisk(dest string, size int64) (ok bool, got int64, err error) {
	info, err := os.Stat(dest)
	if errors.Is(err, os.ErrNotExist) {
		return false, -1, nil
	}
	if err != nil {
		return false, -1, err
	}
	return size == 0 || info.Size() == size, info.Size(), nil
}

// contentError says that the bytes received are not the file that was asked
// for: nothing of them is worth keeping for a later attempt.
type contentError struct{ msg string }

func (e *contentError) Error() string { return e.msg }

// download brings a to dest through dest's partial file, which it holds
// locked from the first byte to the rename. ready, when set, is asked once
// the lock is held: whoever held it before may have finished the same job,
// and then nothing is fetched. It returns the digest and size of what it
// wrote and whether it wrote anything. After a failure the partial file
// stays only when a later attempt can build on it.
func (d Downloader) download(ctx context.Context, what string, a Asset, dest string, ready func() bool) (sum string, n int64, fetched bool, err error) {
	dir := filepath.Dir(dest)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return "", 0, false, err
	}
	release, err := inProcess.acquire(ctx, dest)
	if err != nil {
		return "", 0, false, err
	}
	defer release()
	partial := partialPath(dest)
	part, err := openLocked(ctx, partial)
	if err != nil {
		return "", 0, false, err
	}
	defer func() { _ = part.Close() }()
	if ready != nil && ready() {
		_ = os.Remove(partial)
		return "", 0, false, nil
	}
	d.logger().Info("downloading "+what, slog.String("url", a.URL), slog.Int64("size", a.Size), slog.String("sha256", a.SHA256), slog.String("dest", dest))
	sum, n, err = d.transferVerified(ctx, a, part)
	if err == nil {
		err = publish(part, partial, dest)
	}
	if err != nil {
		if !resumable(part, a, err) {
			_ = os.Remove(partial)
		}
		return "", 0, false, err
	}
	return sum, n, true, nil
}

// resumable reports whether the partial file left by a failed download is
// worth keeping: it holds bytes, of a pinned asset, that the failure did
// not prove wrong, and a later process can find it (see partialPath).
func resumable(part *os.File, a Asset, err error) bool {
	var bad *contentError
	if !crossProcessLocks || a.SHA256 == "" || errors.As(err, &bad) {
		return false
	}
	info, statErr := part.Stat()
	return statErr == nil && info.Size() > 0
}

// publish makes a verified partial file durable and gives it its final
// name: the bytes are flushed before the rename, and the directory after,
// so a crash leaves either no file or the whole one.
func publish(part *os.File, partial, dest string) error {
	if err := part.Sync(); err != nil {
		return err
	}
	if runtime.GOOS == "windows" {
		_ = part.Close() // a file that is open there cannot be renamed
	}
	if err := os.Rename(partial, dest); err != nil {
		return err
	}
	syncDir(filepath.Dir(dest))
	return nil
}

// syncDir flushes a directory's entries to disk. Not every file system can
// (some network mounts refuse), and the rename already happened, so a
// failure is not reported.
func syncDir(dir string) {
	d, err := os.Open(dir) //nolint:gosec // dir is the directory of the caller's own destination
	if err != nil {
		return
	}
	_ = d.Sync()
	_ = d.Close()
}

// transferVerified fills part with a's content and checks it. When the
// bytes an earlier attempt left do not add up to the pinned file, it starts
// over once from the first byte: a stale or corrupted partial file must not
// fail a download for good.
func (d Downloader) transferVerified(ctx context.Context, a Asset, part *os.File) (sum string, n int64, err error) {
	sum, n, resumed, err := d.transfer(ctx, a, part, a.SHA256 != "")
	var bad *contentError
	if resumed && errors.As(err, &bad) {
		sum, n, _, err = d.transfer(ctx, a, part, false)
	}
	return sum, n, err
}

// transfer writes a's content into part, after the bytes part already holds
// when resume is set, and checks the cap, the size and the digest of the
// whole. resumed says whether earlier bytes were built on.
func (d Downloader) transfer(ctx context.Context, a Asset, part *os.File, resume bool) (sum string, n int64, resumed bool, err error) {
	limit := max(d.Limit, a.Size)
	if limit <= 0 {
		limit = math.MaxInt64 - 1 // neither a cap nor a size: unbounded
	}
	h := sha256.New()
	have, err := kept(part, h, a.Size, resume)
	if err != nil {
		return "", 0, false, err
	}
	n, resumed = have, have > 0
	if have == 0 || a.Size == 0 || have < a.Size {
		if n, resumed, err = d.get(ctx, a.URL, part, h, have, limit); err != nil {
			return "", 0, resumed, err
		}
	}
	sum = hex.EncodeToString(h.Sum(nil))
	switch {
	case n > limit:
		err = &contentError{fmt.Sprintf("%s exceeds the %d byte download cap", a.URL, limit)}
	case a.Size != 0 && n != a.Size:
		err = &contentError{fmt.Sprintf("%s: got %d bytes, expected %d", a.URL, n, a.Size)}
	case a.SHA256 != "" && !strings.EqualFold(sum, a.SHA256):
		err = &contentError{fmt.Sprintf("%s: sha256 %s, expected %s", a.URL, sum, a.SHA256)}
	}
	if err != nil {
		return "", 0, resumed, err
	}
	return sum, n, resumed, nil
}

// kept prepares part for a transfer and returns how many bytes of an
// earlier attempt it keeps, hashed into h and with the file positioned
// after them. Without resume, or when part is longer than the asset can
// be, it is emptied instead.
func kept(part *os.File, h hash.Hash, size int64, resume bool) (int64, error) {
	info, err := part.Stat()
	if err != nil {
		return 0, err
	}
	have := info.Size()
	if have == 0 {
		return 0, nil
	}
	if !resume || (size != 0 && have > size) {
		return 0, empty(part)
	}
	if _, err := part.Seek(0, io.SeekStart); err != nil {
		return 0, err
	}
	if _, err := io.Copy(h, part); err != nil {
		return 0, err
	}
	return have, nil
}

// empty truncates part and positions it at its start.
func empty(part *os.File) error {
	if err := part.Truncate(0); err != nil {
		return err
	}
	_, err := part.Seek(0, io.SeekStart)
	return err
}

// get requests url, from byte have on when part already holds that many,
// and appends the reply to part and h, at most limit bytes in all. A server
// that ignores the range and sends the whole file is taken at its word:
// part and h start over. n is the size part ends with, and resumed whether
// the bytes it held were kept.
func (d Downloader) get(ctx context.Context, url string, part *os.File, h hash.Hash, have, limit int64) (n int64, resumed bool, err error) {
	client := d.Client
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, false, err
	}
	req.Header.Set("User-Agent", d.UserAgent)
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
	}
	resp, err := client.Do(req)
	if err != nil {
		return have, have > 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	switch {
	case resp.StatusCode == http.StatusOK:
		if have > 0 {
			if err := empty(part); err != nil {
				return 0, false, err
			}
			h.Reset()
			have = 0
		}
	case resp.StatusCode == http.StatusPartialContent && have > 0:
		if !strings.HasPrefix(resp.Header.Get("Content-Range"), fmt.Sprintf("bytes %d-", have)) {
			return have, true, &contentError{fmt.Sprintf("GET %s: asked for the bytes from %d on, got %q", url, have, resp.Header.Get("Content-Range"))}
		}
	case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable && have > 0:
		// nothing follows what part holds: it is the whole file, or not this
		// file at all, which the digest tells
		return have, true, nil
	default:
		return have, have > 0, fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	copied, err := io.Copy(io.MultiWriter(part, h), io.LimitReader(resp.Body, limit+1-have))
	return have + copied, have > 0, err
}

package fetch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const testRecord = "install.json"

// libArchive serves a release tarball with one library and its symlink,
// plus whatever extra entries the test adds.
func libArchive(t *testing.T, extra ...tarEntry) Asset {
	t.Helper()
	entries := append([]tarEntry{
		{name: "lib/libx.so.1", body: "x"},
		{name: "lib/libx.so", link: "libx.so.1"},
		{name: "include/x.h", body: "h"},
	}, extra...)
	_, a := serve(t, tarGzBytes(t, "rel-1.0", entries))
	return a
}

// countedArchive serves the same kind of tarball and counts the requests
// for it.
func countedArchive(t *testing.T) (Asset, *atomic.Int32) {
	t.Helper()
	body := tarGzBytes(t, "rel-1.0", []tarEntry{{name: "lib/libx.so.1", body: "x"}, {name: "lib/libx.so", link: "libx.so.1"}})
	requests := new(atomic.Int32)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	sum := sha256.Sum256(body)
	return Asset{URL: srv.URL + "/libs.tgz", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(body))}, requests
}

// install describes an install of a into dir with the test's extraction,
// ready once the library and the record are there.
func install(dir string, a Asset) DirInstall {
	return DirInstall{
		Dir: dir, What: "libx 1.0", Archive: a, RecordFile: testRecord, Record: []byte(`{"v":1}`),
		Extract: func(archive, dir string) error {
			n, err := ExtractTarGz(archive, dir, wantLib, cap)
			if err == nil && n == 0 {
				err = errors.New("no libraries")
			}
			return err
		},
		Ready: func() (bool, error) {
			_, lib := os.Stat(filepath.Join(dir, "libx.so"))
			_, rec := os.Stat(filepath.Join(dir, testRecord))
			return lib == nil && rec == nil, nil
		},
	}
}

// names lists the entries of dir, sorted.
func names(t *testing.T, dir string) string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = e.Name()
	}
	return strings.Join(out, ",")
}

func TestInstallDir_DownloadsExtractsAndRecords(t *testing.T) {
	a := libArchive(t)
	dir := filepath.Join(t.TempDir(), "libs", "1.0")
	used := 0
	in := install(dir, a)
	in.Use = func() error { used++; return nil }
	installed, err := dl(cap).InstallDir(context.Background(), in)
	if err != nil || !installed {
		t.Fatalf("InstallDir = %v, %v", installed, err)
	}
	if got := names(t, dir); got != testRecord+",libx.so,libx.so.1" {
		t.Errorf("installed %s", got)
	}
	if rec, _ := os.ReadFile(filepath.Join(dir, testRecord)); string(rec) != `{"v":1}` {
		t.Errorf("record = %q", rec)
	}
	// only the directory and its lock file stay: no archive, no staging
	if got := names(t, filepath.Dir(dir)); got != "1.0,1.0.lock" {
		t.Errorf("left next to the directory: %s", got)
	}
	// a ready directory is used as it is: the second call must not need the network
	in.Archive = Asset{URL: "http://127.0.0.1:1/unreachable", SHA256: a.SHA256}
	installed, err = dl(cap).InstallDir(context.Background(), in)
	if err != nil || installed {
		t.Fatalf("second InstallDir = %v, %v; want the directory reused", installed, err)
	}
	if used != 2 {
		t.Errorf("Use ran %d times, want once per call", used)
	}
}

// TestInstallDir_FailureLeavesNothing: a failed download or extraction never
// produces the directory, leaves no staging behind, and does not run Use.
func TestInstallDir_FailureLeavesNothing(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(srv.Close)
	_, empty := serve(t, tarGzBytes(t, "rel-1.0", []tarEntry{{name: "README", body: "no libs"}}))
	for name, c := range map[string]struct {
		a    Asset
		want string
	}{
		"download":   {Asset{URL: srv.URL + "/missing", SHA256: strings.Repeat("0", 64)}, "download libx 1.0"},
		"extraction": {empty, "extract libx 1.0"},
	} {
		dir := filepath.Join(t.TempDir(), "libs", "1.0")
		in := install(dir, c.a)
		in.Use = func() error { t.Errorf("%s: Use ran on a directory that was not installed", name); return nil }
		_, err := dl(cap).InstallDir(context.Background(), in)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", name, err, c.want)
		}
		if got := names(t, filepath.Dir(dir)); got != "1.0.lock" {
			t.Errorf("%s: left behind: %s", name, got)
		}
	}
}

// TestInstallDir_ReplacesOnlyItsOwnDirectories: a managed directory pointed
// at something else's files (LibDir "/opt/app") must not be wiped. Missing,
// empty and recorded directories are replaced; anything else is an error,
// raised before the download.
func TestInstallDir_ReplacesOnlyItsOwnDirectories(t *testing.T) {
	cases := []struct {
		name    string
		prepare func(t *testing.T, dir string)
		refused bool
	}{
		{"missing", func(*testing.T, string) {}, false},
		{"empty", func(t *testing.T, dir string) { mkdir(t, dir) }, false},
		{"an earlier install", func(t *testing.T, dir string) {
			mkdir(t, dir)
			write(t, filepath.Join(dir, testRecord), "{}")
			write(t, filepath.Join(dir, "libold.so"), "old")
		}, false},
		{"someone else's files", func(t *testing.T, dir string) {
			mkdir(t, dir)
			write(t, filepath.Join(dir, "app.conf"), "precious")
		}, true},
		{"a file", func(t *testing.T, dir string) { write(t, dir, "not a directory") }, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a, requests := countedArchive(t)
			dir := filepath.Join(t.TempDir(), "app")
			c.prepare(t, dir)
			_, err := dl(cap).InstallDir(context.Background(), install(dir, a))
			if !c.refused {
				if err != nil {
					t.Fatal(err)
				}
				if got := names(t, dir); got != testRecord+",libx.so,libx.so.1" {
					t.Errorf("after the install: %s", got)
				}
				return
			}
			if err == nil {
				t.Fatal("the directory was replaced")
			}
			if requests.Load() != 0 {
				t.Error("the archive was downloaded before the refusal")
			}
			if c.name == "someone else's files" {
				if !strings.Contains(err.Error(), "refusing to replace") || !strings.Contains(err.Error(), testRecord) {
					t.Errorf("err = %v, want a refusal naming the record", err)
				}
				if got, _ := os.ReadFile(filepath.Join(dir, "app.conf")); string(got) != "precious" {
					t.Error("the foreign file is gone")
				}
			}
		})
	}
}

// TestInstallDir_ConcurrentCallsInstallOnce: goroutines after the same
// directory take turns; the first installs, the others find it ready, and
// each runs Use while it holds the lock, so no Use overlaps an install.
func TestInstallDir_ConcurrentCallsInstallOnce(t *testing.T) {
	var installing, overlaps atomic.Int32
	a, requests := countedArchive(t)
	dir := filepath.Join(t.TempDir(), "libs", "1.0")
	var wg sync.WaitGroup
	errs := make([]error, 8)
	for i := range errs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			in := install(dir, a)
			extract := in.Extract
			in.Extract = func(archive, dir string) error {
				installing.Add(1)
				defer installing.Add(-1)
				time.Sleep(20 * time.Millisecond)
				return extract(archive, dir)
			}
			in.Use = func() error {
				if installing.Load() != 0 {
					overlaps.Add(1)
				}
				_, err := os.Stat(filepath.Join(dir, "libx.so"))
				return err
			}
			_, errs[i] = dl(cap).InstallDir(context.Background(), in)
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("call %d: %v", i, err)
		}
	}
	if n := requests.Load(); n != 1 {
		t.Errorf("the archive was downloaded %d times", n)
	}
	if n := overlaps.Load(); n != 0 {
		t.Errorf("Use ran during an install %d times", n)
	}
	if got := names(t, filepath.Dir(dir)); got != "1.0,1.0.lock" {
		t.Errorf("left next to the directory: %s", got)
	}
}

// TestInstallDir_ReadyErrorAndLeftovers: a Ready that refuses stops the
// install before anything is fetched, and what a killed install left next
// to the directory (its staging, the directory it was replacing, a staging
// directory of the old naming) is cleared.
func TestInstallDir_ReadyErrorAndLeftovers(t *testing.T) {
	a := libArchive(t)
	parent := t.TempDir()
	dir := filepath.Join(parent, "1.0")
	for _, left := range []string{dir + stagingSuffix, dir + retiredSuffix, filepath.Join(parent, ".partial-123"), filepath.Join(parent, ".partial-fresh")} {
		mkdir(t, left)
		write(t, filepath.Join(left, "junk"), "x")
	}
	past := time.Now().Add(-2 * staleAfter)
	if err := os.Chtimes(filepath.Join(parent, ".partial-123"), past, past); err != nil {
		t.Fatal(err)
	}
	in := install(dir, a)
	boom := errors.New("in use")
	in.Ready = func() (bool, error) { return false, boom }
	if _, err := dl(cap).InstallDir(context.Background(), in); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want Ready's", err)
	}
	if got := names(t, parent); got != ".partial-fresh,1.0.lock" {
		t.Errorf("after a refused install: %s", got)
	}
	if _, err := dl(cap).InstallDir(context.Background(), install(dir, a)); err != nil {
		t.Fatal(err)
	}
	if got := names(t, parent); got != ".partial-fresh,1.0,1.0.lock" {
		t.Errorf("after the install: %s", got)
	}
}

// TestInstallDir_UnlockableButReady: a directory whose parent cannot take a
// lock file (a read-only image) is used when it is ready, and is an error
// only when it would have to be written.
func TestInstallDir_UnlockableButReady(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root writes into read-only directories")
	}
	a := libArchive(t)
	parent := filepath.Join(t.TempDir(), "ro")
	dir := filepath.Join(parent, "1.0")
	if _, err := dl(cap).InstallDir(context.Background(), install(dir, a)); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(dir + lockSuffix); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(parent, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(parent, 0o700) }) // so the temp dir can be removed
	used := false
	in := install(dir, a)
	in.Use = func() error { used = true; return nil }
	if _, err := dl(cap).InstallDir(context.Background(), in); err != nil || !used {
		t.Errorf("ready directory without a lock: used %v, err %v", used, err)
	}
	in.Ready = nil
	if _, err := dl(cap).InstallDir(context.Background(), in); err == nil {
		t.Error("an install into a read-only parent succeeded")
	}
}

func mkdir(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o750); err != nil {
		t.Fatal(err)
	}
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestEnsureFile(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	present := filepath.Join(dir, "present.bin")
	write(t, present, "model")
	// a present file is kept: nothing is requested
	if err := dl(cap).EnsureFile(ctx, "model m", Asset{URL: "http://127.0.0.1:1/unreachable", SHA256: strings.Repeat("0", 64)}, present); err != nil {
		t.Errorf("present file: %v", err)
	}
	absent := filepath.Join(dir, "absent.bin")
	err := dl(cap).EnsureFile(ctx, "model m", Asset{}, absent)
	if err == nil || !strings.Contains(err.Error(), "no URL") || !strings.Contains(err.Error(), absent) || !strings.Contains(err.Error(), "model m") {
		t.Errorf("absent without URL: err = %v", err)
	}
	_, a := serve(t, []byte("downloaded model"))
	dest := filepath.Join(dir, "sub", "model.bin")
	if err := dl(cap).EnsureFile(ctx, "model m", a, dest); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); string(got) != "downloaded model" {
		t.Errorf("downloaded %q", got)
	}
	a.SHA256 = strings.Repeat("1", 64)
	bad := filepath.Join(dir, "bad.bin")
	if err := dl(cap).EnsureFile(ctx, "model m", a, bad); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Errorf("bad digest: err = %v", err)
	}
	if _, err := os.Stat(bad); err == nil {
		t.Error("a file with the wrong digest appeared under its final name")
	}
}

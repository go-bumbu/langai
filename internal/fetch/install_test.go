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

// siblings puts next to dir what could be taken for an install's leftovers
// and is not: directories under the names an older version used for its
// staging and its retired directory (a user's "libs.old" backup has that
// name too), a staging directory of the oldest naming, and directories under
// today's names that carry no marker. Each holds a file that must survive.
// It returns the files.
func siblings(t *testing.T, dir string) []string {
	t.Helper()
	var precious []string
	for _, d := range []string{dir + ".old", dir + ".staging", filepath.Join(filepath.Dir(dir), ".partial-123")} {
		mkdir(t, d)
		p := filepath.Join(d, "precious.txt")
		write(t, p, "mine")
		precious = append(precious, p)
		past := time.Now().Add(-30 * 24 * time.Hour)
		if err := os.Chtimes(d, past, past); err != nil {
			t.Fatal(err)
		}
	}
	return precious
}

// TestInstallDir_NeverTouchesWhatItDoesNotOwn: the directory's parent is
// the caller's (with LibDir "~/llama" it is the home directory), so nothing
// next to the directory is removed for looking like a leftover. A call that
// finds the directory ready changes nothing at all, not even a real
// leftover; one that refuses does not either; and one that installs clears
// only what carries this package's marker or the install's record.
func TestInstallDir_NeverTouchesWhatItDoesNotOwn(t *testing.T) {
	for _, tc := range []struct {
		name string
		// ready is what Ready answers before the install; nil means the
		// directory's own state
		ready func() (bool, error)
		// installs says whether the call fills the directory
		installs bool
	}{
		{name: "the directory is ready", ready: func() (bool, error) { return true, nil }},
		{name: "the install is refused", ready: func() (bool, error) { return false, errors.New("in use") }},
		{name: "the directory is installed", installs: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := libArchive(t)
			parent := t.TempDir()
			dir := filepath.Join(parent, "libs")
			precious := siblings(t, dir)
			// a killed install's own leftovers: marked staging, a retired
			// directory with the record
			mkdir(t, dir+stagingSuffix)
			write(t, filepath.Join(dir+stagingSuffix, markerFile), "")
			write(t, filepath.Join(dir+stagingSuffix, "libhalf.so"), "half")
			mkdir(t, dir+retiredSuffix)
			write(t, filepath.Join(dir+retiredSuffix, testRecord), "{}")

			in := install(dir, a)
			if tc.ready != nil {
				in.Ready = tc.ready
			}
			installed, _ := dl(cap).InstallDir(context.Background(), in)
			if installed != tc.installs {
				t.Fatalf("installed = %v, want %v", installed, tc.installs)
			}
			for _, p := range precious {
				if got, _ := os.ReadFile(p); string(got) != "mine" {
					t.Errorf("%s is gone", p)
				}
			}
			after := names(t, parent)
			if !tc.installs {
				if want := ".partial-123,libs.langai-retired,libs.langai-staging,libs.lock,libs.old,libs.staging"; after != want {
					t.Errorf("next to the directory: %s\nwant                   %s", after, want)
				}
				return
			}
			if want := ".partial-123,libs,libs.lock,libs.old,libs.staging"; after != want {
				t.Errorf("after the install: %s\nwant               %s", after, want)
			}
			if got := names(t, dir); got != testRecord+",libx.so,libx.so.1" {
				t.Errorf("installed %s", got)
			}
		})
	}
}

// TestInstallDir_LeftoversOfItsOwnNames: a directory under one of this
// package's own sibling names is cleared when it is a leftover of an install
// (it carries the marker or the record, or is empty) and stops the install,
// untouched and before the download, when it is anything else.
func TestInstallDir_LeftoversOfItsOwnNames(t *testing.T) {
	for _, tc := range []struct {
		name    string
		suffix  string
		files   map[string]string
		refused bool
	}{
		{name: "staging with the marker", suffix: stagingSuffix, files: map[string]string{markerFile: "", "libhalf.so": "half"}},
		{name: "staging with the record", suffix: stagingSuffix, files: map[string]string{testRecord: "{}", "libx.so": "x"}},
		{name: "empty staging", suffix: stagingSuffix},
		{name: "retired with the record", suffix: retiredSuffix, files: map[string]string{testRecord: "{}", "libold.so": "old"}},
		{name: "empty retired", suffix: retiredSuffix},
		{name: "someone's files under the staging name", suffix: stagingSuffix, files: map[string]string{"notes.txt": "mine"}, refused: true},
		{name: "someone's files under the retired name", suffix: retiredSuffix, files: map[string]string{"notes.txt": "mine"}, refused: true},
		{name: "a directory called like the marker", suffix: stagingSuffix, files: map[string]string{"notes.txt": "mine", markerFile + "/x": "x"}, refused: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, requests := countedArchive(t)
			parent := t.TempDir()
			dir := filepath.Join(parent, "libs")
			mkdir(t, dir)
			write(t, filepath.Join(dir, testRecord), "{}") // an earlier install, to be replaced
			in := install(dir, a)
			in.Ready = nil
			left := dir + tc.suffix
			mkdir(t, left)
			for name, content := range tc.files {
				mkdir(t, filepath.Dir(filepath.Join(left, name)))
				write(t, filepath.Join(left, name), content)
			}
			_, err := dl(cap).InstallDir(context.Background(), in)
			if !tc.refused {
				if err != nil {
					t.Fatal(err)
				}
				if got := names(t, parent); got != "libs,libs.lock" {
					t.Errorf("after the install: %s", got)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), filepath.Base(left)) {
				t.Fatalf("err = %v, want a refusal naming %s", err, filepath.Base(left))
			}
			if requests.Load() != 0 {
				t.Error("the archive was downloaded before the refusal")
			}
			if got, _ := os.ReadFile(filepath.Join(left, "notes.txt")); string(got) != "mine" {
				t.Error("the foreign file is gone")
			}
			if got := names(t, dir); got != testRecord {
				t.Errorf("the directory was changed: %s", got)
			}
		})
	}
	// a plain file under one of the names is nobody's leftover either
	a, requests := countedArchive(t)
	dir := filepath.Join(t.TempDir(), "libs")
	write(t, dir+retiredSuffix, "mine")
	in := install(dir, a)
	if _, err := dl(cap).InstallDir(context.Background(), in); err == nil || requests.Load() != 0 {
		t.Errorf("a file at the retired name: err = %v after %d downloads, want a refusal before any", err, requests.Load())
	}
	if got, _ := os.ReadFile(dir + retiredSuffix); string(got) != "mine" {
		t.Error("the file at the retired name is gone")
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

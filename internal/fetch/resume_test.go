package fetch

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// rangeServer serves body the way a file server does, Range requests
// included, and records the Range header of every request. cutAfter > 0
// makes the first request break off after that many bytes; noRanges makes
// the server ignore Range and always send the whole body.
type rangeServer struct {
	body     []byte
	cutAfter int
	noRanges bool

	mu     sync.Mutex
	ranges []string
}

func (s *rangeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.ranges = append(s.ranges, r.Header.Get("Range"))
	first := len(s.ranges) == 1
	s.mu.Unlock()
	switch {
	case first && s.cutAfter > 0:
		w.Header().Set("Content-Length", strconv.Itoa(len(s.body)))
		_, _ = w.Write(s.body[:s.cutAfter])
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		panic(http.ErrAbortHandler) // the connection drops mid-body
	case s.noRanges:
		_, _ = w.Write(s.body)
	default:
		http.ServeContent(w, r, "file", time.Time{}, bytes.NewReader(s.body))
	}
}

// requests returns the Range headers seen so far.
func (s *rangeServer) requests() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.ranges...)
}

// serveRanges starts s and returns the asset describing its body.
func serveRanges(t *testing.T, s *rangeServer) Asset {
	t.Helper()
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	sum := sha256.Sum256(s.body)
	return Asset{URL: srv.URL + "/file", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(s.body))}
}

// payload is n bytes that differ along the file, so a misplaced range shows.
func payload(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 + i/251)
	}
	return b
}

// TestFile_ResumesAnInterruptedDownload: a download that broke off keeps
// what it got, and the next attempt asks only for the rest.
func TestFile_ResumesAnInterruptedDownload(t *testing.T) {
	s := &rangeServer{body: payload(64 << 10), cutAfter: 20 << 10}
	a := serveRanges(t, s)
	dest := filepath.Join(t.TempDir(), "model.bin")
	if err := dl(cap).File(context.Background(), "test", a, dest); err == nil {
		t.Fatal("the interrupted download succeeded")
	}
	if _, err := os.Stat(dest); err == nil {
		t.Fatal("an interrupted download appeared under its final name")
	}
	info, err := os.Stat(dest + partialSuffix)
	if err != nil || info.Size() == 0 || info.Size() >= int64(len(s.body)) {
		t.Fatalf("partial file after the interruption: %v, %v", info, err)
	}
	if err := dl(cap).File(context.Background(), "test", a, dest); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, s.body) {
		t.Fatal("the resumed file is not the served one")
	}
	reqs := s.requests()
	if len(reqs) != 2 || reqs[0] != "" || reqs[1] != "bytes="+strconv.FormatInt(info.Size(), 10)+"-" {
		t.Errorf("requests = %q, want a full one then the rest from byte %d", reqs, info.Size())
	}
	if got := names(t, filepath.Dir(dest)); got != "model.bin" {
		t.Errorf("left next to the file: %s", got)
	}
}

// TestFile_FallsBackToAFullDownload: a partial file is a shortcut, never a
// reason to fail. A server without ranges, bytes that do not belong to the
// file and a partial file longer than the file all end in a full download.
func TestFile_FallsBackToAFullDownload(t *testing.T) {
	body := payload(8 << 10)
	wrong := bytes.Repeat([]byte("x"), 1000)
	cases := []struct {
		name     string
		partial  []byte
		noRanges bool
		want     []string // the Range header of each request
	}{
		{"server ignores ranges", body[:1000], true, []string{"bytes=1000-"}},
		{"stale bytes", wrong, false, []string{"bytes=1000-", ""}},
		{"longer than the file", append(append([]byte(nil), body...), 'x'), false, []string{""}},
		{"complete but never renamed", body, false, nil},
		{"complete with other bytes", bytes.Repeat([]byte("y"), len(body)), false, []string{""}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := &rangeServer{body: body, noRanges: c.noRanges}
			a := serveRanges(t, s)
			dest := filepath.Join(t.TempDir(), "model.bin")
			if err := os.WriteFile(dest+partialSuffix, c.partial, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := dl(cap).File(context.Background(), "test", a, dest); err != nil {
				t.Fatal(err)
			}
			if got, _ := os.ReadFile(dest); !bytes.Equal(got, body) {
				t.Error("the file is not the served one")
			}
			if got := s.requests(); strings.Join(got, "|") != strings.Join(c.want, "|") {
				t.Errorf("requests = %q, want %q", got, c.want)
			}
			if got := names(t, filepath.Dir(dest)); got != "model.bin" {
				t.Errorf("left next to the file: %s", got)
			}
		})
	}
}

// TestFile_UnknownSizeResumes: without a pinned size only the digest can
// say whether a partial file is complete; a server that has nothing more to
// send (416) leaves that to the digest.
func TestFile_UnknownSizeResumes(t *testing.T) {
	body := payload(4 << 10)
	for name, partial := range map[string][]byte{"half": body[:2000], "whole": body} {
		s := &rangeServer{body: body}
		a := serveRanges(t, s)
		a.Size = 0
		dest := filepath.Join(t.TempDir(), "model.bin")
		if err := os.WriteFile(dest+partialSuffix, partial, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := dl(cap).File(context.Background(), "test", a, dest); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got, _ := os.ReadFile(dest); !bytes.Equal(got, body) {
			t.Errorf("%s: the file is not the served one", name)
		}
		if got := s.requests(); len(got) != 1 || got[0] == "" {
			t.Errorf("%s: requests = %q, want one for the rest", name, got)
		}
	}
}

// TestUnpinned_NeverResumes: with no digest to check the result against, a
// partial file is not built on.
func TestUnpinned_NeverResumes(t *testing.T) {
	s := &rangeServer{body: payload(4 << 10)}
	a := serveRanges(t, s)
	dest := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(dest+partialSuffix, []byte("left over"), 0o600); err != nil {
		t.Fatal(err)
	}
	sum, n, err := dl(cap).Unpinned(context.Background(), a.URL, dest)
	if err != nil || sum != a.SHA256 || n != a.Size {
		t.Fatalf("Unpinned = %s, %d, %v", sum, n, err)
	}
	if got := s.requests(); len(got) != 1 || got[0] != "" {
		t.Errorf("requests = %q, want one full download", got)
	}
}

func TestFile_DigestComparisonIgnoresCase(t *testing.T) {
	_, a := serve(t, []byte("hello"))
	a.SHA256 = strings.ToUpper(a.SHA256)
	dest := filepath.Join(t.TempDir(), "out.bin")
	if err := dl(cap).File(context.Background(), "test", a, dest); err != nil {
		t.Fatalf("an upper-case digest was refused: %v", err)
	}
}

// TestEnsureFile_ChecksThePinnedSize: a file of another size than the pin
// is what a crash or a full disk leaves, and is fetched again; without a
// URL it is an error; without a pinned size any file is taken.
func TestEnsureFile_ChecksThePinnedSize(t *testing.T) {
	ctx := context.Background()
	body := []byte("the whole model")
	_, a := serve(t, body)
	dest := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(dest, body[:5], 0o600); err != nil {
		t.Fatal(err)
	}
	local := Asset{Size: a.Size}
	err := dl(cap).EnsureFile(ctx, "model m", local, dest)
	if err == nil || !strings.Contains(err.Error(), "has 5 bytes") || !strings.Contains(err.Error(), "no URL") {
		t.Errorf("truncated file without a URL: err = %v", err)
	}
	if err := dl(cap).EnsureFile(ctx, "model m", Asset{}, dest); err != nil {
		t.Errorf("no pinned size: %v", err)
	}
	if err := dl(cap).EnsureFile(ctx, "model m", a, dest); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, body) {
		t.Errorf("the truncated file was kept: %q", got)
	}
	// the right size is taken as it is: nothing is requested
	a.URL = "http://127.0.0.1:1/unreachable"
	if err := dl(cap).EnsureFile(ctx, "model m", a, dest); err != nil {
		t.Errorf("a file of the pinned size: %v", err)
	}
}

// TestEnsureFile_ConcurrentCallsDownloadOnce: goroutines after the same
// file take turns on its partial file, and all but the first find the work
// done.
func TestEnsureFile_ConcurrentCallsDownloadOnce(t *testing.T) {
	s := &rangeServer{body: payload(32 << 10)}
	a := serveRanges(t, s)
	dest := filepath.Join(t.TempDir(), "models", "model.bin")
	var wg sync.WaitGroup
	errs := make([]error, 8)
	for i := range errs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = dl(cap).EnsureFile(context.Background(), "model m", a, dest)
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("call %d: %v", i, err)
		}
	}
	if got := s.requests(); len(got) != 1 {
		t.Errorf("%d requests for one file", len(got))
	}
	if got := names(t, filepath.Dir(dest)); got != "model.bin" {
		t.Errorf("left next to the file: %s", got)
	}
}

// TestDownload_RemovesNothingNextToItsFile: a download's directory is the
// caller's (a models directory, a home directory), and whatever else is in
// it is not the download's to judge. Files that look like the temp files an
// old version left, however old, are still there afterwards.
func TestDownload_RemovesNothingNextToItsFile(t *testing.T) {
	_, a := serve(t, []byte("hello"))
	dir := t.TempDir()
	past := time.Now().Add(-30 * 24 * time.Hour)
	for _, name := range []string{".download-111", ".download-222", ".partial-333", "notes.partial"} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, past, past); err != nil {
			t.Fatal(err)
		}
	}
	if err := dl(cap).File(context.Background(), "test", a, filepath.Join(dir, "out.bin")); err != nil {
		t.Fatal(err)
	}
	if got := names(t, dir); got != ".download-111,.download-222,.partial-333,notes.partial,out.bin" {
		t.Errorf("next to the download: %s", got)
	}
}

func TestWithDefaultTimeout(t *testing.T) {
	ctx, cancel := WithDefaultTimeout(context.Background(), time.Hour)
	defer cancel()
	if dl, ok := ctx.Deadline(); !ok || time.Until(dl) > time.Hour || time.Until(dl) < 59*time.Minute {
		t.Errorf("no deadline: got %v, %v; want the default", dl, ok)
	}
	// a caller's deadline stands, shorter or longer than the default
	for _, own := range []time.Duration{time.Minute, 5 * time.Hour} {
		parent, stop := context.WithTimeout(context.Background(), own)
		ctx, cancel := WithDefaultTimeout(parent, time.Hour)
		want, _ := parent.Deadline()
		if got, _ := ctx.Deadline(); !got.Equal(want) {
			t.Errorf("caller's %v deadline became %v", own, time.Until(got))
		}
		cancel()
		stop()
	}
}

// TestKeyedLock_WaitsAndGivesUp: a second caller waits for the first, and a
// cancelled one stops waiting.
func TestKeyedLock_WaitsAndGivesUp(t *testing.T) {
	k := keyedLock{held: map[string]chan struct{}{}}
	release, err := k.acquire(context.Background(), "a")
	if err != nil {
		t.Fatal(err)
	}
	if other, err := k.acquire(context.Background(), "b"); err != nil {
		t.Fatalf("another key waited: %v", err)
	} else {
		other()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := k.acquire(ctx, "a"); err == nil {
		t.Fatal("a held key was handed out twice")
	}
	got := make(chan struct{})
	go func() {
		again, err := k.acquire(context.Background(), "a")
		if err == nil {
			again()
		}
		close(got)
	}()
	release()
	select {
	case <-got:
	case <-time.After(5 * time.Second):
		t.Fatal("the waiter never got the released key")
	}
}

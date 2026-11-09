//go:build unix

package ffmpeg

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestCut_NeverTouchesAFileItDidNotWrite: a cut whose out is the input, under
// any spelling, is refused before ffmpeg runs and leaves the video alone,
// and a cut that fails, or cannot even start, leaves a file that was already
// at out as it was: only what this call wrote is ever removed.
func TestCut_NeverTouchesAFileItDidNotWrite(t *testing.T) {
	// an ffmpeg that would overwrite its output, the last argument
	overwrite := `for a; do out=$a; done; printf written > "${out#file:}"` + "\n"
	for _, tc := range []struct {
		name string
		// ffmpeg is the script body, "" for a binary that does not exist
		ffmpeg string
		// out names the output from the directory and the input in it
		out     func(t *testing.T, dir, in string) string
		wantErr string
	}{
		{
			name: "out is the input", ffmpeg: overwrite,
			out:     func(_ *testing.T, _, in string) string { return in },
			wantErr: "same file",
		},
		{
			name: "out is the input, spelled another way", ffmpeg: overwrite,
			out:     func(_ *testing.T, dir, _ string) string { return dir + "/sub/../clip.mp4" },
			wantErr: "same file",
		},
		{
			name: "out is a link to the input", ffmpeg: overwrite,
			out: func(t *testing.T, dir, in string) string {
				link := filepath.Join(dir, "link.mp4")
				if err := os.Symlink(in, link); err != nil {
					t.Fatal(err)
				}
				return link
			},
			wantErr: "same file",
		},
		{
			name: "ffmpeg fails over an earlier cut", ffmpeg: overwrite + "exit 1\n",
			out:     func(_ *testing.T, dir, _ string) string { return filepath.Join(dir, "earlier.mp4") },
			wantErr: "exit status 1",
		},
		{
			name: "no ffmpeg binary, an earlier cut", ffmpeg: "",
			out:     func(_ *testing.T, dir, _ string) string { return filepath.Join(dir, "earlier.mp4") },
			wantErr: "no such file",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.Mkdir(filepath.Join(dir, "sub"), 0o700); err != nil {
				t.Fatal(err)
			}
			in, earlier := filepath.Join(dir, "clip.mp4"), filepath.Join(dir, "earlier.mp4")
			for _, p := range []string{in, earlier} {
				if err := os.WriteFile(p, []byte("precious"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			bin := filepath.Join(dir, "no-such-ffmpeg")
			if tc.ffmpeg != "" {
				bin = script(t, tc.ffmpeg)
			}
			s, _ := New(Config{FFmpeg: bin})
			err := s.Cut(context.Background(), in, 0, time.Second, tc.out(t, dir, in))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("err = %v, want one saying %q", err, tc.wantErr)
			}
			for _, p := range []string{in, earlier} {
				if got, _ := os.ReadFile(p); string(got) != "precious" {
					t.Errorf("%s holds %q after the call, want it untouched", filepath.Base(p), got)
				}
			}
			// nothing else is left next to them
			entries, _ := os.ReadDir(dir)
			for _, e := range entries {
				if n := e.Name(); n != "clip.mp4" && n != "earlier.mp4" && n != "sub" && n != "link.mp4" {
					t.Errorf("the call left %s behind", n)
				}
			}
		})
	}
}

// TestCut_ReplacesAnEarlierOutputOnSuccess: a cut that succeeds takes the
// place of a file already at out, in one rename, and leaves nothing else.
func TestCut_ReplacesAnEarlierOutputOnSuccess(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "seg.mp4")
	if err := os.WriteFile(out, []byte("earlier"), 0o600); err != nil {
		t.Fatal(err)
	}
	s, _ := New(Config{FFmpeg: script(t, `for a; do out=$a; done; printf fresh > "${out#file:}"`+"\n")})
	if err := s.Cut(context.Background(), filepath.Join(dir, "in.mp4"), 0, time.Second, out); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(out); string(got) != "fresh" {
		t.Errorf("out holds %q, want the new cut", got)
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 1 {
		t.Errorf("%d files next to the cut, want it alone", len(entries))
	}
}

// TestCut_RemovesThePartialOutputOfAFailedCut: an ffmpeg that writes part of
// the segment and then fails, or is cancelled while it writes, leaves no
// file at out, so nobody embeds half a segment.
func TestCut_RemovesThePartialOutputOfAFailedCut(t *testing.T) {
	// the last argument is "file:<out>"; the script writes there and fails,
	// or writes and then hangs until it is killed
	for name, tc := range map[string]struct {
		body   string
		cancel time.Duration
	}{
		"ffmpeg fails":     {body: `for a; do out=$a; done; printf partial > "${out#file:}"; exit 1` + "\n"},
		"ffmpeg cancelled": {body: `for a; do out=$a; done; printf partial > "${out#file:}"; exec sleep 30` + "\n", cancel: 300 * time.Millisecond},
	} {
		t.Run(name, func(t *testing.T) {
			s, _ := New(Config{FFmpeg: script(t, tc.body)})
			out := filepath.Join(t.TempDir(), "seg.mp4")
			ctx := context.Background()
			if tc.cancel > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, tc.cancel)
				defer cancel()
			}
			if err := s.Cut(ctx, "/in.mp4", 0, time.Second, out); err == nil {
				t.Fatal("Cut succeeded")
			}
			if _, err := os.Stat(out); !errors.Is(err, os.ErrNotExist) {
				t.Errorf("the failed cut left %s behind (stat: %v)", out, err)
			}
		})
	}
}

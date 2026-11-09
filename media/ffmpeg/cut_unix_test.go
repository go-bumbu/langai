//go:build unix

package ffmpeg

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

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

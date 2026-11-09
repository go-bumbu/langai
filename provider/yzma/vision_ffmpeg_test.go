package yzma

import (
	"context"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/go-bumbu/langai/media/ffmpeg"
)

// renderVideo runs ffmpeg with args to write a test video named name into a
// fresh directory and returns its path; without ffmpeg on PATH the test is
// skipped.
func renderVideo(t *testing.T, name string, args ...string) string {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	path := filepath.Join(t.TempDir(), name)
	if out, err := exec.Command("ffmpeg", append(append([]string{"-v", "error", "-y"}, args...), path)...).CombinedOutput(); err != nil {
		t.Fatalf("render %s: %v: %s", name, err, out)
	}
	return path
}

// TestVision_EmbedVideoThroughFFmpeg runs EmbedVideo with the real Cutter,
// media/ffmpeg, under the fake model: what ffmpeg and ffprobe make of real
// files is what decides the segments, and no stand-in for them can tell.
// The videos are 6 s long and cut into segments of 2 s.
func TestVision_EmbedVideoThroughFFmpeg(t *testing.T) {
	picture := []string{"-f", "lavfi", "-i", "testsrc=s=160x120:d=6:r=25"}
	mpeg4 := []string{"-c:v", "mpeg4", "-q:v", "5", "-g", "25"}
	for _, tc := range []struct {
		name string
		file string
		args []string
		// segments is how many come back, and end where the last one ends
		segments int
		end      time.Duration
	}{
		{
			// the container is as long as its sound, 11 s; the segments are
			// those of the picture
			name: "sound runs on after the picture", file: "clip.mp4",
			args:     append(append(append([]string{}, picture...), "-f", "lavfi", "-i", "sine=d=11", "-c:a", "aac"), mpeg4...),
			segments: 3, end: 6 * time.Second,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := renderVideo(t, tc.file, tc.args...)
			cutter, err := ffmpeg.New(ffmpeg.Config{})
			if err != nil {
				t.Fatal(err)
			}
			f := newFakeVision()
			v := openVisionFake(t, f, VisionOptions{Cutter: cutter, SegmentLength: 2 * time.Second})
			segs, err := v.EmbedVideo(context.Background(), path)
			if err != nil {
				t.Fatal(err)
			}
			if len(segs) != tc.segments {
				t.Fatalf("%d segments, want %d: %+v", len(segs), tc.segments, segs)
			}
			if last := segs[len(segs)-1]; last.End < tc.end-200*time.Millisecond || last.End > tc.end+200*time.Millisecond {
				t.Errorf("the last segment ends at %v, want about %v", last.End, tc.end)
			}
			if n := f.count("bitmapVideo"); n != tc.segments {
				t.Errorf("%d videos went to the model for %d segments", n, tc.segments)
			}
		})
	}
}

package yzma

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
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

// hasEncoder reports whether the ffmpeg on PATH was built with the named
// encoder.
func hasEncoder(t *testing.T, name string) bool {
	t.Helper()
	out, err := exec.Command("ffmpeg", "-hide_banner", "-encoders").Output()
	if err != nil {
		return false
	}
	for line := range strings.Lines(string(out)) {
		if fields := strings.Fields(line); len(fields) > 1 && fields[1] == name {
			return true
		}
	}
	return false
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
		// encoder, when set, is the ffmpeg encoder the video needs; the case
		// is skipped without it
		encoder string
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
		{
			// ffmpeg picks the container of a cut by its extension and knows
			// none for these: the segments are Matroska files
			name: "an extension ffmpeg has no container for", file: "film.divx",
			args:     append(append(append([]string{}, picture...), mpeg4...), "-f", "avi"),
			segments: 3, end: 6 * time.Second,
		},
		{
			name: "a download that was renamed", file: "film.mp4.part",
			args:     append(append(append([]string{}, picture...), mpeg4...), "-f", "mp4"),
			segments: 3, end: 6 * time.Second,
		},
		{
			// the muxer ffmpeg has behind .m4v takes no HEVC; MP4 does
			name: "HEVC in an .m4v", file: "film.m4v", encoder: "libx265",
			args:     append(append([]string{}, picture...), "-c:v", "libx265", "-x265-params", "log-level=error", "-g", "25", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-f", "mp4"),
			segments: 3, end: 6 * time.Second,
		},
		{
			// a raw stream has no timestamps for a Matroska file, so it is
			// cut into a raw stream, and, having no length, as one segment
			name: "a raw H.264 stream", file: "stream.h264", encoder: "libx264",
			args:     append(append([]string{}, picture...), "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p", "-f", "h264"),
			segments: 1, end: 2 * time.Second,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.encoder != "" && !hasEncoder(t, tc.encoder) {
				t.Skipf("this ffmpeg has no %s encoder", tc.encoder)
			}
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

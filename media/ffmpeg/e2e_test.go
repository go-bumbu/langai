package ffmpeg

import (
	"context"
	"errors"
	"image"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/sense/media"
)

// cutsVideo renders a 10 s, 25 fps, 320×240 video with two hard cuts: red
// for 3 s, green (lime, #00ff00: the half-bright "green" scores below the
// scene threshold against red) for 3 s, then ffmpeg's test pattern for 4 s.
// The native mpeg4 encoder keeps the test free of libx264.
func cutsVideo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	path := filepath.Join(t.TempDir(), "cuts.mp4")
	cmd := exec.Command("ffmpeg", "-v", "error", "-y",
		"-f", "lavfi", "-i", "color=c=red:s=320x240:d=3:r=25",
		"-f", "lavfi", "-i", "color=c=lime:s=320x240:d=3:r=25",
		"-f", "lavfi", "-i", "testsrc=s=320x240:d=4:r=25",
		"-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
		"-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", path)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("render test video: %v: %s", err, out)
	}
	return path
}

// dominant names the colour a frame mostly shows, by its centre pixel.
func dominant(img image.Image) string {
	b := img.Bounds()
	r, g, bl, _ := img.At(b.Min.X+b.Dx()/2, b.Min.Y+b.Dy()/2).RGBA()
	switch {
	case r > 0x8000 && g < 0x4000 && bl < 0x4000:
		return "red"
	case g > 0x8000 && r < 0x4000 && bl < 0x4000:
		return "green"
	}
	return "other"
}

func near(d, want time.Duration) bool {
	return d >= want-100*time.Millisecond && d <= want+100*time.Millisecond
}

// TestSample_TakesTheCutsWithTheirTimes: the first frame, the two cuts, and
// nothing else within the 5 s interval; the colour of each frame proves the
// pixels and the timestamps line up.
func TestSample_TakesTheCutsWithTheirTimes(t *testing.T) {
	path := cutsVideo(t)
	s, _ := New(Config{})
	frames, err := s.Sample(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 3 {
		t.Fatalf("got %d frames at %v, want 3 (start and two cuts)", len(frames), times(frames))
	}
	if !near(frames[0].At, 0) || dominant(frames[0].Image) != "red" {
		t.Errorf("frame 0 at %v is %s, want red at 0s", frames[0].At, dominant(frames[0].Image))
	}
	if !near(frames[1].At, 3*time.Second) || dominant(frames[1].Image) != "green" {
		t.Errorf("frame 1 at %v is %s, want green at 3s", frames[1].At, dominant(frames[1].Image))
	}
	if !near(frames[2].At, 6*time.Second) || dominant(frames[2].Image) != "other" {
		t.Errorf("frame 2 at %v is %s, want the test pattern at 6s", frames[2].At, dominant(frames[2].Image))
	}
	for _, f := range frames {
		if b := f.Image.Bounds(); b.Dx() != 320 || b.Dy() != 240 {
			t.Errorf("frame at %v is %v, want the original 320x240 (below MaxSide)", f.At, b)
		}
	}
}

func TestSample_IntervalScaleAndCap(t *testing.T) {
	path := cutsVideo(t)
	// a 2 s interval adds frames between the cuts: 0, 2, 3, 5, 6, 8
	s, _ := New(Config{MaxInterval: 2 * time.Second, MaxSide: 100})
	frames, err := s.Sample(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	want := []time.Duration{0, 2 * time.Second, 3 * time.Second, 5 * time.Second, 6 * time.Second, 8 * time.Second}
	if len(frames) != len(want) {
		t.Fatalf("got frames at %v, want %v", times(frames), want)
	}
	for i, f := range frames {
		if !near(f.At, want[i]) {
			t.Errorf("frame %d at %v, want %v", i, f.At, want[i])
		}
		if b := f.Image.Bounds(); b.Dx() != 100 || b.Dy() != 75 {
			t.Errorf("frame %d is %v, want 100x75", i, b)
		}
	}
	// MaxFrames stops ffmpeg; MinInterval is raised to 10s/4 = 2.5s, so the
	// cut at 3s is kept (2.5s after 0) and the one at 6s too
	s, _ = New(Config{MaxFrames: 4})
	frames, err = s.Sample(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 3 || !near(frames[1].At, 3*time.Second) {
		t.Errorf("MaxFrames 4: frames at %v, want 0, 3s, 6s", times(frames))
	}
	s, _ = New(Config{MaxFrames: 2, MinInterval: time.Second})
	frames, err = s.Sample(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 2 {
		t.Errorf("MaxFrames 2: got %d frames at %v", len(frames), times(frames))
	}
}

func TestSample_KeyframesOnly(t *testing.T) {
	path := cutsVideo(t)
	s, _ := New(Config{KeyframesOnly: true})
	frames, err := s.Sample(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	// the encoder opens the stream with a keyframe and puts one at each cut;
	// how many more it adds is its business
	if len(frames) < 3 || !near(frames[0].At, 0) || dominant(frames[0].Image) != "red" {
		t.Errorf("frames at %v (first %s), want a red one at 0 and the cuts", times(frames), dominant(frames[0].Image))
	}
	for i := 1; i < len(frames); i++ {
		if frames[i].At <= frames[i-1].At {
			t.Errorf("frames out of order: %v", times(frames))
		}
	}
}

func TestSample_StillImageIsOneFrame(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	path := filepath.Join(t.TempDir(), "still.png")
	if out, err := exec.Command("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=lime:s=64x48:d=1", "-frames:v", "1", path).CombinedOutput(); err != nil {
		t.Fatalf("render still: %v: %s", err, out)
	}
	s, _ := New(Config{})
	frames, err := s.Sample(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 1 || frames[0].At != 0 || dominant(frames[0].Image) != "green" {
		t.Errorf("still image: frames = %v", frames)
	}
}

func TestSample_Errors(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	dir := t.TempDir()
	s, _ := New(Config{})
	ctx := context.Background()

	if _, err := s.Sample(ctx, filepath.Join(dir, "missing.mp4")); err == nil || !strings.Contains(err.Error(), "ffprobe") {
		t.Errorf("missing file: err = %v, want ffprobe's", err)
	}

	audio := filepath.Join(dir, "audio.mp4")
	if out, err := exec.Command("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=1", audio).CombinedOutput(); err != nil {
		t.Fatalf("render audio: %v: %s", err, out)
	}
	if _, err := s.Sample(ctx, audio); err == nil || !strings.Contains(err.Error(), "ffmpeg") {
		t.Errorf("audio only: err = %v, want ffmpeg's", err)
	}

	garbage := filepath.Join(dir, "garbage.mp4")
	if err := os.WriteFile(garbage, []byte("not a video"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Sample(ctx, garbage); err == nil {
		t.Error("garbage: want an error")
	}

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := s.Sample(cancelled, cutsVideo(t)); err == nil || !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("cancelled context: err = %v", err)
	}

	missing, _ := New(Config{FFmpeg: filepath.Join(dir, "no-such-ffmpeg")})
	if _, err := missing.Sample(ctx, cutsVideo(t)); err == nil {
		t.Error("missing binary: want an error")
	}
}

func times(frames []media.Frame) []time.Duration {
	out := make([]time.Duration, len(frames))
	for i, f := range frames {
		out[i] = f.At
	}
	return out
}

// soundOutlastsVideo renders 6 s of video (a keyframe every second) under
// 11 s of sound into a file of the given name: a container whose length is
// that of the sound.
func soundOutlastsVideo(t *testing.T, name string, codec ...string) string {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	path := filepath.Join(t.TempDir(), name)
	args := []string{"-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=320x240:d=6:r=25", "-f", "lavfi", "-i", "sine=d=11", "-g", "25"}
	if out, err := exec.Command("ffmpeg", append(append(args, codec...), path)...).CombinedOutput(); err != nil {
		t.Fatalf("render %s: %v: %s", name, err, out)
	}
	return path
}

// TestDuration_IsTheVideoStreams: sound that runs on after the picture makes
// the container longer than its video. Duration is the video's, in a
// container that states it per stream and in Matroska, which notes it in a
// tag, so nobody plans segments over a stretch with nothing to see.
func TestDuration_IsTheVideoStreams(t *testing.T) {
	for name, codec := range map[string][]string{
		"clip.mp4": {"-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac"},
		"clip.mkv": {"-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac"},
	} {
		t.Run(name, func(t *testing.T) {
			s, _ := New(Config{})
			d, err := s.Duration(context.Background(), soundOutlastsVideo(t, name, codec...))
			if err != nil || !near(d, 6*time.Second) {
				t.Errorf("Duration = %v, %v; want the video's 6 s, not the sound's 11 s", d, err)
			}
		})
	}
}

// TestCut_PastTheEndIsAnError: a stream copy that starts past the last
// picture is not empty, ffmpeg writes the last group of pictures again and
// exits 0. Cut must say media.ErrPastEnd instead and write nothing, so that
// a caller walking a video in segments stops there; a cut that still holds
// pictures after its start is a cut like any other. The line between the
// two is within a tenth of a second of the end.
func TestCut_PastTheEndIsAnError(t *testing.T) {
	for _, name := range []string{"clip.mp4", "clip.mkv"} {
		path := soundOutlastsVideo(t, name, "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac")
		for _, tc := range []struct {
			start   time.Duration
			pastEnd bool
		}{
			{0, false},
			{4 * time.Second, false},
			{5500 * time.Millisecond, false},
			{6 * time.Second, true},
			{8 * time.Second, true},
			{10 * time.Hour, true},
		} {
			t.Run(name+" from "+tc.start.String(), func(t *testing.T) {
				s, _ := New(Config{})
				out := filepath.Join(t.TempDir(), "seg"+filepath.Ext(name))
				err := s.Cut(context.Background(), path, tc.start, 2*time.Second, out)
				_, statErr := os.Stat(out)
				if !tc.pastEnd {
					if err != nil || statErr != nil {
						t.Errorf("Cut = %v, the file: %v; want a segment", err, statErr)
					}
					return
				}
				if !errors.Is(err, media.ErrPastEnd) {
					t.Errorf("Cut = %v, want media.ErrPastEnd", err)
				}
				if statErr == nil {
					t.Error("a cut past the end left a file")
				}
			})
		}
	}
}

// TestSample_MetadataCannotForgeAFrameLine: ffmpeg prints a file's metadata
// into the log the frame times are read from. A title that spells a showinfo
// line, with another time or another size, must change nothing: the frames
// are the three of the plain video, at their own times.
func TestSample_MetadataCannotForgeAFrameLine(t *testing.T) {
	plain := cutsVideo(t)
	for name, title := range map[string]string{
		"another time":         "showinfo n: 0 pts_time:999 s:320x240 x",
		"another size":         "showinfo n: 0 pts_time:999 s:1x1 x",
		"the filter's name":    "[Parsed_showinfo_2 @ 0x1] n:   0 pts:      0 pts_time:999 s:1x1 x",
		"on a line of its own": "x\n[Parsed_showinfo_2 @ 0x1] n:   0 pts:      0 pts_time:999 s:1x1 x",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "titled.mp4")
			cmd := exec.Command("ffmpeg", "-v", "error", "-y", "-i", plain, "-c", "copy", "-metadata", "title="+title, path)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("write the title: %v: %s", err, out)
			}
			s, _ := New(Config{})
			frames, err := s.Sample(context.Background(), path)
			if err != nil {
				t.Fatal(err)
			}
			if len(frames) != 3 || !near(frames[0].At, 0) || !near(frames[1].At, 3*time.Second) || !near(frames[2].At, 6*time.Second) {
				t.Errorf("frames at %v, want 0s, 3s and 6s", times(frames))
			}
		})
	}
}

// TestCut_CopiesAStretch: a cut of the 10 s video from 3 s for 4 s is about
// 4 s long and shows the colours of that stretch; a cut to the end runs from
// the start given to the end; the segment's timestamps start at zero.
func TestCut_CopiesAStretch(t *testing.T) {
	path := cutsVideo(t)
	s, _ := New(Config{})
	ctx := context.Background()
	if d, err := s.Duration(ctx, path); err != nil || !near(d, 10*time.Second) {
		t.Fatalf("Duration = %v, %v; want ~10 s", d, err)
	}
	out := filepath.Join(t.TempDir(), "cut.mp4")
	if err := s.Cut(ctx, path, 3*time.Second, 4*time.Second, out); err != nil {
		t.Fatal(err)
	}
	d, err := s.Duration(ctx, out)
	if err != nil || d < 3500*time.Millisecond || d > 4600*time.Millisecond {
		t.Errorf("cut duration = %v, %v; want about 4 s", d, err)
	}
	// the cut starts at the keyframe before 3 s, in the green shot that
	// runs 3 s to 6 s or just before it; its first sampled frame is at 0
	frames, err := s.Sample(ctx, out)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) == 0 || frames[0].At != 0 {
		t.Fatalf("frames of the cut = %d, first at %v", len(frames), frames[0].At)
	}
	if c := dominant(frames[len(frames)-1].Image); c != "other" {
		t.Errorf("last frame of the cut is %s, want the test pattern (the stretch past 6 s)", c)
	}
	tail := filepath.Join(t.TempDir(), "tail.mp4")
	if err := s.Cut(ctx, path, 8*time.Second, 0, tail); err != nil {
		t.Fatal(err)
	}
	if d, err := s.Duration(ctx, tail); err != nil || d < 1500*time.Millisecond || d > 2600*time.Millisecond {
		t.Errorf("tail duration = %v, %v; want about 2 s", d, err)
	}
	if err := s.Cut(ctx, filepath.Join(t.TempDir(), "missing.mp4"), 0, time.Second, out); err == nil || !strings.Contains(err.Error(), "ffmpeg") {
		t.Errorf("missing input: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if err := s.Cut(cancelled, path, 0, time.Second, out); err == nil {
		t.Error("cancelled context: want error")
	}
}

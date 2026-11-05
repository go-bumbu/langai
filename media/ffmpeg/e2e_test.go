package ffmpeg

import (
	"context"
	"image"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai/media"
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

func near(d, want time.Duration) bool { return d >= want-100*time.Millisecond && d <= want+100*time.Millisecond }

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

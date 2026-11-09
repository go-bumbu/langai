package ffmpeg

import (
	"context"
	"image"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"
	"unsafe"
)

// tinyVideo renders a 1 s, 10 fps, 64×48 test pattern and returns its bytes,
// to be written under whatever name a test needs.
func tinyVideo(t *testing.T) []byte {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	path := filepath.Join(t.TempDir(), "tiny.mp4")
	cmd := exec.Command("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=64x48:d=1:r=10",
		"-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", path)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("render test video: %v: %s", err, out)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// still renders a 64×48 PNG of one colour and writes it at path, named
// exactly so.
func still(t *testing.T, path, colour string) {
	t.Helper()
	tmp := filepath.Join(t.TempDir(), "still.png")
	if out, err := exec.Command("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c="+colour+":s=64x48:d=1", "-frames:v", "1", tmp).CombinedOutput(); err != nil {
		t.Fatalf("render still: %v: %s", err, out)
	}
	data, err := os.ReadFile(tmp)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// TestLocalPaths_AreNeverProtocolsOrOptions: ffmpeg reads "clip:final.mp4" as
// the protocol "clip" and "-out.mp4" as an option. Whatever a file is called,
// relative or absolute, Sample, Duration and Cut must read that file and
// write that file.
func TestLocalPaths_AreNeverProtocolsOrOptions(t *testing.T) {
	video := tinyVideo(t)
	dir := t.TempDir()
	t.Chdir(dir)
	if err := os.Mkdir("a:b", 0o700); err != nil {
		t.Fatal(err)
	}
	s, _ := New(Config{})
	ctx := context.Background()
	cases := []struct{ in, out string }{
		{"clip:final.mp4", "cut:final.mp4"},
		{"-dash.mp4", "-out.mp4"},
		{"sp ace 100%.mp4", "out 100%.mp4"},
		{"a%03d#b?.mp4", "out%03d#?.mp4"},
		{"pct%20x.mp4", "out%20x.mp4"},
		{"file:odd.mp4", "file:out.mp4"},
		{"http:nowhere.mp4", "pipe:1.mp4"},
		{filepath.Join("a:b", "in dir.mp4"), filepath.Join("a:b", "-out.mp4")},
	}
	for _, tc := range cases {
		if err := os.WriteFile(tc.in, video, 0o600); err != nil {
			t.Fatal(err)
		}
		forms := map[string][2]string{
			"relative": {tc.in, tc.out},
			"absolute": {filepath.Join(dir, tc.in), filepath.Join(dir, tc.out)},
		}
		for form, p := range forms {
			t.Run(form+" "+tc.in, func(t *testing.T) {
				in, out := p[0], p[1]
				frames, err := s.Sample(ctx, in)
				if err != nil || len(frames) == 0 {
					t.Fatalf("Sample(%q) = %d frames, %v", in, len(frames), err)
				}
				if d, err := s.Duration(ctx, in); err != nil || !near(d, time.Second) {
					t.Errorf("Duration(%q) = %v, %v; want ~1 s", in, d, err)
				}
				_ = os.Remove(tc.out)
				if err := s.Cut(ctx, in, 0, 0, out); err != nil {
					t.Fatalf("Cut(%q, %q): %v", in, out, err)
				}
				if st, err := os.Stat(tc.out); err != nil || st.Size() == 0 {
					t.Fatalf("Cut(%q, %q) wrote no file of that name: %v", in, out, err)
				}
				if d, err := s.Duration(ctx, out); err != nil || !near(d, time.Second) {
					t.Errorf("Duration(%q) = %v, %v; want ~1 s", out, d, err)
				}
			})
		}
	}
	// nothing was written anywhere but under the names asked for
	want := map[string]bool{"a:b": true}
	for _, tc := range cases {
		want[tc.in], want[tc.out] = true, true
	}
	err := filepath.WalkDir(".", func(path string, _ os.DirEntry, err error) error {
		if err == nil && path != "." && !want[path] {
			t.Errorf("stray file %q", path)
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestSample_StillNamedLikeASequence: the image demuxer takes "shot %d.png"
// for the pattern of a numbered sequence and would read "shot 0.png", the red
// one, in its place, or fail when there is none.
func TestSample_StillNamedLikeASequence(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	dir := t.TempDir()
	t.Chdir(dir)
	s, _ := New(Config{})
	ctx := context.Background()
	still(t, "alone %d.png", "lime")
	still(t, "shot %d.png", "lime")
	still(t, "shot 0.png", "red")
	still(t, "half 50%.png", "lime")
	for _, name := range []string{"alone %d.png", "shot %d.png", "half 50%.png", filepath.Join(dir, "shot %d.png")} {
		frames, err := s.Sample(ctx, name)
		if err != nil {
			t.Errorf("Sample(%q): %v", name, err)
			continue
		}
		if len(frames) != 1 || dominant(frames[0].Image) != "green" {
			t.Errorf("Sample(%q) = %d frames, the first %s; want the one green frame of that file", name, len(frames), dominant(frames[0].Image))
		}
	}
	// the sequence must not stand in for a file that is not there
	if err := os.Remove("shot %d.png"); err != nil {
		t.Fatal(err)
	}
	if frames, err := s.Sample(ctx, "shot %d.png"); err == nil {
		t.Errorf("Sample of a missing file read %d frames of the sequence its name spells", len(frames))
	}
	if d, err := s.Duration(ctx, "shot %d.png"); err == nil {
		t.Errorf("Duration of a missing file = %v, from the sequence its name spells", d)
	}
}

func TestEmptyPathIsAnError(t *testing.T) {
	s, _ := New(Config{})
	ctx := context.Background()
	if _, err := s.Sample(ctx, ""); err == nil {
		t.Error("Sample of an empty path: want an error")
	}
	if _, err := s.Duration(ctx, ""); err == nil {
		t.Error("Duration of an empty path: want an error")
	}
	if err := s.Cut(ctx, "", 0, 0, "out.mp4"); err == nil {
		t.Error("Cut of an empty path: want an error")
	}
	if err := s.Cut(ctx, "in.mp4", 0, 0, ""); err == nil {
		t.Error("Cut into an empty path: want an error")
	}
}

// TestSample_FramesOwnTheirPixels: each frame has an allocation of its own,
// so keeping one frame does not keep the pixels of all the others alive. Cut
// out of one buffer, a frame would start where the one before it ends.
func TestSample_FramesOwnTheirPixels(t *testing.T) {
	path := cutsVideo(t)
	s, _ := New(Config{MaxInterval: time.Second, MaxSide: 100})
	frames, err := s.Sample(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) < 2 {
		t.Fatalf("got %d frames, want several", len(frames))
	}
	pix := func(i int) []byte { return frames[i].Image.(*image.RGBA).Pix }
	for i := 1; i < len(frames); i++ {
		prev, cur := pix(i-1), pix(i)
		if len(prev) != 100*75*4 || cap(prev) != len(prev) {
			t.Fatalf("frame %d holds %d bytes (cap %d), want exactly 100x75 RGBA", i-1, len(prev), cap(prev))
		}
		end := uintptr(unsafe.Pointer(unsafe.SliceData(prev))) + uintptr(len(prev))
		if uintptr(unsafe.Pointer(unsafe.SliceData(cur))) == end {
			t.Errorf("frame %d starts where frame %d ends: the two share one buffer", i, i-1)
		}
	}
	// writing one frame leaves the next one alone
	second := slices.Clone(pix(1))
	for i := range pix(0) {
		pix(0)[i] = 0x55
	}
	if !slices.Equal(second, pix(1)) {
		t.Error("writing frame 0 changed frame 1")
	}
}

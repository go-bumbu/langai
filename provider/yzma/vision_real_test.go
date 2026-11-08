package yzma

import (
	"context"
	"image"
	"image/color"
	"image/png"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	xdraw "golang.org/x/image/draw"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/media/ffmpeg"
)

// scene renders a picture the model can tell apart from others: a filled
// disc of one colour on a light background with a dark bar below it.
func scene(size int, c color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := range size {
		for x := range size {
			img.Set(x, y, color.RGBA{235, 235, 230, 255})
		}
	}
	cx, cy, r := size/2, size*2/5, size/4
	for y := cy - r; y <= cy+r; y++ {
		for x := cx - r; x <= cx+r; x++ {
			if (x-cx)*(x-cx)+(y-cy)*(y-cy) <= r*r {
				img.Set(x, y, c)
			}
		}
	}
	for y := size * 3 / 4; y < size*3/4+size/16; y++ {
		for x := size / 6; x < size*5/6; x++ {
			img.Set(x, y, color.RGBA{40, 40, 40, 255})
		}
	}
	return img
}

func resized(img image.Image, size int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), xdraw.Src, nil)
	return dst
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// TestVisionReal runs Qwen3-VL-Embedding through llama.cpp and libmtmd: a
// picture and its resized copy land together, a different picture further
// away, a text describing a picture lands nearer to it than to the other,
// and a two-shot video comes back as two segments that each match their
// shot. On first run it downloads the model and the projector (2.7 GB) into
// SEMSEARCH_YZMA_TEST_DIR, so it only runs when SEMSEARCH_MODELS is set,
// and it needs ffmpeg and ffprobe on PATH:
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=~/.cache/semsearch-yzma go test ./provider/yzma/ -run TestVisionReal -v
func TestVisionReal(t *testing.T) {
	if os.Getenv("SEMSEARCH_MODELS") == "" {
		t.Skip("set SEMSEARCH_MODELS=1 to download and test the real model")
	}
	dir := os.Getenv("SEMSEARCH_YZMA_TEST_DIR")
	if dir == "" {
		t.Skip("SEMSEARCH_YZMA_TEST_DIR not set; skipping real-model test")
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	ctx := context.Background()
	cutter, err := ffmpeg.New(ffmpeg.Config{})
	if err != nil {
		t.Fatal(err)
	}
	v, err := NewVision(ctx, VisionConfig{
		Config: Config{
			ModelsDir:     filepath.Join(dir, "models"),
			LibDir:        filepath.Join(dir, "llama-cpp", LibTag),
			ManagedLibDir: true,
			Threads:       8,
			Logger:        slog.New(slog.NewTextHandler(os.Stderr, nil)),
		},
		SegmentLength: 4 * time.Second,
		Cutter:        cutter,
	})
	if err != nil {
		t.Fatalf("NewVision: %v", err)
	}
	defer func() { _ = v.Close() }()
	if v.Dimensions() != 2048 {
		t.Fatalf("Dimensions = %d, want 2048", v.Dimensions())
	}
	t.Logf("accel: %s", v.Accel())

	red, blue := color.RGBA{200, 30, 30, 255}, color.RGBA{30, 60, 200, 255}
	start := time.Now()
	vecs, err := v.EmbedImages(ctx, []image.Image{scene(512, red), resized(scene(512, red), 320), scene(512, blue)})
	if err != nil {
		t.Fatalf("EmbedImages: %v", err)
	}
	t.Logf("three pictures in %v", time.Since(start))
	same, other := dot(vecs[0], vecs[1]), dot(vecs[0], vecs[2])
	t.Logf("red vs resized red %.3f, red vs blue %.3f", same, other)
	if same < 0.9 || same <= other+0.1 {
		t.Errorf("a resized copy should score above 0.9 and well above a different picture: %.3f vs %.3f", same, other)
	}

	texts, err := v.Embed(ctx, langai.RoleQuery, []string{"a red circle on a light background", "a blue circle on a light background"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	redText, blueText := dot(texts[0], vecs[0]), dot(texts[0], vecs[2])
	t.Logf("'red circle' vs red %.3f, vs blue %.3f; 'blue circle' vs blue %.3f, vs red %.3f", redText, blueText, dot(texts[1], vecs[2]), dot(texts[1], vecs[0]))
	if redText <= blueText || dot(texts[1], vecs[2]) <= dot(texts[1], vecs[0]) {
		t.Errorf("a text should land nearer to the picture it describes")
	}

	checkVideoSegments(t, ctx, v, vecs)
}

// checkVideoSegments renders a 7 s clip, the red scene for 4 s then the blue
// one, and checks that segments of 4 s give one of each, matching its shot.
func checkVideoSegments(t *testing.T, ctx context.Context, v *Vision, vecs [][]float32) {
	t.Helper()
	red, blue := color.RGBA{200, 30, 30, 255}, color.RGBA{30, 60, 200, 255}
	clip := filepath.Join(t.TempDir(), "clip.mp4")
	frames := t.TempDir()
	for name, img := range map[string]*image.RGBA{"red.png": scene(320, red), "blue.png": scene(320, blue)} {
		f, err := os.Create(filepath.Join(frames, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(f, img); err != nil {
			t.Fatal(err)
		}
		_ = f.Close()
	}
	cmd := exec.Command("ffmpeg", "-v", "error", "-y",
		"-loop", "1", "-t", "4", "-i", filepath.Join(frames, "red.png"),
		"-loop", "1", "-t", "3", "-i", filepath.Join(frames, "blue.png"),
		"-filter_complex", "[0:v]fps=25[a];[1:v]fps=25[b];[a][b]concat=n=2:v=1:a=0[v]", "-map", "[v]",
		"-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", clip)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("render clip: %v: %s", err, out)
	}
	start := time.Now()
	segments, err := v.EmbedVideo(ctx, clip)
	if err != nil {
		t.Fatalf("EmbedVideo: %v", err)
	}
	t.Logf("%d segments in %v", len(segments), time.Since(start))
	if len(segments) != 2 || segments[0].Start != 0 || segments[1].Start != 4*time.Second || segments[1].End < 6*time.Second {
		t.Fatalf("segments = %+v", segments)
	}
	s0red, s0blue := dot(segments[0].Vector, vecs[0]), dot(segments[0].Vector, vecs[2])
	s1red, s1blue := dot(segments[1].Vector, vecs[0]), dot(segments[1].Vector, vecs[2])
	t.Logf("segment 0 vs red %.3f, vs blue %.3f; segment 1 vs red %.3f, vs blue %.3f", s0red, s0blue, s1red, s1blue)
	if s0red <= s0blue || s1blue <= s1red {
		t.Errorf("each segment should match its own shot")
	}
}

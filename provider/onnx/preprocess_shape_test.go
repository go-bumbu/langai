package onnx

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"runtime"
	"testing"

	"golang.org/x/image/draw"
)

// detailed is an opaque w×h picture with gradients and stripes, so that a
// resampling difference shows.
func detailed(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8(128 + 100*math.Sin(float64(x)/5)),
				G: uint8((x*255/max(w-1, 1) + y*255/max(h-1, 1)) / 2),
				B: uint8((x*31 + y*17) % 256),
				A: 255,
			})
		}
	}
	return img
}

// scaledBefore is how a picture became the model's square before the
// extreme shapes and the transparency were dealt with: the whole picture
// scaled by Kernel.Scale, replacing the destination. It is kept here as the
// reference the current preparation must still match for ordinary pictures.
func scaledBefore(p preprocessor, img image.Image) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, p.side, p.side))
	src := img.Bounds()
	if p.resize != ResizeCrop {
		p.kernel().Scale(dst, dst.Bounds(), img, src, draw.Src, nil)
		return dst
	}
	w, h := src.Dx(), src.Dy()
	sw, sh := p.shortestEdge, p.shortestEdge
	if w < h {
		sh = (h*p.shortestEdge + w/2) / w
	} else {
		sw = (w*p.shortestEdge + h/2) / h
	}
	x0, y0 := (sw-p.side)/2, (sh-p.side)/2
	p.kernel().Scale(dst, image.Rect(-x0, -y0, sw-x0, sh-y0), img, src, draw.Src, nil)
	return dst
}

// TestPreprocessor_OrdinaryPicturesAreUnchanged: the pixels a model gets for
// an opaque picture are the ones it got before, in both resize modes and
// with both filters, so the vectors already stored for a collection still
// match what the model produces.
func TestPreprocessor_OrdinaryPicturesAreUnchanged(t *testing.T) {
	for _, p := range []preprocessor{
		{side: 224, shortestEdge: 224, resize: ResizeSquash, interpolation: InterpolationBilinear},
		{side: 224, shortestEdge: 256, resize: ResizeCrop, interpolation: InterpolationBicubic},
		{side: 224, shortestEdge: 224, resize: ResizeCrop, interpolation: InterpolationBilinear},
	} {
		for _, size := range [][2]int{{1024, 768}, {768, 1024}, {640, 640}, {224, 224}, {256, 256}, {100, 60}, {333, 1000}} {
			img := detailed(size[0], size[1])
			want, got := scaledBefore(p, img), p.scaled(img)
			if !bytes.Equal(want.Pix, got.Pix) {
				worst := 0
				for i := range want.Pix {
					d := int(want.Pix[i]) - int(got.Pix[i])
					worst = max(worst, d, -d)
				}
				t.Errorf("%s/%s %dx%d: pixels moved by up to %d", p.resize, p.interpolation, size[0], size[1], worst)
			}
		}
	}
}

// TestPreprocessor_ExtremeAspectRatio: a picture one pixel wide and twenty
// thousand tall is 80 kB of pixels. Cropping it used to scale the whole
// picture to 224 by 4,480,000 through buffers of 664 MB; it must cost about
// what the square does, and still be the picture.
func TestPreprocessor_ExtremeAspectRatio(t *testing.T) {
	blue := color.RGBA{20, 40, 220, 255}
	for _, size := range [][2]int{{1, 20000}, {20000, 1}, {2, 50000}} {
		img := image.NewRGBA(image.Rect(0, 0, size[0], size[1]))
		for i := 0; i < len(img.Pix); i += 4 {
			img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = blue.R, blue.G, blue.B, blue.A
		}
		for _, p := range []preprocessor{
			{side: 224, shortestEdge: 256, resize: ResizeCrop, interpolation: InterpolationBicubic, std: [3]float32{1, 1, 1}},
			{side: 224, shortestEdge: 224, resize: ResizeSquash, interpolation: InterpolationBilinear, std: [3]float32{1, 1, 1}},
		} {
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			out, err := p.tensor([]image.Image{img})
			runtime.ReadMemStats(&after)
			if err != nil {
				t.Fatal(err)
			}
			if used := after.TotalAlloc - before.TotalAlloc; used > 16<<20 {
				t.Errorf("%dx%d %s: preparing it allocated %d MiB", size[0], size[1], p.resize, used>>20)
			}
			for _, at := range [][2]int{{0, 0}, {112, 112}, {223, 223}} {
				r, g, b := out[at[1]*224+at[0]], out[224*224+at[1]*224+at[0]], out[2*224*224+at[1]*224+at[0]]
				if !near(r, 20.0/255) || !near(g, 40.0/255) || !near(b, 220.0/255) {
					t.Errorf("%dx%d %s: pixel %v = %v %v %v, want the picture's blue", size[0], size[1], p.resize, at, r, g, b)
				}
			}
		}
	}
}

// TestPreprocessor_TransparentPNG: a PNG with an alpha channel decodes to
// non-premultiplied pixels. Scaled as they are, its transparent parts came
// out black, while the vision model's provider put them on white; both now
// compose on white, so a logo on a transparent background is the same
// picture to every model.
func TestPreprocessor_TransparentPNG(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	for y := range 64 {
		for x := range 64 {
			switch {
			case x < 32:
				src.SetNRGBA(x, y, color.NRGBA{0, 0, 0, 0}) // fully transparent
			case y < 32:
				src.SetNRGBA(x, y, color.NRGBA{255, 0, 0, 255}) // solid red
			default:
				src.SetNRGBA(x, y, color.NRGBA{0, 0, 0, 128}) // half-transparent black
			}
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	decoded, err := png.Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []preprocessor{
		{side: 16, shortestEdge: 16, resize: ResizeSquash, interpolation: InterpolationBilinear, std: [3]float32{1, 1, 1}},
		{side: 16, shortestEdge: 16, resize: ResizeCrop, interpolation: InterpolationBicubic, std: [3]float32{1, 1, 1}},
		{side: 64, shortestEdge: 64, resize: ResizeSquash, interpolation: InterpolationBilinear, std: [3]float32{1, 1, 1}}, // unscaled
	} {
		out, err := p.tensor([]image.Image{decoded})
		if err != nil {
			t.Fatal(err)
		}
		s := p.side
		rgb := func(x, y int) [3]float32 {
			return [3]float32{at(out, s, 0, x, y), at(out, s, 1, x, y), at(out, s, 2, x, y)}
		}
		if got := rgb(s/8, s/2); !near(got[0], 1) || !near(got[1], 1) || !near(got[2], 1) {
			t.Errorf("side %d %s: the transparent part is %v, want white", s, p.resize, got)
		}
		if got := rgb(s*7/8, s/8); !near(got[0], 1) || !near(got[1], 0) || !near(got[2], 0) {
			t.Errorf("side %d %s: the solid part is %v, want red", s, p.resize, got)
		}
		if got := rgb(s*7/8, s*7/8); got[0] < 0.45 || got[0] > 0.55 || got[0] != got[1] || got[1] != got[2] {
			t.Errorf("side %d %s: half-transparent black is %v, want mid grey", s, p.resize, got)
		}
	}
}

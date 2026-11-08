package onnx

import (
	"image"
	"image/color"
	"math"
	"strings"
	"testing"
)

// halves draws a w×h picture split into two colours, left and right when
// wider than tall, top and bottom otherwise.
func halves(w, h int, first, second color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			c := first
			if (w >= h && x >= w/2) || (w < h && y >= h/2) {
				c = second
			}
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

var (
	pureRed  = color.RGBA{255, 0, 0, 255}
	pureBlue = color.RGBA{0, 0, 255, 255}
)

// at reads channel c of pixel (x, y) out of a prepared picture of the given side.
func at(t []float32, side, c, x, y int) float32 { return t[c*side*side+y*side+x] }

func near(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-4 }

func TestPreprocessor_LayoutAndNormalization(t *testing.T) {
	p := preprocessor{side: 2, shortestEdge: 2, resize: ResizeSquash, interpolation: InterpolationBilinear,
		mean: [3]float32{0.1, 0.2, 0.3}, std: [3]float32{0.5, 0.25, 2}}
	solid := image.NewRGBA(image.Rect(0, 0, 3, 3))
	for i := range solid.Pix {
		solid.Pix[i] = []byte{255, 0, 51, 255}[i%4]
	}
	out, err := p.tensor([]image.Image{solid, solid})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2*3*2*2 {
		t.Fatalf("len = %d, want 2 pictures of 3x2x2", len(out))
	}
	// (255/255-0.1)/0.5, (0-0.2)/0.25, (51/255-0.3)/2, for every pixel of both pictures
	for i, want := range []float32{1.8, -0.8, -0.05} {
		for _, v := range []float32{out[i*4], out[i*4+3], out[12+i*4]} {
			if !near(v, want) {
				t.Errorf("channel %d = %v, want %v", i, v, want)
			}
		}
	}
}

func TestPreprocessor_SquashIgnoresTheAspectRatio(t *testing.T) {
	p := preprocessor{side: 4, shortestEdge: 4, resize: ResizeSquash, interpolation: InterpolationBilinear, std: [3]float32{1, 1, 1}}
	out, err := p.tensor([]image.Image{halves(200, 100, pureRed, pureBlue)})
	if err != nil {
		t.Fatal(err)
	}
	// the whole picture lands in the square: red on the left, blue on the right, top to bottom
	for _, y := range []int{0, 3} {
		if r, b := at(out, 4, 0, 0, y), at(out, 4, 2, 0, y); !near(r, 1) || !near(b, 0) {
			t.Errorf("(0,%d) = R %v B %v, want red", y, r, b)
		}
		if r, b := at(out, 4, 0, 3, y), at(out, 4, 2, 3, y); !near(r, 0) || !near(b, 1) {
			t.Errorf("(3,%d) = R %v B %v, want blue", y, r, b)
		}
	}
}

func TestPreprocessor_CropKeepsTheCentre(t *testing.T) {
	for _, interp := range []Interpolation{InterpolationBilinear, InterpolationBicubic} {
		p := preprocessor{side: 4, shortestEdge: 4, resize: ResizeCrop, interpolation: interp, std: [3]float32{1, 1, 1}}
		// landscape: scaled to 8x4, the central 4 columns are 2 red, 2 blue
		out, err := p.tensor([]image.Image{halves(400, 200, pureRed, pureBlue)})
		if err != nil {
			t.Fatal(err)
		}
		if r := at(out, 4, 0, 0, 0); !near(r, 1) {
			t.Errorf("%s landscape (0,0) R = %v, want red", interp, r)
		}
		if b := at(out, 4, 2, 3, 3); !near(b, 1) {
			t.Errorf("%s landscape (3,3) B = %v, want blue", interp, b)
		}
		// portrait: scaled to 4x8, the central 4 rows are 2 red, 2 blue
		out, err = p.tensor([]image.Image{halves(200, 400, pureRed, pureBlue)})
		if err != nil {
			t.Fatal(err)
		}
		if r := at(out, 4, 0, 3, 0); !near(r, 1) {
			t.Errorf("%s portrait (3,0) R = %v, want red", interp, r)
		}
		if b := at(out, 4, 2, 0, 3); !near(b, 1) {
			t.Errorf("%s portrait (0,3) B = %v, want blue", interp, b)
		}
	}
	// a wider shortest edge crops away more: 400x200 scaled to 16x8, the
	// central 4x4 sits across the colour boundary, so both corners are mixed
	// or pure but the square is no longer half and half at its edges
	p := preprocessor{side: 4, shortestEdge: 8, resize: ResizeCrop, interpolation: InterpolationBilinear, std: [3]float32{1, 1, 1}}
	out, err := p.tensor([]image.Image{halves(400, 200, pureRed, pureBlue)})
	if err != nil {
		t.Fatal(err)
	}
	if r, b := at(out, 4, 0, 0, 0), at(out, 4, 2, 3, 0); !near(r, 1) || !near(b, 1) {
		t.Errorf("shortest edge 8: (0,0) R = %v, (3,0) B = %v, want red and blue", r, b)
	}
}

func TestPreprocessor_SubImageAndOddFormats(t *testing.T) {
	p := preprocessor{side: 2, shortestEdge: 2, resize: ResizeSquash, interpolation: InterpolationBilinear, std: [3]float32{1, 1, 1}}
	full := halves(100, 50, pureRed, pureBlue)
	sub := full.SubImage(image.Rect(50, 0, 100, 50)) // the blue half, bounds not at the origin
	grey := image.NewGray(image.Rect(0, 0, 10, 10))
	for i := range grey.Pix {
		grey.Pix[i] = 255
	}
	out, err := p.tensor([]image.Image{sub, grey})
	if err != nil {
		t.Fatal(err)
	}
	if r, b := at(out, 2, 0, 0, 0), at(out, 2, 2, 0, 0); !near(r, 0) || !near(b, 1) {
		t.Errorf("sub-image (0,0) = R %v B %v, want blue", r, b)
	}
	if r, g, b := out[12], out[16], out[20]; !near(r, 1) || !near(g, 1) || !near(b, 1) {
		t.Errorf("white gray picture = %v %v %v, want 1 1 1", r, g, b)
	}
}

func TestPreprocessor_Errors(t *testing.T) {
	p := preprocessor{side: 2, shortestEdge: 2, resize: ResizeSquash, std: [3]float32{1, 1, 1}}
	if _, err := p.tensor(nil); err == nil {
		t.Error("no images: want an error")
	}
	if _, err := p.tensor([]image.Image{nil}); err == nil || !strings.Contains(err.Error(), "nil") {
		t.Errorf("nil image: err = %v", err)
	}
	if _, err := p.tensor([]image.Image{image.NewRGBA(image.Rect(0, 0, 0, 5))}); err == nil || !strings.Contains(err.Error(), "no pixels") {
		t.Errorf("empty image: err = %v", err)
	}
}

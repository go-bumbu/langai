package phash

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"slices"
	"testing"
)

// noise is a w×h picture of independent random colours, so that every pixel
// and every weight shows in an average.
func noise(seed uint64, w, h int) *image.RGBA {
	r := rng(seed)
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = uint8(r.intn(256))
		if i%4 == 3 {
			img.Pix[i] = 0xff
		}
	}
	return img
}

// reference is grid the slow, obvious way for an opaque picture: every pixel
// through At, every weight from the definition, no strips.
func reference(img image.Image, cols, rows int) []int64 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	shared := func(a0, a1, b0, b1 int) int64 { return int64(max(0, min(a1, b1)-max(a0, b0))) }
	sums := make([]int64, cols*rows)
	for y := range h {
		for x := range w {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			level := lumaR*int64(r>>8) + lumaG*int64(g>>8) + lumaB*int64(bl>>8)
			for i := range rows {
				wy := shared(y*rows, (y+1)*rows, i*h, (i+1)*h)
				if wy == 0 {
					continue
				}
				for j := range cols {
					sums[i*cols+j] += wy * shared(x*cols, (x+1)*cols, j*w, (j+1)*w) * level
				}
			}
		}
	}
	whole := int64(w) * int64(h) * lumaOne
	for i, s := range sums {
		sums[i] = (2*s + whole) / (2 * whole)
	}
	return sums
}

// hidden wraps a picture so that nothing can tell its concrete type: it is
// read through At, the way an image type this package has never heard of is.
type hidden struct{ image.Image }

func TestLumaWeightsAddUpToOne(t *testing.T) {
	if lumaR+lumaG+lumaB != lumaOne {
		t.Errorf("the luma weights add up to %d, want %d", lumaR+lumaG+lumaB, lumaOne)
	}
}

// The strips, the in-place read of an opaque RGBA and the conversion of every
// other type must all give the average the definition gives.
func TestGridIsTheAreaAverage(t *testing.T) {
	sizes := []struct{ w, h int }{
		{1, 1}, {2, 3}, {8, 8}, {31, 33}, {64, 64}, // stretched, or cells that split pixels
		{120, 300},          // several strips down
		{tileWidth + 4, 5},  // several strips across
		{3, tilePixels / 2}, // tall and thin
	}
	grids := []struct{ cols, rows int }{{9, 8}, {32, 32}, {17, 16}, {64, 64}}
	for si, size := range sizes {
		img := noise(uint64(si+1), size.w, size.h)
		for _, g := range grids {
			if size.w*size.h > 4096 && g.cols > 32 {
				continue // the reference is slow, and the 32×32 grid covers these
			}
			t.Run(fmt.Sprintf("%dx%d to %dx%d", size.w, size.h, g.cols, g.rows), func(t *testing.T) {
				want := reference(img, g.cols, g.rows)
				if got := grid(img, g.cols, g.rows); !slices.Equal(got, want) {
					t.Errorf("read in place: grid differs from the reference\n got %v\nwant %v", got, want)
				}
				if got := grid(hidden{img}, g.cols, g.rows); !slices.Equal(got, want) {
					t.Errorf("read through At: grid differs from the reference\n got %v\nwant %v", got, want)
				}
			})
		}
	}
}

func TestGridOfKnownPictures(t *testing.T) {
	// two pixels stretched to four cells: the cells in between are not blends
	two := image.NewGray(image.Rect(0, 0, 2, 1))
	two.Pix[0], two.Pix[1] = 10, 200
	if got, want := grid(two, 4, 1), []int64{10, 10, 200, 200}; !slices.Equal(got, want) {
		t.Errorf("2 pixels to 4 cells = %v, want %v", got, want)
	}
	// three pixels to two cells: the middle one counts half in each
	three := image.NewGray(image.Rect(0, 0, 3, 1))
	three.Pix[0], three.Pix[1], three.Pix[2] = 0, 90, 240
	if got, want := grid(three, 2, 1), []int64{30, 190}; !slices.Equal(got, want) {
		t.Errorf("3 pixels to 2 cells = %v, want %v", got, want)
	}
	// the luma of the primaries, and rounding half up: (0+255)/2 is 128
	primaries := image.NewRGBA(image.Rect(0, 0, 5, 1))
	for x, c := range []color.RGBA{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {0, 0, 0, 255}, {255, 255, 255, 255}} {
		primaries.SetRGBA(x, 0, c)
	}
	if got, want := grid(primaries, 5, 1), []int64{76, 150, 29, 0, 255}; !slices.Equal(got, want) {
		t.Errorf("primaries = %v, want %v", got, want)
	}
	if got, want := grid(primaries.SubImage(image.Rect(3, 0, 5, 1)), 1, 1), []int64{128}; !slices.Equal(got, want) {
		t.Errorf("black and white averaged = %v, want %v", got, want)
	}
}

func TestGridComposesTransparencyOnWhite(t *testing.T) {
	// fully transparent, whatever colour hides under the alpha: white
	clear := image.NewNRGBA(image.Rect(0, 0, 6, 6))
	for i := range clear.Pix {
		if i%4 != 3 {
			clear.Pix[i] = 40
		}
	}
	if got, want := grid(clear, 2, 2), []int64{255, 255, 255, 255}; !slices.Equal(got, want) {
		t.Errorf("transparent picture = %v, want white %v", got, want)
	}
	// black at alpha 128 of 255 lets 127 of the white through
	veil := image.NewNRGBA(image.Rect(0, 0, 6, 6))
	draw.Draw(veil, veil.Bounds(), image.NewUniform(color.NRGBA{A: 128}), image.Point{}, draw.Src)
	if got, want := grid(veil, 2, 2), []int64{127, 127, 127, 127}; !slices.Equal(got, want) {
		t.Errorf("half-transparent black = %v, want %v", got, want)
	}
	// premultiplied alpha composes the same way
	pre := image.NewRGBA(veil.Bounds())
	draw.Draw(pre, pre.Bounds(), veil, image.Point{}, draw.Src)
	if got, want := grid(pre, 2, 2), []int64{127, 127, 127, 127}; !slices.Equal(got, want) {
		t.Errorf("half-transparent black, premultiplied = %v, want %v", got, want)
	}
}

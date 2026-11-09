package imageutil

import (
	"image"
	"image/color"
	"math"
	"math/rand"
	"runtime"
	"testing"

	"golang.org/x/image/draw"
)

// photo is an opaque w×h picture with detail at every scale: gradients,
// stripes and noise, so that a resampling difference shows.
func photo(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	r := rand.New(rand.NewSource(int64(w)*7919 + int64(h)))
	for y := range h {
		for x := range w {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8(128 + 100*math.Sin(float64(x)/5)),
				G: uint8((x*255/max(w-1, 1) + y*255/max(h-1, 1)) / 2),
				B: uint8(r.Intn(256)),
				A: 255,
			})
		}
	}
	return img
}

// glass is photo with an alpha that runs from clear to solid, in the
// non-premultiplied form a PNG decodes to.
func glass(w, h int) *image.NRGBA {
	src := photo(w, h)
	img := image.NewNRGBA(src.Bounds())
	for y := range h {
		for x := range w {
			c := src.RGBAAt(x, y)
			img.SetNRGBA(x, y, color.NRGBA{R: c.R, G: c.G, B: c.B, A: uint8(x * 255 / max(w-1, 1))})
		}
	}
	return img
}

// jpegLike is photo in the Y'CbCr form a JPEG decodes to.
func jpegLike(w, h int) *image.YCbCr {
	src := photo(w, h)
	img := image.NewYCbCr(src.Bounds(), image.YCbCrSubsampleRatio420)
	for y := range h {
		for x := range w {
			c := src.RGBAAt(x, y)
			yy, cb, cr := color.RGBToYCbCr(c.R, c.G, c.B)
			img.Y[img.YOffset(x, y)] = yy
			img.Cb[img.COffset(x, y)] = cb
			img.Cr[img.COffset(x, y)] = cr
		}
	}
	return img
}

// maxDiff is the largest difference between two pictures' bytes.
func maxDiff(a, b *image.RGBA) int {
	worst := 0
	for i := range a.Pix {
		d := int(a.Pix[i]) - int(b.Pix[i])
		worst = max(worst, d, -d)
	}
	return worst
}

// crop is where a w×h picture lands when its shortest edge is scaled to
// edge and the central side×side is kept: the placement an image model's
// processor uses.
func crop(w, h, side, edge int) image.Rectangle {
	sw, sh := edge, edge
	if w < h {
		sh = (h*edge + w/2) / w
	} else {
		sw = (w*edge + h/2) / h
	}
	x0, y0 := (sw-side)/2, (sh-side)/2
	return image.Rect(-x0, -y0, sw-x0, sh-y0)
}

func TestScaleOnWhite_ComposesTransparencyOnWhite(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 2))
	for y := range 2 {
		src.SetNRGBA(0, y, color.NRGBA{255, 0, 0, 255}) // solid red
		src.SetNRGBA(1, y, color.NRGBA{0, 0, 255, 0})   // clear, whatever its colour
		src.SetNRGBA(2, y, color.NRGBA{0, 0, 0, 0})     // clear black
		src.SetNRGBA(3, y, color.NRGBA{0, 0, 0, 128})   // half black
	}
	// unscaled, and scaled up fourfold with both paths
	same := image.NewRGBA(src.Bounds())
	ScaleOnWhite(same, same.Bounds(), src, draw.BiLinear)
	big := image.NewRGBA(image.Rect(0, 0, 16, 8))
	ScaleOnWhite(big, big.Bounds(), src, draw.BiLinear)
	frugal := image.NewRGBA(big.Bounds())
	draw.Draw(frugal, frugal.Bounds(), image.White, image.Point{}, draw.Src)
	viaTransform(frugal, frugal.Bounds(), src, src.Bounds(), draw.BiLinear)
	for name, c := range map[string]struct {
		img  *image.RGBA
		step int // pixels per source pixel
	}{"unscaled": {same, 1}, "scaled": {big, 4}, "scaled, frugal path": {frugal, 4}} {
		at := func(col int) color.RGBA { return c.img.RGBAAt(col*c.step+c.step/2, c.step/2) }
		// the left edge, away from the clear column the scaled red blends into
		if got := c.img.RGBAAt(0, c.step/2); got.R < 250 || got.G > 5 || got.B > 5 {
			t.Errorf("%s: solid red came out %v", name, got)
		}
		// the middle of the two clear columns: white, not the black that
		// premultiplied alpha makes of it
		if got := c.img.RGBAAt(2*c.step, c.step/2); got != (color.RGBA{255, 255, 255, 255}) {
			t.Errorf("%s: a transparent pixel came out %v, want white", name, got)
		}
		if got := at(3); got.R < 120 || got.R > 135 || got.R != got.G || got.G != got.B || got.A != 255 {
			t.Errorf("%s: half-transparent black came out %v, want mid grey", name, got)
		}
	}
	// a destination that held something else is covered, not blended with
	dirty := image.NewRGBA(src.Bounds())
	draw.Draw(dirty, dirty.Bounds(), image.Black, image.Point{}, draw.Src)
	ScaleOnWhite(dirty, dirty.Bounds(), src, draw.BiLinear)
	if maxDiff(dirty, same) != 0 {
		t.Error("what the destination held before shows through")
	}
}

// TestScaleOnWhite_OpaquePicturesAreWhatScaleGives: for a picture without
// transparency the result is the one Kernel.Scale gives when it replaces the
// destination, which is how pictures were prepared before the helper
// existed; a model's vectors for ordinary pictures must not move.
func TestScaleOnWhite_OpaquePicturesAreWhatScaleGives(t *testing.T) {
	const side = 64
	for _, k := range []*draw.Kernel{draw.BiLinear, draw.CatmullRom} {
		for _, size := range [][2]int{{640, 480}, {480, 640}, {300, 300}, {64, 64}, {96, 64}, {40, 30}, {1000, 90}} {
			for name, src := range map[string]image.Image{"rgba": photo(size[0], size[1]), "ycbcr": jpegLike(size[0], size[1])} {
				for place, dr := range map[string]image.Rectangle{"squash": image.Rect(0, 0, side, side), "crop": crop(size[0], size[1], side, 72)} {
					want := image.NewRGBA(image.Rect(0, 0, side, side))
					k.Scale(want, dr, src, src.Bounds(), draw.Src, nil)
					got := image.NewRGBA(want.Bounds())
					ScaleOnWhite(got, dr, src, k)
					if d := maxDiff(got, want); d != 0 {
						t.Errorf("%s %dx%d %s, support %.0f: differs from Kernel.Scale by up to %d", name, size[0], size[1], place, k.Support, d)
					}
				}
			}
		}
	}
}

// TestScaleOnWhite_BothPathsAgree: the frugal path must give the pixels of
// the fast one, opaque or not, or a picture's vector would depend on which
// path its shape happened to take.
func TestScaleOnWhite_BothPathsAgree(t *testing.T) {
	const side = 48
	for _, k := range []*draw.Kernel{draw.BiLinear, draw.CatmullRom} {
		for _, size := range [][2]int{{400, 300}, {300, 400}, {31, 500}, {500, 31}, {7, 9}, {48, 48}} {
			for name, src := range map[string]image.Image{"rgba": photo(size[0], size[1]), "ycbcr": jpegLike(size[0], size[1]), "nrgba with alpha": glass(size[0], size[1])} {
				for place, dr := range map[string]image.Rectangle{"squash": image.Rect(0, 0, side, side), "crop": crop(size[0], size[1], side, 56)} {
					fast, frugal := image.NewRGBA(image.Rect(0, 0, side, side)), image.NewRGBA(image.Rect(0, 0, side, side))
					for _, dst := range []*image.RGBA{fast, frugal} {
						draw.Draw(dst, dst.Bounds(), image.White, image.Point{}, draw.Src)
					}
					viaScale(fast, dr, src, src.Bounds(), k)
					viaTransform(frugal, dr, src, src.Bounds(), k)
					if d := maxDiff(fast, frugal); d > 1 {
						t.Errorf("%s %dx%d %s, support %.0f: the paths differ by up to %d", name, size[0], size[1], place, k.Support, d)
					}
				}
			}
		}
	}
}

// TestScaleOnWhite_ThinPictureTakesNoMemory: a picture one pixel wide and
// twenty thousand tall, cropped the way an image model's processor crops,
// used to cost 664 MB of working memory for 80 kB of input. It must cost
// about what its output does.
func TestScaleOnWhite_ThinPictureTakesNoMemory(t *testing.T) {
	const side, edge = 224, 256
	for _, size := range [][2]int{{1, 20000}, {20000, 1}, {3, 40000}} {
		src := image.NewRGBA(image.Rect(0, 0, size[0], size[1]))
		for i := 0; i < len(src.Pix); i += 4 {
			src.Pix[i], src.Pix[i+1], src.Pix[i+2], src.Pix[i+3] = 200, 30, 30, 255
		}
		for place, dr := range map[string]image.Rectangle{"squash": image.Rect(0, 0, side, side), "crop": crop(size[0], size[1], side, edge)} {
			dst := image.NewRGBA(image.Rect(0, 0, side, side))
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			ScaleOnWhite(dst, dr, src, draw.CatmullRom)
			runtime.ReadMemStats(&after)
			if used := after.TotalAlloc - before.TotalAlloc; used > 8<<20 {
				t.Errorf("%dx%d %s: allocated %d MiB", size[0], size[1], place, used>>20)
			}
			for _, p := range []image.Point{{0, 0}, {side / 2, side / 2}, {side - 1, side - 1}} {
				if got := dst.RGBAAt(p.X, p.Y); got != (color.RGBA{200, 30, 30, 255}) {
					t.Errorf("%dx%d %s: pixel %v is %v, want the picture's one colour", size[0], size[1], place, p, got)
				}
			}
		}
	}
}

func TestScaleOnWhite_EmptyAndClipped(t *testing.T) {
	dst := image.NewRGBA(image.Rect(0, 0, 8, 8))
	draw.Draw(dst, dst.Bounds(), image.Black, image.Point{}, draw.Src)
	// nothing to draw, nowhere to draw it: dst is left alone
	ScaleOnWhite(dst, dst.Bounds(), image.NewRGBA(image.Rect(0, 0, 0, 5)), draw.BiLinear)
	ScaleOnWhite(dst, image.Rect(20, 20, 30, 30), photo(4, 4), draw.BiLinear)
	if got := dst.RGBAAt(4, 4); got != (color.RGBA{0, 0, 0, 255}) {
		t.Errorf("an empty draw changed the destination: %v", got)
	}
	// a rectangle that covers part of dst changes only that part
	ScaleOnWhite(dst, image.Rect(4, 0, 12, 8), photo(4, 4), draw.BiLinear)
	if left, right := dst.RGBAAt(1, 4), dst.RGBAAt(6, 4); left != (color.RGBA{0, 0, 0, 255}) || right == (color.RGBA{0, 0, 0, 255}) {
		t.Errorf("a partial draw: left %v, right %v", left, right)
	}
	// a source whose bounds do not start at the origin
	sub := photo(40, 40).SubImage(image.Rect(10, 10, 30, 30))
	a, b := image.NewRGBA(image.Rect(0, 0, 8, 8)), image.NewRGBA(image.Rect(0, 0, 8, 8))
	ScaleOnWhite(a, a.Bounds(), sub, draw.CatmullRom)
	draw.Draw(b, b.Bounds(), image.White, image.Point{}, draw.Src)
	viaTransform(b, b.Bounds(), sub, sub.Bounds(), draw.CatmullRom)
	if d := maxDiff(a, b); d > 1 {
		t.Errorf("a sub-image: the paths differ by up to %d", d)
	}
}

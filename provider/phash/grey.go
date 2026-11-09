package phash

import (
	"errors"
	"fmt"
	"image"
	"image/draw"
	"reflect"
)

const (
	// maxPixels is the largest picture hashed, 256 megapixels: what
	// search.MediaIndex decodes at most, and little enough that every sum
	// below fits an int64 with room to spare.
	maxPixels = 1 << 28
	// tileWidth and tilePixels bound the strip of a picture that is
	// converted to RGBA at a time (128 KiB), so memory does not grow with
	// the picture, whatever its shape.
	tileWidth  = 4096
	tilePixels = 1 << 15
)

// The Rec. 601 luma weights of red, green and blue as 16-bit fractions, the
// ones image/color's GrayModel uses; they add up to lumaOne.
const (
	lumaR   = 19595
	lumaG   = 38470
	lumaB   = 7471
	lumaOne = 1 << 16
)

// check says what is wrong with img as a picture to hash, in words that
// follow "image" or "image 3" in an error message, or nil.
func check(img image.Image) error {
	if isNil(img) {
		return errors.New("is nil")
	}
	b := img.Bounds()
	if b.Empty() {
		return errors.New("has no pixels")
	}
	w, h := int64(b.Max.X)-int64(b.Min.X), int64(b.Max.Y)-int64(b.Min.Y)
	if w > maxPixels/h {
		return fmt.Errorf("is %dx%d pixels, more than %d", w, h, maxPixels)
	}
	return nil
}

// isNil reports whether img is nil or a nil pointer in the interface, which
// would panic on its first method call.
func isNil(img image.Image) bool {
	if img == nil {
		return true
	}
	v := reflect.ValueOf(img)
	return v.Kind() == reflect.Pointer && v.IsNil()
}

// source reads a picture strip by strip as opaque 8-bit sRGB: whatever the
// picture's colour model, and composed on white where it is transparent.
type source struct {
	img    image.Image
	opaque bool        // no pixel is transparent: there is nothing to compose
	direct *image.RGBA // img, when its pixels are what a strip would hold and are read in place
	tile   *image.RGBA // where every other picture's strips are converted
}

// newSource prepares img to be read in strips of at most tw×th pixels.
func newSource(img image.Image, tw, th int) *source {
	s := &source{img: img}
	if o, ok := img.(interface{ Opaque() bool }); ok {
		s.opaque = o.Opaque()
	}
	if rgba, ok := img.(*image.RGBA); ok && s.opaque {
		s.direct = rgba
		return s
	}
	s.tile = image.NewRGBA(image.Rect(0, 0, tw, th))
	return s
}

// strip returns the part r of the picture as rows of RGBA pixels, stride
// bytes apart, valid until the next call. An opaque picture is converted as
// it is, which is what composing it on white would give.
func (s *source) strip(r image.Rectangle) (pix []uint8, stride int) {
	if s.direct != nil {
		return s.direct.Pix[s.direct.PixOffset(r.Min.X, r.Min.Y):], s.direct.Stride
	}
	dst := image.Rect(0, 0, r.Dx(), r.Dy())
	if s.opaque {
		draw.Draw(s.tile, dst, s.img, r.Min, draw.Src)
	} else {
		draw.Draw(s.tile, dst, image.White, image.Point{}, draw.Src)
		draw.Draw(s.tile, dst, s.img, r.Min, draw.Over)
	}
	return s.tile.Pix, s.tile.Stride
}

// grid shrinks (or stretches) img to cols×rows grey levels, 0 to 255, row by
// row, aspect ratio ignored. The filter is an area average (a box filter):
// each cell is the mean colour of the part of the picture it covers, a pixel
// that straddles two cells counted in each by the fraction inside it. A
// cell's level is the Rec. 601 luma of that mean, rounded half up to a whole
// level. All of it is exact in integers; the filter has no kernel to choose
// and no ringing and counts every pixel once whatever the scale, so a
// picture and its resized copy get the same grid but for rounding. The
// rounding to whole levels keeps most of the ripple a JPEG leaves in a flat
// area from reaching the hash.
//
// The picture is read a strip at a time, so memory is one strip and the grid
// whatever the picture's size and shape. img must have passed check.
func grid(img image.Image, cols, rows int) []int64 {
	b := img.Bounds()
	w, h := int64(b.Dx()), int64(b.Dy())
	nc, nr := int64(cols), int64(rows)

	// Along each axis, in units of 1/cells of a pixel, pixel x covers
	// [x*cells, (x+1)*cells) and cell j covers [j*size, (j+1)*size); a
	// pixel's weight in a cell is the length the two share. A cell's weights
	// add up to size along each axis, so sums ends up holding each cell's
	// mean luma times w*h*lumaOne: under 2^52 for maxPixels.
	sums := make([]int64, cols*rows)
	line := make([]int64, cols)
	tw := min(b.Dx(), tileWidth)
	th := max(1, min(b.Dy(), tilePixels/tw))
	src := newSource(img, tw, th)
	for y0 := 0; y0 < b.Dy(); y0 += th {
		y1 := min(y0+th, b.Dy())
		for x0 := 0; x0 < b.Dx(); x0 += tw {
			x1 := min(x0+tw, b.Dx())
			pix, stride := src.strip(image.Rect(x0, y0, x1, y1).Add(b.Min))
			// the cells this strip's columns reach
			j0, j1 := int64(x0)*nc/w, (int64(x1)*nc-1)/w
			for y := int64(y0); y < int64(y1); y++ {
				px := pix[(y-int64(y0))*int64(stride):]
				for j := j0; j <= j1; j++ {
					line[j] = cellSum(px, int64(x0), int64(x1), j, w, nc)
				}
				// the cells this row reaches, and its weight in each
				for i := y * nr / h; i <= ((y+1)*nr-1)/h; i++ {
					weight := min((i+1)*h, (y+1)*nr) - max(i*h, y*nr)
					for j := j0; j <= j1; j++ {
						sums[i*nc+j] += weight * line[j]
					}
				}
			}
		}
	}
	whole := w * h * lumaOne
	for i, s := range sums {
		sums[i] = (2*s + whole) / (2 * whole)
	}
	return sums
}

// cellSum is what one row of pixels adds to cell j: the luma, times lumaOne,
// of the pixels' colours weighted by how much of each lies inside the cell.
// px holds the RGBA of pixels x0 to x1-1 of a picture w wide that is
// averaged into cols cells; only those pixels are counted, and at least one
// of them must touch the cell.
func cellSum(px []uint8, x0, x1, j, w, cols int64) int64 {
	lo, hi := j*w, (j+1)*w
	first, last := max(lo/cols, x0), min((hi-1)/cols, x1-1)
	// the first and the last pixel may lie partly outside the cell
	weight := min(hi, (first+1)*cols) - max(lo, first*cols)
	p := px[(first-x0)*4 : (first-x0)*4+3]
	r, g, b := weight*int64(p[0]), weight*int64(p[1]), weight*int64(p[2])
	if last > first {
		weight = min(hi, (last+1)*cols) - max(lo, last*cols)
		p = px[(last-x0)*4 : (last-x0)*4+3]
		r, g, b = r+weight*int64(p[0]), g+weight*int64(p[1]), b+weight*int64(p[2])
		// the pixels between them lie wholly inside it
		var ir, ig, ib int64
		inner := px[(first+1-x0)*4 : (last-x0)*4]
		for i := 0; i+3 < len(inner); i += 4 {
			ir += int64(inner[i])
			ig += int64(inner[i+1])
			ib += int64(inner[i+2])
		}
		r, g, b = r+cols*ir, g+cols*ig, b+cols*ib
	}
	return lumaR*r + lumaG*g + lumaB*b
}

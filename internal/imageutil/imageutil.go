// Package imageutil is the picture preparation the providers share, so that
// every model sees the same pixels for the same picture: scaling with a
// resampling kernel, and transparency composed on white. It knows nothing
// about the models.
package imageutil

import (
	"image"
	"math"

	"golang.org/x/image/draw"
	"golang.org/x/image/math/f64"
)

// maxScaleBytes is how much working memory the fast scaling path may take
// before the frugal one is considered (see ScaleOnWhite).
const maxScaleBytes = 64 << 20

// ScaleOnWhite draws the whole picture src, scaled with kernel k, onto the
// rectangle dr of dst. dr is in dst's coordinates and may reach beyond dst:
// what falls outside is clipped, so a dr larger than dst crops the scaled
// picture. Where src is transparent it is composed on white, the background
// a picture is usually meant for, rather than on the black that its
// premultiplied colours amount to.
//
// The result is the one golang.org/x/image/draw's Kernel.Scale gives. That
// scales in two passes through a buffer of dr's width times src's height,
// which is fine for a photograph and absurd for a picture that is a few
// pixels wide and tens of thousands tall, scaled up to fill dr: hundreds of
// megabytes for an input of kilobytes. When the buffers would be large, and
// large against the picture itself, only the pixels inside dst are computed
// instead (Kernel.Transform), each from the source pixels under it, with
// the same weights and no buffer.
func ScaleOnWhite(dst *image.RGBA, dr image.Rectangle, src image.Image, k *draw.Kernel) {
	sr := src.Bounds()
	visible := dr.Intersect(dst.Bounds())
	if visible.Empty() || sr.Empty() {
		return
	}
	draw.Draw(dst, visible, image.White, image.Point{}, draw.Src)
	switch cost := scaleBytes(k, dr, sr); {
	case dr.Size() == sr.Size():
		draw.Draw(dst, dr, src, sr.Min, draw.Over)
	case cost > maxScaleBytes && cost > 4*pictureBytes(sr):
		viaTransform(dst, dr, src, sr, k)
	default:
		viaScale(dst, dr, src, sr, k)
	}
}

// viaScale is the two-pass scale: fast, with buffers sized by dr and sr.
func viaScale(dst *image.RGBA, dr image.Rectangle, src image.Image, sr image.Rectangle, k *draw.Kernel) {
	k.Scale(dst, dr, src, sr, draw.Over, nil)
}

// viaTransform computes only the pixels of dr that are inside dst, through
// the affine map that takes sr onto dr. It allocates next to nothing and
// costs more per pixel.
func viaTransform(dst *image.RGBA, dr image.Rectangle, src image.Image, sr image.Rectangle, k *draw.Kernel) {
	fx := float64(dr.Dx()) / float64(sr.Dx())
	fy := float64(dr.Dy()) / float64(sr.Dy())
	k.Transform(dst, f64.Aff3{
		fx, 0, float64(dr.Min.X) - fx*float64(sr.Min.X),
		0, fy, float64(dr.Min.Y) - fy*float64(sr.Min.Y),
	}, src, sr, draw.Over, nil)
}

// scaleBytes estimates what Kernel.Scale allocates to scale sr onto dr: the
// intermediate picture, four float64 per pixel of dr's width by sr's
// height, and the weights of every source pixel under each destination
// column and row.
func scaleBytes(k *draw.Kernel, dr, sr image.Rectangle) int64 {
	dw, dh, sw, sh := int64(dr.Dx()), int64(dr.Dy()), int64(sr.Dx()), int64(sr.Dy())
	weights := func(d, s int64) int64 {
		taps := 2*k.Support*math.Max(1, float64(s)/float64(d)) + 2
		return d * int64(taps) * 16
	}
	return 32*dw*sh + weights(dw, sw) + weights(dh, sh)
}

// pictureBytes is the size of a picture of the given bounds as RGBA.
func pictureBytes(r image.Rectangle) int64 {
	return 4 * int64(r.Dx()) * int64(r.Dy())
}

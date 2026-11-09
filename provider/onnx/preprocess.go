package onnx

import (
	"errors"
	"fmt"
	"image"

	"golang.org/x/image/draw"

	"github.com/go-bumbu/sense/internal/imageutil"
)

// preprocessor turns pictures into the model's input tensor the way the
// model's own image processor does: resize to the square, scale to [0, 1],
// normalize per channel, lay out as [n, 3, side, side] (planar RGB).
type preprocessor struct {
	side, shortestEdge int
	resize             Resize
	interpolation      Interpolation
	mean, std          [3]float32
}

// tensor prepares every picture into one float32 slice of
// len(images)*3*side*side values.
func (p preprocessor) tensor(images []image.Image) ([]float32, error) {
	if len(images) == 0 {
		return nil, errors.New("no images")
	}
	per := 3 * p.side * p.side
	out := make([]float32, len(images)*per)
	for i, img := range images {
		if img == nil {
			return nil, fmt.Errorf("image %d is nil", i)
		}
		if img.Bounds().Empty() {
			return nil, fmt.Errorf("image %d has no pixels", i)
		}
		p.fill(out[i*per:(i+1)*per], img)
	}
	return out, nil
}

// fill writes one prepared picture into dst (3*side*side values).
func (p preprocessor) fill(dst []float32, img image.Image) {
	square := p.scaled(img)
	plane := p.side * p.side
	for y := range p.side {
		row := square.Pix[y*square.Stride:]
		for x := range p.side {
			px := row[x*4:]
			i := y*p.side + x
			dst[i] = (float32(px[0])/255 - p.mean[0]) / p.std[0]
			dst[plane+i] = (float32(px[1])/255 - p.mean[1]) / p.std[1]
			dst[2*plane+i] = (float32(px[2])/255 - p.mean[2]) / p.std[2]
		}
	}
}

// kernel is the x/image kernel for the spec's interpolation. Both widen
// their support when downscaling, so a photo shrunk to 224 pixels is
// averaged the way PIL averages it, not sampled.
func (p preprocessor) kernel() *draw.Kernel {
	if p.interpolation == InterpolationBicubic {
		return draw.CatmullRom
	}
	return draw.BiLinear
}

// scaled returns img as the side×side RGBA square the model takes, composed
// on white where it is transparent (see imageutil.ScaleOnWhite, which the
// other providers prepare pictures with too). Only the square is computed:
// a picture of an extreme shape costs no more than any other.
func (p preprocessor) scaled(img image.Image) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, p.side, p.side))
	imageutil.ScaleOnWhite(dst, p.placement(img.Bounds()), img, p.kernel())
	return dst
}

// placement is where the whole picture, scaled, lands in the square's
// coordinates. Squashing puts it on the square itself. Cropping scales it
// so that its shortest edge is shortestEdge long and centres it on the
// square, which then shows the central side×side and clips the rest.
func (p preprocessor) placement(src image.Rectangle) image.Rectangle {
	if p.resize != ResizeCrop {
		return image.Rect(0, 0, p.side, p.side)
	}
	w, h := src.Dx(), src.Dy()
	sw, sh := p.shortestEdge, p.shortestEdge
	if w < h {
		sh = (h*p.shortestEdge + w/2) / w
	} else {
		sw = (w*p.shortestEdge + h/2) / h
	}
	x0, y0 := (sw-p.side)/2, (sh-p.side)/2
	return image.Rect(-x0, -y0, sw-x0, sh-y0)
}

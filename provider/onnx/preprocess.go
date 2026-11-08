package onnx

import (
	"errors"
	"fmt"
	"image"

	"golang.org/x/image/draw"
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

// scaler is the x/image kernel for the spec's interpolation. Both widen
// their support when downscaling, so a photo shrunk to 224 pixels is
// averaged the way PIL averages it, not sampled.
func (p preprocessor) scaler() draw.Scaler {
	if p.interpolation == InterpolationBicubic {
		return draw.CatmullRom
	}
	return draw.BiLinear
}

// scaled returns img as the side×side RGBA square the model takes.
func (p preprocessor) scaled(img image.Image) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, p.side, p.side))
	src := img.Bounds()
	if p.resize != ResizeCrop {
		p.scaler().Scale(dst, dst.Bounds(), img, src, draw.Src, nil)
		return dst
	}
	// scale so the shortest edge is shortestEdge long, then keep the centre:
	// the scaled picture is drawn into a rectangle positioned so that its
	// central side×side lands on dst, which clips the rest
	w, h := src.Dx(), src.Dy()
	sw, sh := p.shortestEdge, p.shortestEdge
	if w < h {
		sh = (h*p.shortestEdge + w/2) / w
	} else {
		sw = (w*p.shortestEdge + h/2) / h
	}
	x0, y0 := (sw-p.side)/2, (sh-p.side)/2
	p.scaler().Scale(dst, image.Rect(-x0, -y0, sw-x0, sh-y0), img, src, draw.Src, nil)
	return dst
}

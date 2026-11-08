package langaitest

import (
	"context"
	"errors"
	"image"

	"github.com/go-bumbu/langai"
)

// fakeImageCells is the thumbnail side of FakeImageEmbedder: the image is
// averaged into fakeImageCells×fakeImageCells cells of three channels each.
const fakeImageCells = 4

// FakeImageEmbedder is a deterministic ImageEmbedder for tests: it averages
// the picture into a 4×4 thumbnail of RGB values centred on mid-grey and
// normalizes it, so a resized or recompressed copy of a picture lands next to
// it and a different picture further away. It understands nothing about what
// a picture shows — it only makes "looks alike" measurable, which is what a
// test needs.
type FakeImageEmbedder struct{}

// FakeImageEmbedderModel is the model name FakeImageEmbedder reports.
const FakeImageEmbedderModel = "langaitest/fake-image-embedder"

// verify the interfaces are actually satisfied
var (
	_ langai.ImageEmbedder = FakeImageEmbedder{}
	_ langai.Identified    = FakeImageEmbedder{}
)

// Model implements langai.Identified: FakeImageEmbedderModel.
func (FakeImageEmbedder) Model() string { return FakeImageEmbedderModel }

// Dimensions implements ImageEmbedder: 4×4 cells of 3 channels.
func (FakeImageEmbedder) Dimensions() int { return fakeImageCells * fakeImageCells * 3 }

// EmbedImages implements ImageEmbedder. A nil image or one without pixels is
// an error, as a real embedder would refuse it.
func (f FakeImageEmbedder) EmbedImages(_ context.Context, images []image.Image) ([][]float32, error) {
	out := make([][]float32, len(images))
	for i, img := range images {
		if img == nil {
			return nil, errors.New("langaitest: nil image")
		}
		b := img.Bounds()
		if b.Empty() {
			return nil, errors.New("langaitest: empty image")
		}
		vec := thumbnail(img, fakeImageCells)
		langai.Normalize(vec)
		out[i] = vec
	}
	return out, nil
}

// thumbnail averages img into cells×cells cells and returns their RGB values
// in [-0.5, 0.5], row by row. Large images are sampled on a grid rather than
// read in full, which keeps the fake fast whatever the input size.
func thumbnail(img image.Image, cells int) []float32 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	stepX, stepY := max(1, w/(cells*8)), max(1, h/(cells*8))
	out := make([]float32, 0, cells*cells*3)
	for cy := range cells {
		y0, y1 := b.Min.Y+cy*h/cells, b.Min.Y+(cy+1)*h/cells
		for cx := range cells {
			x0, x1 := b.Min.X+cx*w/cells, b.Min.X+(cx+1)*w/cells
			var r, g, bl, n float64
			for y := y0; y < max(y1, y0+1); y += stepY {
				for x := x0; x < max(x1, x0+1); x += stepX {
					pr, pg, pb, _ := img.At(x, y).RGBA()
					r += float64(pr) / 0xffff
					g += float64(pg) / 0xffff
					bl += float64(pb) / 0xffff
					n++
				}
			}
			out = append(out, float32(r/n-0.5), float32(g/n-0.5), float32(bl/n-0.5))
		}
	}
	return out
}

package sensetest

import (
	"context"
	"image"
	"image/color"
	"math"
	"testing"
)

// picture draws a w×h image whose left half is one colour and right half
// another, so pictures with the same halves look alike at any size.
func picture(w, h int, left, right color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			c := left
			if x >= w/2 {
				c = right
			}
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}

func TestFakeImageEmbedder_ResizedCopyIsCloserThanAnotherPicture(t *testing.T) {
	red, blue, green := color.RGBA{200, 30, 30, 255}, color.RGBA{30, 30, 200, 255}, color.RGBA{30, 200, 30, 255}
	emb := FakeImageEmbedder{}
	vecs, err := emb.EmbedImages(context.Background(), []image.Image{
		picture(64, 48, red, blue),
		picture(640, 480, red, blue), // the same picture, ten times larger
		picture(64, 48, green, red),  // another picture
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 3 {
		t.Fatalf("got %d vectors, want 3", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != emb.Dimensions() {
			t.Fatalf("vector %d has %d dims, want %d", i, len(v), emb.Dimensions())
		}
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		if math.Abs(sum-1) > 1e-5 {
			t.Errorf("vector %d is not unit length: %f", i, sum)
		}
	}
	same, other := cosine(vecs[0], vecs[1]), cosine(vecs[0], vecs[2])
	if same < 0.99 {
		t.Errorf("resized copy scored %f, want > 0.99", same)
	}
	if other >= same {
		t.Errorf("another picture scored %f, not below the copy's %f", other, same)
	}
}

func TestFakeImageEmbedder_RefusesNilAndEmptyImages(t *testing.T) {
	emb := FakeImageEmbedder{}
	if _, err := emb.EmbedImages(context.Background(), []image.Image{nil}); err == nil {
		t.Error("nil image: want an error")
	}
	if _, err := emb.EmbedImages(context.Background(), []image.Image{image.NewRGBA(image.Rect(0, 0, 0, 0))}); err == nil {
		t.Error("empty image: want an error")
	}
	out, err := emb.EmbedImages(context.Background(), nil)
	if err != nil || len(out) != 0 {
		t.Errorf("no images: got %v, %v; want an empty result", out, err)
	}
}

// TestFakeImageEmbedder_OffsetBoundsAndOddSizes: an image whose bounds do not
// start at the origin (a sub-image) and one smaller than the cell grid must
// both embed, with the sub-image scoring as the region it shows.
func TestFakeImageEmbedder_OffsetBoundsAndOddSizes(t *testing.T) {
	red, blue := color.RGBA{200, 30, 30, 255}, color.RGBA{30, 30, 200, 255}
	full := picture(100, 50, red, blue)
	sub := full.SubImage(image.Rect(0, 0, 50, 50)) // the red half only
	tiny := picture(3, 2, red, red)                // all red, below 4×4
	emb := FakeImageEmbedder{}
	vecs, err := emb.EmbedImages(context.Background(), []image.Image{sub, tiny, full})
	if err != nil {
		t.Fatal(err)
	}
	if c := cosine(vecs[0], vecs[1]); c < 0.99 {
		t.Errorf("the red half and an all-red picture scored %f, want > 0.99", c)
	}
	if c := cosine(vecs[0], vecs[2]); c >= 0.99 {
		t.Errorf("the red half and the whole picture scored %f, want them apart", c)
	}
}

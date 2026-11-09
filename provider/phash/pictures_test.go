package phash

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"testing"

	xdraw "golang.org/x/image/draw"
)

// rng is a splitmix64 generator: the test pictures must come out the same on
// every platform and Go release, since the golden hashes are pinned to them,
// so they depend on no library's random stream and on no floating point.
type rng uint64

func (r *rng) next() uint64 {
	*r += 0x9e3779b97f4a7c15
	z := uint64(*r)
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (r *rng) intn(n int) int { return int(r.next() % uint64(n)) }

func (r *rng) colour() color.RGBA {
	return color.RGBA{R: uint8(r.intn(256)), G: uint8(r.intn(256)), B: uint8(r.intn(256)), A: 0xff}
}

// picture draws a w×h picture with the structure of a photo, different for
// every seed: a smooth colour field (random knots, interpolated), with three
// rectangles and three discs on top. Integer arithmetic only.
func picture(seed uint64, w, h int) *image.RGBA {
	r := rng(seed)
	const knotsX, knotsY = 7, 6
	var knots [knotsY][knotsX][3]int
	for y := range knots {
		for x := range knots[y] {
			for c := range knots[y][x] {
				knots[y][x][c] = r.intn(256)
			}
		}
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		fy := y * (knotsY - 1) * 256 / h
		iy, ty := fy>>8, fy&255
		for x := range w {
			fx := x * (knotsX - 1) * 256 / w
			ix, tx := fx>>8, fx&255
			px := img.Pix[y*img.Stride+x*4:]
			for c := range 3 {
				top := knots[iy][ix][c]*(256-tx) + knots[iy][ix+1][c]*tx
				bottom := knots[iy+1][ix][c]*(256-tx) + knots[iy+1][ix+1][c]*tx
				px[c] = uint8((top*(256-ty) + bottom*ty) >> 16)
			}
			px[3] = 0xff
		}
	}
	for range 3 {
		x0, y0 := r.intn(w), r.intn(h)
		rect := image.Rect(x0, y0, x0+w/10+r.intn(w/5), y0+h/10+r.intn(h/5)).Intersect(img.Bounds())
		fill(img, rect, r.colour())
	}
	for range 3 {
		cx, cy, radius, c := r.intn(w), r.intn(h), h/12+r.intn(h/6), r.colour()
		for y := max(0, cy-radius); y < min(h, cy+radius+1); y++ {
			for x := max(0, cx-radius); x < min(w, cx+radius+1); x++ {
				if (x-cx)*(x-cx)+(y-cy)*(y-cy) <= radius*radius {
					img.SetRGBA(x, y, c)
				}
			}
		}
	}
	return img
}

// fill paints rect in one colour.
func fill(img *image.RGBA, rect image.Rectangle, c color.RGBA) {
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			img.SetRGBA(x, y, c)
		}
	}
}

// jpegBytes is img encoded as a JPEG of the given quality.
func jpegBytes(t testing.TB, img image.Image, quality int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// pngBytes is img encoded as a PNG.
func pngBytes(t testing.TB, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// reencoded is img after a round trip through a JPEG of the given quality.
func reencoded(t testing.TB, img image.Image, quality int) image.Image {
	t.Helper()
	out, err := jpeg.Decode(bytes.NewReader(jpegBytes(t, img, quality)))
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// resized is img scaled to w×h with a bicubic filter, as an image editor
// would.
func resized(img image.Image, w, h int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), xdraw.Src, nil)
	return dst
}

// brightened is img with every channel raised by add levels, clipped.
func brightened(img *image.RGBA, add int) *image.RGBA {
	out := image.NewRGBA(img.Bounds())
	for i, v := range img.Pix {
		if i%4 == 3 {
			out.Pix[i] = v
			continue
		}
		out.Pix[i] = uint8(min(255, int(v)+add))
	}
	return out
}

// marked is img with a watermark-sized patch near its bottom-right corner:
// an eighth of the width by a twentieth of the height, in light grey.
func marked(img *image.RGBA) *image.RGBA {
	out := image.NewRGBA(img.Bounds())
	copy(out.Pix, img.Pix)
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	fill(out, image.Rect(w-w/8-w/40, h-h/20-h/40, w-w/40, h-h/40), color.RGBA{R: 230, G: 230, B: 230, A: 0xff})
	return out
}

// variant is one way a duplicate differs from its original.
type variant struct {
	name string
	make func(t testing.TB, img *image.RGBA) image.Image
}

// variants are the near-duplicates the package promises to keep close.
var variants = []variant{
	{"jpeg-60", func(t testing.TB, img *image.RGBA) image.Image { return reencoded(t, img, 60) }},
	{"half-size", func(_ testing.TB, img *image.RGBA) image.Image {
		return resized(img, img.Bounds().Dx()/2, img.Bounds().Dy()/2)
	}},
	{"double-size", func(_ testing.TB, img *image.RGBA) image.Image {
		return resized(img, img.Bounds().Dx()*2, img.Bounds().Dy()*2)
	}},
	{"brighter", func(_ testing.TB, img *image.RGBA) image.Image { return brightened(img, 12) }},
	{"marked", func(_ testing.TB, img *image.RGBA) image.Image { return marked(img) }},
}

// configs are the four vector spaces the package makes.
var configs = []Config{
	{Algorithm: AlgorithmPHash, Bits: 64},
	{Algorithm: AlgorithmDHash, Bits: 64},
	{Algorithm: AlgorithmPHash, Bits: 256},
	{Algorithm: AlgorithmDHash, Bits: 256},
}

// mustNew is New for a config known to be valid.
func mustNew(t testing.TB, cfg Config) *Embedder {
	t.Helper()
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// mustHash is e.Hash for a picture known to be valid.
func mustHash(t testing.TB, e *Embedder, img image.Image) Hash {
	t.Helper()
	h, err := e.Hash(img)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

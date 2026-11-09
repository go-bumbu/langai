package phash

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/go-bumbu/sense/search"
	"github.com/go-bumbu/sense/vector"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name      string
		cfg       Config
		wantModel string
		wantDims  int
		wantErr   string
	}{
		{name: "zero value is the 64-bit pHash", cfg: Config{}, wantModel: "phash-64", wantDims: 64},
		{name: "pHash 64", cfg: Config{Algorithm: AlgorithmPHash, Bits: 64}, wantModel: "phash-64", wantDims: 64},
		{name: "dHash", cfg: Config{Algorithm: AlgorithmDHash}, wantModel: "dhash-64", wantDims: 64},
		{name: "pHash 256", cfg: Config{Bits: 256}, wantModel: "phash-256", wantDims: 256},
		{name: "dHash 256", cfg: Config{Algorithm: AlgorithmDHash, Bits: 256}, wantModel: "dhash-256", wantDims: 256},
		{name: "unknown algorithm", cfg: Config{Algorithm: "ahash"}, wantErr: `unknown algorithm "ahash"`},
		{name: "unsupported size", cfg: Config{Bits: 128}, wantErr: "Bits 128 is not supported"},
		{name: "negative size", cfg: Config{Bits: -64}, wantErr: "Bits -64 is not supported"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := New(tt.cfg)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("New() error = %v, want one containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if e.Model() != tt.wantModel || e.Dimensions() != tt.wantDims {
				t.Errorf("Model() = %q, Dimensions() = %d; want %q, %d", e.Model(), e.Dimensions(), tt.wantModel, tt.wantDims)
			}
		})
	}
}

// goldenHashes pin the four vector spaces: the hashes of picture(seed, 256,
// 192) for seeds 1 to 4, which are drawn in integer arithmetic and so are
// the same pixels everywhere. A change here means the algorithm changed, and
// then every vector already in a store is in another space than the new
// ones: do not update these values, give the changed algorithm a new Model()
// id instead, so that the stores refuse to mix the two.
var goldenHashes = map[string][4]string{
	"phash-64": {"58cd4743b29a3157", "5aad333232c36b86", "6887f2b995641d0d", "0a6d3657f02a6d70"},
	"dhash-64": {"323232b19793798f", "4d37a3a1b5998889", "9c49656672331399", "8fe7f39398e8e9b9"},
	"phash-256": {
		"581fcd3747204360926d9add316057e936c9e41be41f8f9597db41af29129aca",
		"5ab0ad9b336c32cbb2d5c3066b6e96998e39696ccc6394c1333629c66ae6431e",
		"68d4878bf263b976956964241d340d9b969f4b669b63b469e494696c901bb49b",
		"0a406d9b36fe5623f0012af5459c70ead36b390ea8f3079d56e8e021bd963cf3",
	},
	"dhash-256": {
		"0400c61cde1ccf1c071c4710cf06cf20c770e01ce00ce7071f831fc300e3c0fc",
		"38e309a30737ce1fcc1fc81fc01fcc33cc33c633c707c7c3e0e0e0e3c0e3c4c3",
		"c7f8c7e0c0e338e318330c1b0c1c0f180f380f0c0f8687868386c387c3cfc7c3",
		"c0e7c07ff83f3a0f63076307a70787c7878887c0bcc0f8e2f4e3e6e3c3e3c7c3",
	},
}

func TestGoldenHashes(t *testing.T) {
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		want, ok := goldenHashes[e.Model()]
		if !ok {
			t.Errorf("%s: no golden hashes", e.Model())
		}
		for i := range want {
			seed := uint64(i + 1)
			if got := mustHash(t, e, picture(seed, 256, 192)).String(); got != want[i] {
				t.Errorf("%s: picture %d hashes to %s, want %s: see goldenHashes before changing it", e.Model(), seed, got, want[i])
			}
		}
	}
}

// The vector is the hash: its length is exactly 1 and its cosine with
// another is exactly 1 - 2*hamming/bits, with no tolerance, because every
// entry is a power of two.
func TestCosineIsOneMinusTwiceHammingOverBits(t *testing.T) {
	base := picture(1, 256, 192)
	pictures := []image.Image{base, reencoded(t, base, 60), marked(base), picture(2, 256, 192), picture(3, 256, 192), image.NewGray(image.Rect(0, 0, 8, 8))}
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		vectors, err := e.EmbedImages(context.Background(), pictures)
		if err != nil {
			t.Fatal(err)
		}
		bits := float64(e.Dimensions())
		for i, a := range vectors {
			if len(a) != e.Dimensions() {
				t.Fatalf("%s: vector %d has %d entries, want %d", e.Model(), i, len(a), e.Dimensions())
			}
			for j, b := range vectors {
				var dot float64
				for k := range a {
					dot += float64(a[k]) * float64(b[k])
				}
				hamming := mustHash(t, e, pictures[i]).Distance(mustHash(t, e, pictures[j]))
				if want := 1 - 2*float64(hamming)/bits; dot != want {
					t.Errorf("%s: cosine of pictures %d and %d = %v, want %v (hamming %d)", e.Model(), i, j, dot, want, hamming)
				}
			}
		}
	}
}

// nearLimits are the distances the package doc gives for telling duplicates
// from the rest: how many bits a copy may differ in from its original, for
// the plain copies (re-encoded, resized, brightened) and for the one with a
// mark stamped on it. The tests below log what they measure, which is where
// the numbers in the package doc come from.
var nearLimits = map[string]struct{ copied, marked int }{
	"phash-64":  {copied: 4, marked: 10},
	"dhash-64":  {copied: 4, marked: 4},
	"phash-256": {copied: 12, marked: 48},
	"dhash-256": {copied: 12, marked: 12},
}

func TestNearDuplicatesStayClose(t *testing.T) {
	const count = 24
	originals := make([]*image.RGBA, count)
	copies := make([][]image.Image, len(variants))
	for i := range originals {
		originals[i] = picture(uint64(i+1), 320, 240)
	}
	for vi, v := range variants {
		copies[vi] = make([]image.Image, count)
		for i, img := range originals {
			copies[vi][i] = v.make(t, img)
		}
	}
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		t.Run(e.Model(), func(t *testing.T) {
			for vi, v := range variants {
				limit := nearLimits[e.Model()].copied
				if v.name == "marked" {
					limit = nearLimits[e.Model()].marked
				}
				var worst, total int
				for i, img := range originals {
					d := mustHash(t, e, img).Distance(mustHash(t, e, copies[vi][i]))
					worst, total = max(worst, d), total+d
					if d > limit {
						t.Errorf("%s copy of picture %d is %d bits away, more than %d", v.name, i+1, d, limit)
					}
				}
				t.Logf("%-11s at most %2d bits away, %.2f on average (limit %d)", v.name, worst, float64(total)/count, limit)
			}
		})
	}
}

// Unrelated pictures differ in about half the bits, and none of the 79,800
// pairs comes as close as a duplicate with a mark on it.
func TestUnrelatedPicturesSitNearHalfTheBits(t *testing.T) {
	const count = 400
	pictures := make([]*image.RGBA, count)
	for i := range pictures {
		pictures[i] = picture(uint64(1000+i), 160, 120)
	}
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		t.Run(e.Model(), func(t *testing.T) {
			hashes := make([]Hash, count)
			for i, img := range pictures {
				hashes[i] = mustHash(t, e, img)
			}
			var total, pairs int
			closest := e.Dimensions()
			for i := range hashes {
				for j := i + 1; j < count; j++ {
					d := hashes[i].Distance(hashes[j])
					total, pairs, closest = total+d, pairs+1, min(closest, d)
				}
			}
			mean, half := float64(total)/float64(pairs), float64(e.Dimensions())/2
			t.Logf("%d pairs: %.2f bits apart on average (half the bits is %.0f), the closest %d", pairs, mean, half, closest)
			if mean < 0.95*half || mean > 1.05*half {
				t.Errorf("unrelated pictures are %.2f bits apart on average, want about %.0f", mean, half)
			}
			if limit := nearLimits[e.Model()].marked; closest <= limit {
				t.Errorf("two unrelated pictures are %d bits apart, within the %d of a duplicate", closest, limit)
			}
		})
	}
}

func TestFlatPictureStillGivesAValidVector(t *testing.T) {
	flats := map[string]image.Image{
		"black":       image.NewRGBA(image.Rect(0, 0, 40, 30)),
		"white":       uniform(40, 30, color.RGBA{R: 255, G: 255, B: 255, A: 255}),
		"colour":      uniform(40, 30, color.RGBA{R: 90, G: 120, B: 200, A: 255}),
		"transparent": image.NewNRGBA(image.Rect(0, 0, 40, 30)),
		"one pixel":   uniform(1, 1, color.RGBA{R: 7, G: 7, B: 7, A: 255}),
	}
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		for name, img := range flats {
			vectors, err := e.EmbedImages(context.Background(), []image.Image{img})
			if err != nil {
				t.Fatalf("%s, %s: %v", e.Model(), name, err)
			}
			if err := vector.Validate(vectors[0], e.Dimensions()); err != nil {
				t.Errorf("%s, %s: the store would refuse the vector: %v", e.Model(), name, err)
			}
			// nothing to hash: every bit is 0, which is still a direction
			if h := mustHash(t, e, img); h != (Hash{alg: e.alg, bits: e.Dimensions()}) {
				t.Errorf("%s, %s: hash %s, want all zeros", e.Model(), name, h)
			}
		}
	}
}

// uniform is a w×h picture of one colour.
func uniform(w, h int, c color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	fill(img, img.Bounds(), c)
	return img
}

// converted is src drawn into a new picture of another type.
func converted(dst draw.Image, src image.Image) draw.Image {
	draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)
	return dst
}

// ycbcr is src as the picture type a JPEG decodes to, without subsampling.
func ycbcr(src *image.RGBA) *image.YCbCr {
	b := src.Bounds()
	dst := image.NewYCbCr(b, image.YCbCrSubsampleRatio444)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := src.RGBAAt(x, y)
			i := dst.YOffset(x, y)
			dst.Y[i], dst.Cb[i], dst.Cr[i] = color.RGBToYCbCr(c.R, c.G, c.B)
		}
	}
	return dst
}

// rendered is img as 8-bit sRGB on white: the picture every image type is
// hashed as.
func rendered(img image.Image) *image.RGBA {
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), image.White, image.Point{}, draw.Src)
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Over)
	return dst
}

// Every image type hashes, and to exactly the bits of its rendering as
// 8-bit sRGB on white: the colour model, the depth, the alpha and where the
// bounds start change nothing.
func TestEveryImageTypeHashesAsItsRendering(t *testing.T) {
	src := picture(5, 200, 150)
	b := src.Bounds()
	shifted := image.NewRGBA(b.Add(image.Pt(-30, 70)))
	copy(shifted.Pix, src.Pix)
	large := image.NewRGBA(image.Rect(0, 0, 260, 190))
	draw.Draw(large, b.Add(image.Pt(40, 20)), src, b.Min, draw.Src)
	// 16 bits per channel that are not 8 bits repeated, and alpha that varies
	deep, veiled, r := image.NewRGBA64(b), image.NewNRGBA(b), rng(5)
	for i := range deep.Pix {
		deep.Pix[i] = uint8(r.intn(256))
	}
	draw.Draw(veiled, b, src, b.Min, draw.Src)
	for i := 3; i < len(veiled.Pix); i += 4 {
		veiled.Pix[i] = uint8(i / 4 % 256)
	}
	alphaYCbCr := image.NewNYCbCrA(b, image.YCbCrSubsampleRatio420)
	for y := range b.Dy() {
		for x := range b.Dx() {
			c := src.RGBAAt(x, y)
			yy, cb, cr := color.RGBToYCbCr(c.R, c.G, c.B)
			alphaYCbCr.Y[alphaYCbCr.YOffset(x, y)], alphaYCbCr.A[alphaYCbCr.AOffset(x, y)] = yy, uint8(x+y)
			alphaYCbCr.Cb[alphaYCbCr.COffset(x, y)], alphaYCbCr.Cr[alphaYCbCr.COffset(x, y)] = cb, cr
		}
	}

	tests := []struct {
		name string
		img  image.Image
	}{
		{"RGBA at another origin", shifted},
		{"RGBA SubImage", large.SubImage(b.Add(image.Pt(40, 20)))},
		{"RGBA with alpha", converted(image.NewRGBA(b), veiled)},
		{"NRGBA", converted(image.NewNRGBA(b), src)},
		{"NRGBA with alpha", veiled},
		{"RGBA64", converted(image.NewRGBA64(b), src)},
		{"RGBA64 with 16-bit detail and alpha", deep},
		{"NRGBA64", converted(image.NewNRGBA64(b), veiled)},
		{"Gray", converted(image.NewGray(b), src)},
		{"Gray16", converted(image.NewGray16(b), src)},
		{"Alpha", converted(image.NewAlpha(b), veiled)},
		{"CMYK", converted(image.NewCMYK(b), src)},
		{"Paletted", converted(image.NewPaletted(b, palette.Plan9), src)},
		{"YCbCr 4:4:4", ycbcr(src)},
		{"YCbCr 4:2:0, as a JPEG decodes", reencoded(t, src, 90)},
		{"NYCbCrA", alphaYCbCr},
		{"a type read through At", hidden{veiled}},
	}
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		// the lossless conversions of the picture are the picture
		for _, name := range []string{"RGBA at another origin", "RGBA SubImage", "NRGBA", "RGBA64"} {
			for _, tt := range tests {
				if tt.name == name && mustHash(t, e, tt.img) != mustHash(t, e, src) {
					t.Errorf("%s: %s hashes differently from the picture it holds", e.Model(), name)
				}
			}
		}
		for _, tt := range tests {
			t.Run(e.Model()+"/"+tt.name, func(t *testing.T) {
				if got, want := mustHash(t, e, tt.img), mustHash(t, e, rendered(tt.img)); got != want {
					t.Errorf("hash %s, want its rendering's %s (%d bits apart)", got, want, got.Distance(want))
				}
			})
		}
	}
}

// A transparent area hashes as the white it is composed on, whatever colour
// its pixels still carry under the alpha.
func TestTransparencyHashesAsWhite(t *testing.T) {
	src := picture(6, 200, 150)
	hole := image.Rect(20, 30, 120, 110)
	painted := image.NewRGBA(src.Bounds())
	copy(painted.Pix, src.Pix)
	fill(painted, hole, color.RGBA{R: 255, G: 255, B: 255, A: 255})

	cut := image.NewNRGBA(src.Bounds())
	draw.Draw(cut, cut.Bounds(), src, image.Point{}, draw.Src)
	for y := hole.Min.Y; y < hole.Max.Y; y++ {
		for x := hole.Min.X; x < hole.Max.X; x++ {
			cut.Pix[cut.PixOffset(x, y)+3] = 0 // the colour stays, only alpha goes
		}
	}
	// a palette whose first entry is transparent, as in a GIF: black with a hole
	palettedCut := image.NewPaletted(src.Bounds(), color.Palette{color.NRGBA{R: 10, G: 200, B: 30, A: 0}, color.Black})
	for i := range palettedCut.Pix {
		palettedCut.Pix[i] = 1
	}
	draw.Draw(palettedCut, hole, image.NewUniform(palettedCut.Palette[0]), image.Point{}, draw.Src)
	blackWithHole := uniform(src.Bounds().Dx(), src.Bounds().Dy(), color.RGBA{A: 255})
	fill(blackWithHole, hole, color.RGBA{R: 255, G: 255, B: 255, A: 255})
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		if got, want := mustHash(t, e, cut), mustHash(t, e, painted); got != want {
			t.Errorf("%s: transparent area hashes to %s, painted white to %s", e.Model(), got, want)
		}
		if got, want := mustHash(t, e, palettedCut), mustHash(t, e, blackWithHole); got != want {
			t.Errorf("%s: transparent palette entry hashes to %s, painted white to %s", e.Model(), got, want)
		}
	}
}

func TestInvalidPicturesAreErrorsNamingTheirIndex(t *testing.T) {
	good := picture(1, 32, 24)
	var nilRGBA *image.RGBA
	tests := []struct {
		name string
		img  image.Image
		want string
	}{
		{"nil", nil, "is nil"},
		{"nil pointer", nilRGBA, "is nil"},
		{"no pixels", image.NewRGBA(image.Rectangle{}), "has no pixels"},
		{"no height", image.NewRGBA(image.Rect(0, 0, 10, 0)), "has no pixels"},
		{"inverted bounds", bounded{image.Rectangle{Min: image.Pt(5, 5), Max: image.Pt(1, 1)}}, "has no pixels"},
		{"unbounded", image.NewUniform(color.White), "is 2000000000x2000000000 pixels, more than 268435456"},
		{"too many pixels", bounded{image.Rect(0, 0, 1<<14+1, 1<<14)}, "is 16385x16384 pixels, more than 268435456"},
	}
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		for _, tt := range tests {
			t.Run(e.Model()+"/"+tt.name, func(t *testing.T) {
				if _, err := e.Hash(tt.img); err == nil || !strings.Contains(err.Error(), "phash: image "+tt.want) {
					t.Errorf("Hash() error = %v, want one containing %q", err, "phash: image "+tt.want)
				}
				vectors, err := e.EmbedImages(context.Background(), []image.Image{good, good, tt.img, good})
				if err == nil || !strings.Contains(err.Error(), "phash: image 2 "+tt.want) {
					t.Errorf("EmbedImages() error = %v, want one containing %q", err, "phash: image 2 "+tt.want)
				}
				if vectors != nil {
					t.Errorf("EmbedImages() returned %d vectors with its error", len(vectors))
				}
			})
		}
	}
}

// bounded is a white picture of any bounds that holds no pixels in memory.
type bounded struct{ bounds image.Rectangle }

func (b bounded) ColorModel() color.Model { return color.RGBAModel }
func (b bounded) Bounds() image.Rectangle { return b.bounds }
func (b bounded) At(int, int) color.Color { return color.RGBA{R: 255, G: 255, B: 255, A: 255} }

func TestSmallAndThinPictures(t *testing.T) {
	gradient := func(w, h int) *image.Gray {
		img := image.NewGray(image.Rect(0, 0, w, h))
		for i := range img.Pix {
			img.Pix[i] = uint8(i * 255 / max(1, len(img.Pix)-1))
		}
		return img
	}
	tests := []struct {
		name string
		img  image.Image
	}{
		{"1x1", gradient(1, 1)},
		{"2x1", gradient(2, 1)},
		{"1x2", gradient(1, 2)},
		{"3x3", gradient(3, 3)},
		{"1x20000", gradient(1, 20000)},
		{"20000x1", gradient(20000, 1)},
		{"2x50000", gradient(2, 50000)},
	}
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		for _, tt := range tests {
			t.Run(e.Model()+"/"+tt.name, func(t *testing.T) {
				var before, after runtime.MemStats
				runtime.ReadMemStats(&before)
				vectors, err := e.EmbedImages(context.Background(), []image.Image{tt.img})
				runtime.ReadMemStats(&after)
				if err != nil {
					t.Fatal(err)
				}
				if err := vector.Validate(vectors[0], e.Dimensions()); err != nil {
					t.Error(err)
				}
				// a strip, the grid and the vector: nothing grows with the picture
				if used := after.TotalAlloc - before.TotalAlloc; used > 512<<10 {
					t.Errorf("hashing allocated %d bytes, want at most %d", used, 512<<10)
				}
			})
		}
	}
	// a gradient across the long side is structure the hash sees: the wide
	// one gets brighter to the right in every cell, the tall one in none
	d := mustNew(t, Config{Algorithm: AlgorithmDHash})
	if got := mustHash(t, d, gradient(20000, 1)).String(); got != "ffffffffffffffff" {
		t.Errorf("dHash of a 20000x1 gradient = %s, want every bit set", got)
	}
	if got := mustHash(t, d, gradient(1, 20000)).String(); got != "0000000000000000" {
		t.Errorf("dHash of a 1x20000 gradient = %s, want no bit set", got)
	}
}

func TestEmbedImagesOfNothing(t *testing.T) {
	e := mustNew(t, Config{})
	for _, images := range [][]image.Image{nil, {}} {
		vectors, err := e.EmbedImages(context.Background(), images)
		if err != nil || len(vectors) != 0 {
			t.Errorf("EmbedImages(%v) = %v, %v; want no vectors and no error", images, vectors, err)
		}
	}
}

// cancelling is a picture that cancels a context when it is read.
type cancelling struct {
	image.Image
	cancel context.CancelFunc
	reads  *int
}

func (c cancelling) Bounds() image.Rectangle {
	*c.reads++
	c.cancel()
	return c.Image.Bounds()
}

func TestEmbedImagesHonoursCancellationBetweenPictures(t *testing.T) {
	e := mustNew(t, Config{})
	good := picture(1, 32, 24)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := e.EmbedImages(ctx, []image.Image{good}); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled before the call: error = %v, want context.Canceled", err)
	}

	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	var first, second int
	vectors, err := e.EmbedImages(ctx, []image.Image{
		cancelling{good, cancel, &first},
		cancelling{good, cancel, &second},
	})
	if !errors.Is(err, context.Canceled) || vectors != nil {
		t.Errorf("cancelled during the first picture: got %d vectors and error %v, want context.Canceled", len(vectors), err)
	}
	if first == 0 || second != 0 {
		t.Errorf("the first picture was read %d times and the second %d: want the first only", first, second)
	}
}

func TestConcurrentUse(t *testing.T) {
	pictures := make([]image.Image, 8)
	for i := range pictures {
		pictures[i] = picture(uint64(i+1), 160, 120)
	}
	pictures[3] = reencoded(t, pictures[3], 80)                                                    // converted through the strip
	pictures[5] = converted(image.NewNRGBA(pictures[5].Bounds()), pictures[5])                     // composed on white
	pictures[6] = converted(image.NewPaletted(pictures[6].Bounds(), palette.WebSafe), pictures[6]) // read through At
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		want, err := e.EmbedImages(context.Background(), pictures)
		if err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		for range 8 {
			wg.Go(func() {
				got, err := e.EmbedImages(context.Background(), pictures)
				if err != nil {
					t.Error(err)
					return
				}
				for i := range got {
					for k := range got[i] {
						if got[i][k] != want[i][k] {
							t.Errorf("%s: picture %d hashed differently under concurrency", e.Model(), i)
							return
						}
					}
				}
			})
		}
		wg.Wait()
	}
}

// indexed is a MediaIndex over e and a MemStore holding six pictures as PNGs
// (picture-1 to picture-6) and picture 3 again as a JPEG (copy-of-3).
type indexed struct {
	ix       *search.MediaIndex
	store    *vector.MemStore
	pictures []*image.RGBA
	copyOf3  []byte
}

func newIndexed(t *testing.T, e *Embedder) indexed {
	t.Helper()
	ctx := context.Background()
	in := indexed{store: vector.NewMemStore(), pictures: make([]*image.RGBA, 6)}
	var err error
	if in.ix, err = search.NewMediaIndex(e, in.store, "tenant", "space", "duplicates"); err != nil {
		t.Fatal(err)
	}
	for i := range in.pictures {
		in.pictures[i] = picture(uint64(i+1), 320, 240)
		if err := in.ix.IndexImage(ctx, fmt.Sprintf("picture-%d", i+1), bytes.NewReader(pngBytes(t, in.pictures[i]))); err != nil {
			t.Fatal(err)
		}
	}
	in.copyOf3 = jpegBytes(t, in.pictures[2], 60)
	if err := in.ix.IndexImage(ctx, "copy-of-3", bytes.NewReader(in.copyOf3)); err != nil {
		t.Fatal(err)
	}
	return in
}

// The whole point: search.MediaIndex and a vector.Store work on the hashes
// as they do on a model's vectors.
func TestMediaIndexFindsTheCopy(t *testing.T) {
	ctx := context.Background()
	for _, cfg := range configs {
		e := mustNew(t, cfg)
		t.Run(e.Model(), func(t *testing.T) {
			in := newIndexed(t, e)
			// the score a plain copy keeps at least: the documented distance, as a cosine
			floor := 1 - 2*float64(nearLimits[e.Model()].copied)/float64(e.Dimensions())

			// more like picture 3, from its stored vector: the copy, then the rest far behind
			hits, err := in.ix.SearchIndexed(ctx, "picture-3", 3)
			if err != nil {
				t.Fatal(err)
			}
			if len(hits) != 3 || hits[0].DocID != "copy-of-3" || hits[0].Score < floor || hits[1].Score > 0.5 {
				t.Errorf("SearchIndexed(picture-3) = %+v, want copy-of-3 first at %.3f or more and the rest under 0.5", hits, floor)
			}

			// the same picture at half the size finds the original and the copy
			half := pngBytes(t, resized(in.pictures[2], 160, 120))
			hits, err = in.ix.SearchImage(ctx, bytes.NewReader(half), 3)
			if err != nil {
				t.Fatal(err)
			}
			if len(hits) != 3 || hits[0].Score < floor || hits[1].Score < floor || hits[2].Score > 0.5 {
				t.Fatalf("SearchImage(half-size picture 3) = %+v, want two hits at %.3f or more and the third under 0.5", hits, floor)
			}
			if found := map[string]bool{hits[0].DocID: true, hits[1].DocID: true}; !found["picture-3"] || !found["copy-of-3"] {
				t.Errorf("SearchImage(half-size picture 3) found %s and %s first, want picture-3 and copy-of-3", hits[0].DocID, hits[1].DocID)
			}

			// the copy's own bytes find the copy with a score of exactly 1
			hits, err = in.ix.SearchImage(ctx, bytes.NewReader(in.copyOf3), 1)
			if err != nil {
				t.Fatal(err)
			}
			if len(hits) != 1 || hits[0].DocID != "copy-of-3" || hits[0].Score != 1 {
				t.Errorf("SearchImage(the copy) = %+v, want copy-of-3 with score 1", hits)
			}
		})
	}
}

// Two algorithms of one size have vectors of one length and nothing else in
// common: Model is what lets the store refuse to mix them.
func TestStoreRefusesAnotherAlgorithmsVectors(t *testing.T) {
	for _, bits := range []int{64, 256} {
		in := newIndexed(t, mustNew(t, Config{Algorithm: AlgorithmPHash, Bits: bits}))
		other, err := search.NewMediaIndex(mustNew(t, Config{Algorithm: AlgorithmDHash, Bits: bits}), in.store, "tenant", "space", "duplicates")
		if err != nil {
			t.Fatal(err)
		}
		if err := other.IndexImage(context.Background(), "intruder", bytes.NewReader(in.copyOf3)); !errors.Is(err, vector.ErrModelMismatch) {
			t.Errorf("%d bits: indexing dHash vectors into a pHash collection: error = %v, want ErrModelMismatch", bits, err)
		}
	}
}

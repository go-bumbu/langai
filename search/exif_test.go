package search

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"testing"
)

var white = color.RGBA{230, 230, 230, 255}

// quadrants draws a w×h picture of four coloured quadrants.
func quadrants(w, h int, tl, tr, bl, br color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			c := map[[2]bool]color.RGBA{{false, false}: tl, {true, false}: tr, {false, true}: bl, {true, true}: br}[[2]bool{x >= w/2, y >= h/2}]
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

// exifSegment is an APP1 segment holding a TIFF header in the given byte
// order and one IFD of two entries: a Make tag, then the orientation.
func exifSegment(order binary.ByteOrder, orientation uint16) []byte {
	var tiff bytes.Buffer
	if order == binary.ByteOrder(binary.LittleEndian) {
		tiff.WriteString("II")
	} else {
		tiff.WriteString("MM")
	}
	w := func(v any) { _ = binary.Write(&tiff, order, v) }
	w(uint16(42))
	w(uint32(8)) // IFD0 right after the header
	w(uint16(2))
	w(uint16(0x010F)) // Make, ASCII, 4 bytes inline
	w(uint16(2))
	w(uint32(4))
	tiff.WriteString("abc\x00")
	w(uint16(0x0112)) // Orientation, SHORT, 1 value, left-justified
	w(uint16(3))
	w(uint32(1))
	w(orientation)
	w(uint16(0))
	w(uint32(0)) // no next IFD
	return app1(append([]byte("Exif\x00\x00"), tiff.Bytes()...))
}

// app1 wraps payload as an APP1 segment.
func app1(payload []byte) []byte {
	n := len(payload) + 2
	return append([]byte{0xFF, 0xE1, byte(n >> 8), byte(n)}, payload...)
}

// jpegWith encodes img as a JPEG with segs inserted right after SOI.
func jpegWith(t testing.TB, img image.Image, segs ...[]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95}); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()
	out := append([]byte{}, data[:2]...)
	for _, s := range segs {
		out = append(out, s...)
	}
	return append(out, data[2:]...)
}

// near reports whether two colours are within a JPEG's rounding of each
// other.
func near(a color.Color, b color.RGBA) bool {
	r, g, bl, _ := a.RGBA()
	d := func(x uint32, y uint8) bool { return int(x>>8)-int(y) < 40 && int(y)-int(x>>8) < 40 }
	return d(r, b.R) && d(g, b.G) && d(bl, b.B)
}

// TestDecodeImage_TurnsJPEGsUpright: for each EXIF orientation, a picture
// stored the way a camera stores it for that orientation decodes upright:
// red top left, green top right, blue bottom left, white bottom right, 32×16.
// The stored layouts are written out by hand, not derived from the code.
func TestDecodeImage_TurnsJPEGsUpright(t *testing.T) {
	const w, h = 32, 16
	for _, tc := range []struct {
		orientation    uint16
		tl, tr, bl, br color.RGBA
		rotated        bool // stored h×w
	}{
		{1, red, green, blue, white, false},
		{2, green, red, white, blue, false}, // mirrored
		{3, white, blue, green, red, false}, // upside down
		{4, blue, white, red, green, false}, // mirrored upside down
		{5, red, blue, green, white, true},  // transposed
		{6, green, white, red, blue, true},  // stored turned 90° counter-clockwise
		{7, white, green, blue, red, true},  // transversed
		{8, blue, red, white, green, true},  // stored turned 90° clockwise
	} {
		for _, order := range []binary.ByteOrder{binary.BigEndian, binary.LittleEndian} {
			t.Run(fmt.Sprintf("%d %v", tc.orientation, order), func(t *testing.T) {
				sw, sh := w, h
				if tc.rotated {
					sw, sh = h, w
				}
				// an XMP APP1 first, which the parser must pass over
				xmp := app1([]byte("http://ns.adobe.com/xap/1.0/\x00<x/>"))
				data := jpegWith(t, quadrants(sw, sh, tc.tl, tc.tr, tc.bl, tc.br), xmp, exifSegment(order, tc.orientation))
				img, err := decodeImage(data)
				if err != nil {
					t.Fatal(err)
				}
				if b := img.Bounds(); b.Dx() != w || b.Dy() != h {
					t.Fatalf("decoded %v, want %dx%d", b, w, h)
				}
				b := img.Bounds().Min
				for _, p := range []struct {
					x, y int
					want color.RGBA
				}{{w / 4, h / 4, red}, {3 * w / 4, h / 4, green}, {w / 4, 3 * h / 4, blue}, {3 * w / 4, 3 * h / 4, white}} {
					if got := img.At(b.X+p.x, b.Y+p.y); !near(got, p.want) {
						t.Errorf("(%d,%d) = %v, want %v", p.x, p.y, got, p.want)
					}
				}
			})
		}
	}
}

// TestJPEGOrientation_Malformed: a broken, truncated or lying EXIF block is
// read as "no orientation", never as an error or a panic.
func TestJPEGOrientation_Malformed(t *testing.T) {
	good := jpegWith(t, quadrants(8, 8, red, green, blue, white), exifSegment(binary.BigEndian, 6))
	if o := jpegOrientation(good); o != 6 {
		t.Fatalf("the well-formed block reads as %d, want 6", o)
	}
	// every truncation of the file
	for n := range len(good) {
		if o := jpegOrientation(good[:n]); o < 1 || o > 8 {
			t.Fatalf("truncated to %d bytes: orientation %d", n, o)
		}
	}
	tiff := func(b ...byte) []byte { return app1(append([]byte("Exif\x00\x00"), b...)) }
	for name, seg := range map[string][]byte{
		"not a TIFF":         tiff('X', 'X', 0, 42, 0, 0, 0, 8),
		"wrong magic":        tiff('M', 'M', 0, 43, 0, 0, 0, 8),
		"IFD beyond the end": tiff('M', 'M', 0, 42, 0xFF, 0xFF, 0xFF, 0xF0),
		"IFD of many":        tiff('M', 'M', 0, 42, 0, 0, 0, 8, 0xFF, 0xFF),
		"orientation 9":      tiff('M', 'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 9, 0, 0),
		"orientation 0":      tiff('M', 'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0),
		"not a SHORT":        tiff('M', 'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 4, 0, 0, 0, 1, 0, 6, 0, 0),
		"no value":           tiff('M', 'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 0, 0, 6, 0, 0),
		"short segment":      {0xFF, 0xE1, 0x00, 0x01},
		"segment past end":   {0xFF, 0xE1, 0xFF, 0xFF, 'E', 'x', 'i', 'f', 0, 0},
	} {
		data := jpegWith(t, quadrants(8, 8, red, green, blue, white), seg)
		if o := jpegOrientation(data); o != 1 {
			t.Errorf("%s: orientation %d, want 1", name, o)
		}
		if _, err := decodeImage(data); err != nil && name != "short segment" && name != "segment past end" {
			t.Errorf("%s: decode: %v", name, err)
		}
	}
	for name, data := range map[string][]byte{"empty": nil, "not a JPEG": []byte("\x89PNG\r\n"), "SOI only": {0xFF, 0xD8}} {
		if o := jpegOrientation(data); o != 1 {
			t.Errorf("%s: orientation %d, want 1", name, o)
		}
	}
}

// FuzzJPEGOrientation: any input reads as an orientation in 1..8, and
// turning a picture by it keeps its pixel count.
func FuzzJPEGOrientation(f *testing.F) {
	good := jpegWith(f, quadrants(8, 4, red, green, blue, white), exifSegment(binary.LittleEndian, 8))
	f.Add(good)
	f.Add(good[:40])
	f.Add([]byte{0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x10, 'E', 'x', 'i', 'f', 0, 0, 'I', 'I', 42, 0, 8, 0, 0, 0})
	f.Add([]byte{0xFF, 0xD8, 0xFF, 0xFF, 0xFF, 0xD0, 0xFF, 0xDA})
	tiny := quadrants(3, 2, red, green, blue, white)
	f.Fuzz(func(t *testing.T, data []byte) {
		o := jpegOrientation(data)
		if o < 1 || o > 8 {
			t.Fatalf("orientation %d", o)
		}
		if b := orient(tiny, o).Bounds(); b.Dx()*b.Dy() != 6 {
			t.Fatalf("orientation %d turned 3x2 into %v", o, b)
		}
	})
}

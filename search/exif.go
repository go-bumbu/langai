package search

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/draw"
)

// JPEG markers the orientation scan knows.
const (
	markerSOI  = 0xD8 // start of image
	markerEOI  = 0xD9 // end of image
	markerSOS  = 0xDA // start of scan: the image data follows
	markerAPP1 = 0xE1 // EXIF and XMP
	markerTEM  = 0x01 // a standalone marker without a length
	markerRST0 = 0xD0 // restart markers, standalone, through RST7 (0xD7)
	markerRST7 = 0xD7
)

// tagOrientation is the EXIF (TIFF) tag of the orientation, a SHORT from 1 to 8.
const tagOrientation = 0x0112

// jpegOrientation returns the EXIF orientation of the JPEG in data, 1 to 8,
// and 1 (upright) when there is none or it cannot be read: a missing,
// truncated or malformed block is no orientation, never an error. Only the
// segments before the image data are read, and every offset is checked
// against the bytes there are.
func jpegOrientation(data []byte) int {
	if len(data) < 2 || data[0] != 0xFF || data[1] != markerSOI {
		return 1
	}
	i := 2
	for i+1 < len(data) {
		if data[i] != 0xFF {
			return 1
		}
		marker := data[i+1]
		switch {
		case marker == 0xFF: // fill byte before a marker
			i++
			continue
		case marker == markerSOI || marker == markerTEM || (marker >= markerRST0 && marker <= markerRST7):
			i += 2
			continue
		case marker == markerSOS || marker == markerEOI:
			return 1
		}
		if i+4 > len(data) {
			return 1
		}
		n := int(binary.BigEndian.Uint16(data[i+2:])) // the length counts itself
		if n < 2 || i+2+n > len(data) {
			return 1
		}
		if seg := data[i+4 : i+2+n]; marker == markerAPP1 && bytes.HasPrefix(seg, []byte("Exif\x00\x00")) {
			return tiffOrientation(seg[6:])
		}
		i += 2 + n
	}
	return 1
}

// tiffOrientation reads the orientation tag of the first IFD of the TIFF
// structure an EXIF block holds; 1 when it is not there or not valid.
func tiffOrientation(t []byte) int {
	if len(t) < 8 {
		return 1
	}
	var order binary.ByteOrder
	switch string(t[:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return 1
	}
	if order.Uint16(t[2:]) != 42 {
		return 1
	}
	size := uint64(len(t))
	ifd := uint64(order.Uint32(t[4:]))
	if ifd+2 > size {
		return 1
	}
	count := uint64(order.Uint16(t[ifd:]))
	for p := ifd + 2; count > 0 && p+12 <= size; p, count = p+12, count-1 {
		if order.Uint16(t[p:]) != tagOrientation {
			continue
		}
		// a SHORT holding at least one value, left-justified in the field
		if order.Uint16(t[p+2:]) != 3 || order.Uint32(t[p+4:]) < 1 {
			return 1
		}
		if v := order.Uint16(t[p+8:]); v >= 1 && v <= 8 {
			return int(v)
		}
		return 1
	}
	return 1
}

// orient returns img as it is meant to be seen under EXIF orientation o:
// mirrored, turned or both; orientations 5 to 8 swap width and height.
// Anything but 2 to 8 returns img as it is.
func orient(img image.Image, o int) image.Image {
	if o < 2 || o > 8 {
		return img
	}
	src := toRGBA(img)
	w, h := src.Rect.Dx(), src.Rect.Dy()
	dw, dh := w, h
	if o >= 5 {
		dw, dh = h, w
	}
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for y := range dh {
		for x := range dw {
			sx, sy := orientedSource(o, x, y, w, h)
			si, di := sy*src.Stride+sx*4, y*dst.Stride+x*4
			copy(dst.Pix[di:di+4], src.Pix[si:si+4])
		}
	}
	return dst
}

// orientedSource is the pixel of the stored w×h picture that lands at
// (x, y) of the upright one under orientation o (2 to 8).
func orientedSource(o, x, y, w, h int) (int, int) {
	switch o {
	case 2: // mirrored
		return w - 1 - x, y
	case 3: // upside down
		return w - 1 - x, h - 1 - y
	case 4: // mirrored upside down
		return x, h - 1 - y
	case 5: // transposed
		return y, x
	case 6: // stored turned 90° counter-clockwise: turn it clockwise
		return y, h - 1 - x
	case 7: // transversed
		return w - 1 - y, h - 1 - x
	default: // 8, stored turned 90° clockwise: turn it counter-clockwise
		return w - 1 - y, x
	}
}

// toRGBA returns img as an *image.RGBA whose bounds start at the origin,
// converting it (fast for the decoders' YCbCr) when it is not one.
func toRGBA(img image.Image) *image.RGBA {
	if r, ok := img.(*image.RGBA); ok && r.Rect.Min == (image.Point{}) {
		return r
	}
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Rect, img, b.Min, draw.Src)
	return dst
}

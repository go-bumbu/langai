package langaitest

import (
	"context"
	"errors"
	"hash/fnv"
	"image"
	"image/color"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/go-bumbu/langai"
)

// FakeVideoSegment is the length of every segment FakeVideoEmbedder returns.
const FakeVideoSegment = 10 * time.Second

// FakeVideoEmbedderModel is the model name FakeVideoEmbedder reports.
const FakeVideoEmbedderModel = "langaitest/fake-video-embedder"

// FakeVideoEmbedder is a deterministic VideoEmbedder for tests. A video is
// a list of pictures, one per segment, each FakeVideoSegment long and
// embedded as FakeImageEmbedder embeds it, so the segments share that fake's
// space: an image query finds the segment drawn from a picture like it. The
// file is never opened; the path alone decides what comes out. With Clips
// set, a video's pictures are the ones listed for its file's base name, and
// a name not listed is an error, as a file that holds no video would be.
// With Clips nil, every path is two segments of solid colours derived from
// its base name, so different names give different videos. Set the fields
// before the first call; EmbedVideo and Calls are safe for concurrent use.
type FakeVideoEmbedder struct {
	// Clips maps a file's base name to the pictures of its segments, in
	// time order. A name mapped to no pictures is a video with no segments.
	Clips map[string][]image.Image
	// Err, when set, fails every call with it.
	Err error

	calls atomic.Int64
}

// verify the interfaces are actually satisfied
var (
	_ langai.VideoEmbedder = (*FakeVideoEmbedder)(nil)
	_ langai.Identified    = (*FakeVideoEmbedder)(nil)
)

// Model implements langai.Identified: FakeVideoEmbedderModel.
func (*FakeVideoEmbedder) Model() string { return FakeVideoEmbedderModel }

// Dimensions implements langai.VideoEmbedder: FakeImageEmbedder's.
func (*FakeVideoEmbedder) Dimensions() int { return FakeImageEmbedder{}.Dimensions() }

// Calls returns how many times EmbedVideo was called, the failed calls
// included.
func (f *FakeVideoEmbedder) Calls() int { return int(f.calls.Load()) }

// EmbedVideo implements langai.VideoEmbedder. It fails with ctx's error when
// ctx is done and with Err when that is set.
func (f *FakeVideoEmbedder) EmbedVideo(ctx context.Context, path string) ([]langai.Segment, error) {
	f.calls.Add(1)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if f.Err != nil {
		return nil, f.Err
	}
	pictures, err := f.pictures(path)
	if err != nil {
		return nil, err
	}
	vectors, err := FakeImageEmbedder{}.EmbedImages(ctx, pictures)
	if err != nil {
		return nil, err
	}
	out := make([]langai.Segment, len(pictures))
	for i, vec := range vectors {
		start := time.Duration(i) * FakeVideoSegment
		out[i] = langai.Segment{Start: start, End: start + FakeVideoSegment, Vector: vec}
	}
	return out, nil
}

// pictures returns the pictures of the video at path: Clips' entry for its
// base name, or with Clips nil two solid colours hashed from that name.
func (f *FakeVideoEmbedder) pictures(path string) ([]image.Image, error) {
	name := filepath.Base(path)
	if f.Clips != nil {
		pictures, ok := f.Clips[name]
		if !ok {
			return nil, errors.New("langaitest: not a video: " + path)
		}
		return pictures, nil
	}
	pictures := make([]image.Image, 2)
	for i := range pictures {
		h := fnv.New32a()
		_, _ = h.Write([]byte{byte(i)})
		_, _ = h.Write([]byte(name))
		sum := h.Sum32()
		pictures[i] = solid(color.RGBA{R: uint8(sum), G: uint8(sum >> 8), B: uint8(sum >> 16), A: 0xff}) //nolint:gosec // the bytes of the hash are wanted, truncation is the point
	}
	return pictures, nil
}

// solid is an 8×8 picture of one colour.
func solid(c color.RGBA) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = c.R, c.G, c.B, c.A
	}
	return img
}

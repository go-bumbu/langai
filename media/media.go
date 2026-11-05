// Package media holds the ports for video input. Sampler turns a video file
// into the Frames that stand for it, which search embeds as images;
// VideoEmbedder embeds a video as whole Segments, for models that take
// video; Cutter is the file surgery such an embedder needs. The
// implementations live in media/ffmpeg (Sampler, Cutter, over the ffmpeg
// binary) and in the providers (VideoEmbedder); SamplerFunc adapts a
// function, e.g. for tests or for an application that already has its own
// frame extraction.
package media

import (
	"context"
	"image"
	"time"
)

// Frame is one still taken from a video: the picture and the time it was
// taken at, from the start of the video.
type Frame struct {
	Image image.Image
	At    time.Duration
}

// Sampler picks the frames that represent a video. A video is a file path
// rather than a reader because the common containers need seeking. The
// implementation decides which frames to take (scene cuts, a fixed interval)
// and how many at most; the frames come back in time order. A file that
// cannot be read or holds no video is an error, not an empty result.
type Sampler interface {
	Sample(ctx context.Context, path string) ([]Frame, error)
}

// SamplerFunc adapts a function to the Sampler interface.
type SamplerFunc func(ctx context.Context, path string) ([]Frame, error)

// Sample implements Sampler by calling f.
func (f SamplerFunc) Sample(ctx context.Context, path string) ([]Frame, error) {
	return f(ctx, path)
}

// Segment is one stretch of a video embedded as a whole: where it starts
// and ends, from the start of the video, and its vector.
type Segment struct {
	Start, End time.Duration
	Vector     []float32
}

// VideoEmbedder embeds a video with a model that takes video, as one vector
// per segment of the file, in time order. The implementation decides the
// segment length (a model reads so many frames at a time) and the frame
// rate it samples at. The vectors are Dimensions() long and share the
// space of the images and texts the same model embeds, so that an image or
// a text query finds a segment. A file that cannot be read or holds no
// video is an error, not an empty result.
type VideoEmbedder interface {
	EmbedVideo(ctx context.Context, path string) ([]Segment, error)
	Dimensions() int
}

// Cutter is the file surgery a VideoEmbedder needs to work in segments:
// how long a video is, and a copy of one stretch of it into another file,
// without re-encoding where the container allows. The implementation lives
// in media/ffmpeg.
type Cutter interface {
	// Duration is the length of the video at path; 0 when the container
	// does not say.
	Duration(ctx context.Context, path string) (time.Duration, error)
	// Cut writes the stretch of path from start for length (or to the end)
	// into out, a file named like path.
	Cut(ctx context.Context, path string, start, length time.Duration, out string) error
}

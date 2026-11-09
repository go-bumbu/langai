// Package media holds the ports for decoding video. Sampler turns a video
// file into the Frames that stand for it, which search embeds as images;
// Cutter is the file surgery a langai.VideoEmbedder needs to embed a video
// in segments. The implementations live in media/ffmpeg, over the ffmpeg
// binary; SamplerFunc adapts a function, e.g. for tests or for an
// application that already has its own frame extraction. The video
// embedding port itself, langai.VideoEmbedder, is in the root package with
// the other embedder ports.
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

// Cutter is the file surgery a langai.VideoEmbedder needs to work in segments:
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

// Package media holds the port for video input: Sampler turns a video file
// into the Frames that stand for it, which search embeds as images. The
// implementation lives in media/ffmpeg, over the ffmpeg binary; SamplerFunc
// adapts a function, e.g. for tests or for an application that already has
// its own frame extraction.
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

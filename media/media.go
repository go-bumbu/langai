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
	"errors"
	"image"
	"time"
)

// MaxDuration is the longest a video is believed to be: a week. A length is
// read from a file's header, which can say anything, and a number that is
// wrong by orders of magnitude must not be computed with as if it were
// known. A Cutter reports a length beyond it as 0, unknown, and whoever
// plans work from a length (a langai.VideoEmbedder its segments) treats one
// beyond it the same way.
const MaxDuration = 7 * 24 * time.Hour

// ErrPastEnd is what Cutter.Cut returns, wrapped, for a stretch that starts
// at or past the end of the video: there is nothing to cut. Match it with
// errors.Is. A caller that walks a video in stretches stops at it, however
// long the file's header said the video was.
var ErrPastEnd = errors.New("media: the cut starts at or past the end of the video")

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
	// Duration is the length of the video at path, of its pictures and not
	// of a sound track that runs on after them; 0 when the file does not
	// say, or says more than MaxDuration.
	Duration(ctx context.Context, path string) (time.Duration, error)
	// Cut writes the stretch of path from start for length (to the end when
	// length is 0 or less) into out, whose extension names the container.
	// out is another file than path, and holds either the whole stretch or
	// what it held before. A start at or past the end of the video is
	// ErrPastEnd and writes nothing.
	Cut(ctx context.Context, path string, start, length time.Duration, out string) error
}

// Package ffmpeg is a media.Sampler over the ffmpeg and ffprobe binaries: it
// decodes a video once and keeps the frames that stand for it, the first
// frame, every scene cut and one frame per interval in between, scaled down
// to a size an image embedder is happy with. Nothing is linked in: the two
// programs are found on PATH (or named in Config) and run per video, so the
// package builds without CGO and works with whatever ffmpeg the host has,
// 5.1 or newer.
//
// A long video is bounded in two ways: no two frames closer than
// duration/MaxFrames, so the count stays near MaxFrames however many cuts
// there are, and ffmpeg stops at MaxFrames outright. Decoding is still the
// cost: scene detection looks at every frame. KeyframesOnly decodes only the
// encoder's keyframes, which sit at cuts and at a fixed interval anyway, and
// makes hours of video cheap at the price of a coarser sample.
package ffmpeg

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/go-bumbu/langai/media"
)

// Defaults for the zero values of Config.
const (
	DefaultSceneThreshold = 0.4
	DefaultMinInterval    = time.Second
	DefaultMaxInterval    = 5 * time.Second
	DefaultMaxFrames      = 64
	DefaultMaxSide        = 512
)

// maxStderr caps what is kept of ffmpeg's log: the frame lines are a few
// hundred bytes each, so a log past this is a decoder spewing errors.
const maxStderr = 32 << 20

// Config tunes which frames are taken and how they come back. Zero values
// take the defaults above.
type Config struct {
	// FFmpeg and FFprobe name the binaries; "" means "ffmpeg" and "ffprobe"
	// on PATH.
	FFmpeg  string
	FFprobe string
	// SceneThreshold is ffmpeg's scene-change score (0 to 1) above which a
	// frame starts a new shot and is taken; 0.4 catches hard cuts without
	// firing on motion. 1 or more turns scene detection off, leaving the
	// interval sampling. Ignored when KeyframesOnly is set.
	SceneThreshold float64
	// MinInterval is the shortest gap between two taken frames: a cut within
	// it of the previous frame is skipped. It is raised to duration/MaxFrames
	// so that a cut-heavy video still yields about MaxFrames frames.
	MinInterval time.Duration
	// MaxInterval is the longest gap: after that long without a cut a frame
	// is taken anyway. It is never shorter than the (raised) MinInterval.
	MaxInterval time.Duration
	// MaxFrames is the most frames per video; ffmpeg stops there.
	MaxFrames int
	// MaxSide scales each frame so that its longer side is at most this,
	// keeping the aspect ratio; a negative value keeps the original size.
	MaxSide int
	// KeyframesOnly decodes only the stream's keyframes and takes those at
	// least MinInterval (raised as above) apart. Scene scores are not used:
	// consecutive keyframes always differ.
	KeyframesOnly bool
}

// withDefaults returns cfg with every zero value replaced by its default.
func (cfg Config) withDefaults() Config {
	if cfg.FFmpeg == "" {
		cfg.FFmpeg = "ffmpeg"
	}
	if cfg.FFprobe == "" {
		cfg.FFprobe = "ffprobe"
	}
	if cfg.SceneThreshold == 0 {
		cfg.SceneThreshold = DefaultSceneThreshold
	}
	if cfg.MinInterval == 0 {
		cfg.MinInterval = DefaultMinInterval
	}
	if cfg.MaxInterval == 0 {
		cfg.MaxInterval = DefaultMaxInterval
	}
	if cfg.MaxFrames == 0 {
		cfg.MaxFrames = DefaultMaxFrames
	}
	if cfg.MaxSide == 0 {
		cfg.MaxSide = DefaultMaxSide
	}
	return cfg
}

// Sampler implements media.Sampler. It holds no state beyond its Config and
// is safe for concurrent use; every Sample runs its own ffmpeg.
type Sampler struct {
	cfg Config
}

// verify the interface is actually satisfied
var _ media.Sampler = (*Sampler)(nil)

// New returns a Sampler for cfg, with the defaults filled in, or an error
// naming the setting that makes no sense.
func New(cfg Config) (*Sampler, error) {
	if cfg.SceneThreshold < 0 {
		return nil, fmt.Errorf("ffmpeg: SceneThreshold %v is negative", cfg.SceneThreshold)
	}
	if cfg.MinInterval < 0 || cfg.MaxInterval < 0 {
		return nil, fmt.Errorf("ffmpeg: MinInterval %v and MaxInterval %v must not be negative", cfg.MinInterval, cfg.MaxInterval)
	}
	if cfg.MaxFrames < 0 {
		return nil, fmt.Errorf("ffmpeg: MaxFrames %d is negative", cfg.MaxFrames)
	}
	return &Sampler{cfg: cfg.withDefaults()}, nil
}

// Sample implements media.Sampler: it probes the video's duration, decodes
// it once and returns the frames taken, in time order, as *image.RGBA. ctx
// bounds both programs; cancelling it kills them.
func (s *Sampler) Sample(ctx context.Context, path string) ([]media.Frame, error) {
	duration, err := s.probe(ctx, path)
	if err != nil {
		return nil, err
	}
	args := s.args(path, duration)
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, s.cfg.FFmpeg, args...) //nolint:gosec // the binary is the operator's configuration and the arguments are built here, the path among them as one argv entry
	cmd.Stdout = &stdout
	cmd.Stderr = &limitedWriter{buf: &stderr, limit: maxStderr}
	if err := cmd.Run(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, fmt.Errorf("ffmpeg: %w", ctxErr)
		}
		return nil, fmt.Errorf("ffmpeg: %w: %s", err, tail(stderr.Bytes()))
	}
	infos, err := parseShowinfo(stderr.Bytes())
	if err != nil {
		return nil, err
	}
	return splitFrames(stdout.Bytes(), infos)
}

// verify the interface is actually satisfied
var _ media.Cutter = (*Sampler)(nil)

// Duration implements media.Cutter: the container's duration from ffprobe,
// 0 when it has none (a stream, a still image).
func (s *Sampler) Duration(ctx context.Context, path string) (time.Duration, error) {
	return s.probe(ctx, path)
}

// Cut implements media.Cutter: it copies the video stream of path from
// start for length (to the end when length <= 0) into out without
// re-encoding, so a cut starts at the last keyframe before start and the
// segment's own timestamps start at zero. Audio, subtitles and data are
// dropped; an MP4 or MOV gets its index at the front, so it plays from a
// pipe. ctx bounds ffmpeg; cancelling it kills the program.
func (s *Sampler) Cut(ctx context.Context, path string, start, length time.Duration, out string) error {
	args := []string{"-nostdin", "-hide_banner", "-v", "error", "-y", "-ss", seconds(start), "-i", path}
	if length > 0 {
		args = append(args, "-t", seconds(length))
	}
	args = append(args, "-map", "0:v:0", "-an", "-sn", "-dn", "-c", "copy", "-movflags", "+faststart", out)
	var stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, s.cfg.FFmpeg, args...) //nolint:gosec // the binary is the operator's configuration and the arguments are built here, the paths among them as single argv entries
	cmd.Stderr = &limitedWriter{buf: &stderr, limit: maxStderr}
	if err := cmd.Run(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return fmt.Errorf("ffmpeg: %w", ctxErr)
		}
		return fmt.Errorf("ffmpeg: %w: %s", err, tail(stderr.Bytes()))
	}
	return nil
}

// probe asks ffprobe for the container's duration; 0 when it has none (a
// stream, a still image).
func (s *Sampler) probe(ctx context.Context, path string) (time.Duration, error) {
	args := []string{"-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", "-i", path}
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, s.cfg.FFprobe, args...) //nolint:gosec // the binary is the operator's configuration and the path is one argv entry, never a shell string
	cmd.Stdout = &stdout
	cmd.Stderr = &limitedWriter{buf: &stderr, limit: maxStderr}
	if err := cmd.Run(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return 0, fmt.Errorf("ffprobe: %w", ctxErr)
		}
		return 0, fmt.Errorf("ffprobe: %w: %s", err, tail(stderr.Bytes()))
	}
	return parseDuration(stdout.String()), nil
}

// parseDuration reads ffprobe's duration line; anything but a number ("N/A",
// nothing at all) is 0, unknown.
func parseDuration(out string) time.Duration {
	secs, err := strconv.ParseFloat(strings.TrimSpace(out), 64)
	if err != nil || secs <= 0 {
		return 0
	}
	return time.Duration(secs * float64(time.Second))
}

// gaps returns the two intervals Sample uses for a video of the given
// duration: gap, the shortest distance between two taken frames, and maxI,
// the longest. gap is MinInterval raised to duration/MaxFrames so that a
// cut-heavy video still comes back with about MaxFrames frames; maxI is
// MaxInterval, never below gap.
func (s *Sampler) gaps(duration time.Duration) (gap, maxI time.Duration) {
	gap = s.cfg.MinInterval
	if duration > 0 && s.cfg.MaxFrames > 0 {
		gap = max(gap, duration/time.Duration(s.cfg.MaxFrames))
	}
	return gap, max(s.cfg.MaxInterval, gap)
}

// selectExpr is the select filter's expression: the first frame
// (prev_selected_t is NaN), a scene cut at least gap after the previous
// frame, or any frame maxI after it. With KeyframesOnly every decoded frame
// is a keyframe, so a frame gap after the previous one is taken and the
// scene score is left out.
func (s *Sampler) selectExpr(gap, maxI time.Duration) string {
	if s.cfg.KeyframesOnly {
		return "isnan(prev_selected_t)+gte(t-prev_selected_t," + seconds(gap) + ")"
	}
	expr := "isnan(prev_selected_t)+gte(t-prev_selected_t," + seconds(maxI) + ")"
	if s.cfg.SceneThreshold < 1 {
		// the scene score is only computed when the expression names it, so
		// a threshold nothing can reach leaves it out altogether
		expr += "+gt(scene," + strconv.FormatFloat(s.cfg.SceneThreshold, 'f', 3, 64) + ")*gte(t-prev_selected_t," + seconds(gap) + ")"
	}
	return expr
}

// seconds formats d as the decimal seconds ffmpeg expressions take.
func seconds(d time.Duration) string {
	return strconv.FormatFloat(d.Seconds(), 'f', 3, 64)
}

// args builds the ffmpeg command line for path: decode (keyframes only when
// asked), select, scale into the MaxSide box, log every frame that got
// through (showinfo) and write them as raw RGBA to stdout, stopping at
// MaxFrames. The three filters form one chain, so showinfo's lines come in
// the order of the frames on stdout.
func (s *Sampler) args(path string, duration time.Duration) []string {
	gap, maxI := s.gaps(duration)
	filters := []string{"select='" + s.selectExpr(gap, maxI) + "'"}
	if s.cfg.MaxSide > 0 {
		side := strconv.Itoa(s.cfg.MaxSide)
		filters = append(filters, "scale=w='min("+side+",iw)':h='min("+side+",ih)':force_original_aspect_ratio=decrease")
	}
	filters = append(filters, "showinfo")
	args := []string{"-nostdin", "-hide_banner", "-v", "info", "-nostats"}
	if s.cfg.KeyframesOnly {
		args = append(args, "-skip_frame", "nokey")
	}
	args = append(args, "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn",
		"-vf", strings.Join(filters, ","), "-fps_mode", "vfr")
	if s.cfg.MaxFrames > 0 {
		args = append(args, "-frames:v", strconv.Itoa(s.cfg.MaxFrames))
	}
	return append(args, "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1")
}

// frameInfo is what one showinfo line says about a frame.
type frameInfo struct {
	at   time.Duration
	w, h int
}

// showinfoLine matches the per-frame lines of the showinfo filter, e.g.
// "[Parsed_showinfo_2 @ 0x...] n:   1 pts:  25600 pts_time:2  ... s:320x240 ...".
var showinfoLine = regexp.MustCompile(`showinfo.*\bn: *\d+ .*\bpts_time:(\S+) .*\bs:(\d+)x(\d+)\b`)

// parseShowinfo picks the frame lines out of ffmpeg's log, in order.
func parseShowinfo(stderr []byte) ([]frameInfo, error) {
	var infos []frameInfo
	for line := range bytes.Lines(stderr) {
		m := showinfoLine.FindSubmatch(line)
		if m == nil {
			continue
		}
		secs, err := strconv.ParseFloat(string(m[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("ffmpeg: frame %d has no timestamp (%q)", len(infos), m[1])
		}
		w, _ := strconv.Atoi(string(m[2]))
		h, _ := strconv.Atoi(string(m[3]))
		if w <= 0 || h <= 0 {
			return nil, fmt.Errorf("ffmpeg: frame %d has size %dx%d", len(infos), w, h)
		}
		infos = append(infos, frameInfo{at: time.Duration(secs * float64(time.Second)), w: w, h: h})
	}
	return infos, nil
}

// splitFrames cuts the raw RGBA stream into one image per showinfo line.
// The filters may have logged more frames than ffmpeg wrote (it stops at
// MaxFrames once the encoder has them), so the lines past the data are
// ignored; data past the lines, or a partial frame, is an error.
func splitFrames(data []byte, infos []frameInfo) ([]media.Frame, error) {
	frames := make([]media.Frame, 0, len(infos))
	for _, info := range infos {
		if len(data) == 0 {
			break
		}
		n := info.w * info.h * 4
		if len(data) < n {
			return nil, fmt.Errorf("ffmpeg: frame %d is %d bytes short of %dx%d RGBA", len(frames), n-len(data), info.w, info.h)
		}
		img := &image.RGBA{Pix: data[:n:n], Stride: info.w * 4, Rect: image.Rect(0, 0, info.w, info.h)}
		frames = append(frames, media.Frame{Image: img, At: info.at})
		data = data[n:]
	}
	if len(data) != 0 {
		return nil, fmt.Errorf("ffmpeg: %d bytes of frame data beyond the %d frames logged", len(data), len(infos))
	}
	return frames, nil
}

// limitedWriter keeps the first limit bytes written and drops the rest, so a
// log that never ends cannot grow without bound.
type limitedWriter struct {
	buf   *bytes.Buffer
	limit int
}

func (w *limitedWriter) Write(p []byte) (int, error) {
	if room := w.limit - w.buf.Len(); room > 0 {
		w.buf.Write(p[:min(len(p), room)])
	}
	return len(p), nil
}

// tail returns the end of a log for an error message: the last lines, at
// most 2 KB, whitespace trimmed.
func tail(log []byte) string {
	const n = 2 << 10
	if len(log) > n {
		log = log[len(log)-n:]
	}
	return strings.TrimSpace(string(log))
}

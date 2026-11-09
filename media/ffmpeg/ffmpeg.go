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
//
// A path is always a local file, whatever it looks like: it reaches the
// programs behind ffmpeg's file: protocol, which takes the rest of the
// argument as the name, byte for byte. Bare, "clip:final.mkv" would be read
// as the protocol "clip" and "-out.mp4" as an option. And a file is only
// ever read through that protocol: a playlist, a concat list or a session
// description names other inputs, and ffmpeg would open those through
// whatever protocol they spell (http, rtp, ...), so every input is opened
// with the file protocol alone on its whitelist, and such a file fails
// instead of reaching the network.
//
// Frames are read from ffmpeg one at a time, each into an allocation of its
// own, so memory holds the frames taken and nothing else, and a frame the
// caller keeps does not keep the others alive.
package ffmpeg

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
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

const (
	// maxLog is how much of the end of a program's log is kept for an error
	// message.
	maxLog = 2 << 10
	// maxLogLine is how much of one log line is kept to look for a frame's
	// showinfo line in it, which is a few hundred bytes long.
	maxLogLine = 16 << 10
	// maxProbeOutput caps what is kept of ffprobe's answer, two short lines.
	maxProbeOutput = 64 << 10
	// maxFramePixels is the largest frame read: 1 GiB of RGBA. Anything
	// larger is a broken header, not a video.
	maxFramePixels = 1 << 28
	// defaultWaitDelay is how long the output pipes of a program that was
	// killed or has exited may stay open, held by a child it left behind,
	// before they are closed under it.
	defaultWaitDelay = 2 * time.Second
)

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
	// MaxFrames is the most frames per video; ffmpeg stops there, and is
	// killed there if it does not.
	MaxFrames int
	// MaxSide scales each frame so that its longer side is at most this,
	// keeping the aspect ratio; a negative value keeps the original size,
	// and with it 4 bytes per pixel of every frame taken in memory.
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
	// waitDelay is exec.Cmd.WaitDelay for every program run.
	waitDelay time.Duration
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
	return &Sampler{cfg: cfg.withDefaults(), waitDelay: defaultWaitDelay}, nil
}

// samplerVersion numbers how this package picks and prepares frames, beyond
// what a Config says; it goes up when a release would take other frames, or
// other pixels, from the same video and Config:
//
//	1: the gap between frames spread MaxFrames over the container's length
//	2: over the video stream's length, which is shorter when sound runs on
const samplerVersion = 2

// Fingerprint describes every setting that changes which frames come out
// and how they look, the same string for two Samplers that take the same
// frames: what search.MediaIndex folds into the hash of every video it
// indexes (it is a search.Fingerprinter), so that a change of setting
// indexes the videos again. The binaries' paths are not part of it, and
// neither is a setting the configuration makes moot: the scene threshold
// with scene detection off or with KeyframesOnly, the longest interval with
// KeyframesOnly.
func (s *Sampler) Fingerprint() string {
	scene, maxI := "off", "-"
	if !s.cfg.KeyframesOnly {
		maxI = s.cfg.MaxInterval.String()
		if s.cfg.SceneThreshold < 1 {
			scene = strconv.FormatFloat(s.cfg.SceneThreshold, 'f', 3, 64)
		}
	}
	side := "orig"
	if s.cfg.MaxSide > 0 {
		side = strconv.Itoa(s.cfg.MaxSide)
	}
	return fmt.Sprintf("ffmpeg/%d scene=%s min=%s max=%s frames=%d side=%s keyframes=%t",
		samplerVersion, scene, s.cfg.MinInterval, maxI, s.cfg.MaxFrames, side, s.cfg.KeyframesOnly)
}

// Sample implements media.Sampler: it probes the video's duration, decodes
// it once and returns the frames taken, in time order, as *image.RGBA, each
// with pixels of its own. A file no frame comes out of is an error. ctx
// bounds both programs; cancelling it kills them.
func (s *Sampler) Sample(ctx context.Context, path string) ([]media.Frame, error) {
	in, err := s.probe(ctx, path)
	if err != nil {
		return nil, err
	}
	images, log, err := s.decode(ctx, path, in)
	if err != nil {
		return nil, err
	}
	if log.err != nil {
		return nil, log.err
	}
	if len(images) == 0 {
		return nil, fmt.Errorf("ffmpeg: no frame came out of %q", path)
	}
	// The filters may have logged more frames than ffmpeg wrote (it stops at
	// MaxFrames once the encoder has them), so the lines past the frames are
	// ignored; a frame without its line has no timestamp. Frame i's line is
	// the one the filter numbered i (see frameLog.endLine).
	if len(log.infos) < len(images) {
		return nil, fmt.Errorf("ffmpeg: %d frames written, %d logged", len(images), len(log.infos))
	}
	frames := make([]media.Frame, len(images))
	for i, img := range images {
		info := log.infos[i]
		if b := img.Bounds(); b.Dx() != info.w || b.Dy() != info.h {
			return nil, fmt.Errorf("ffmpeg: frame %d is %dx%d, logged as %dx%d", i, b.Dx(), b.Dy(), info.w, info.h)
		}
		frames[i] = media.Frame{Image: img, At: info.at}
	}
	return frames, nil
}

// decode runs ffmpeg over path and reads the frames it writes, at most
// MaxFrames. ffmpeg is gone when it returns, on every path: it is killed
// when ctx is cancelled, when a frame cannot be read and when MaxFrames are
// in, and it is always waited for, which also brings in the rest of its log.
// An error ends with the end of that log.
func (s *Sampler) decode(ctx context.Context, path string, in probed) ([]*image.RGBA, *frameLog, error) {
	runCtx, kill := context.WithCancel(ctx)
	defer kill()
	log := &frameLog{}
	cmd := exec.CommandContext(runCtx, s.cfg.FFmpeg, s.args(path, in)...) //nolint:gosec // the binary is the operator's configuration and the arguments are built here, the path among them as one argv entry
	cmd.Stderr = log
	cmd.WaitDelay = s.waitDelay
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("ffmpeg: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("ffmpeg: %w", err)
	}
	images, readErr := s.readFrames(stdout)
	full := s.cfg.MaxFrames > 0 && len(images) == s.cfg.MaxFrames
	if full || readErr != nil {
		kill()
	}
	waitErr := cmd.Wait()
	switch {
	case ctx.Err() != nil:
		return nil, nil, fmt.Errorf("ffmpeg: %w", ctx.Err())
	case readErr != nil:
		return nil, nil, fmt.Errorf("ffmpeg: %w: %s", readErr, log.tail.String())
	case waitErr != nil && !full:
		// with all frames in, the only error is the kill above
		return nil, nil, fmt.Errorf("ffmpeg: %w: %s", waitErr, log.tail.String())
	}
	return images, log, nil
}

// readFrames reads the frames ffmpeg writes until it stops writing or
// MaxFrames are in.
func (s *Sampler) readFrames(r io.Reader) ([]*image.RGBA, error) {
	br := bufio.NewReader(r)
	var images []*image.RGBA
	for s.cfg.MaxFrames <= 0 || len(images) < s.cfg.MaxFrames {
		img, err := readPAM(br, s.cfg.MaxSide)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("frame %d: %w", len(images), err)
		}
		images = append(images, img)
	}
	return images, nil
}

// verify the interface is actually satisfied
var _ media.Cutter = (*Sampler)(nil)

// Duration implements media.Cutter: the length ffprobe reports for the
// video stream, or for the container when the stream states none, and 0
// when neither does (a stream dump, a still image) or what they state is no
// length a video has (not a number, beyond media.MaxDuration). The
// container's length is that of its longest stream, so a file whose sound
// runs on after the picture is as long as its picture here.
func (s *Sampler) Duration(ctx context.Context, path string) (time.Duration, error) {
	in, err := s.probe(ctx, path)
	return in.duration, err
}

// Cut implements media.Cutter: it copies the video stream of path from
// start for length (to the end when length <= 0) into out without
// re-encoding, so a cut starts at the last keyframe before start and the
// segment's own timestamps start at zero. Audio, subtitles and data are
// dropped; an MP4 or MOV gets its index at the front, so it plays from a
// pipe. ctx bounds ffmpeg; cancelling it kills the program.
//
// ffmpeg writes into a file of this call's own, next to out and with out's
// extension, which takes out's name once the cut is whole: a file already at
// out is replaced by a cut that succeeds and left as it was by one that
// fails or is cancelled, and no partial segment ever appears there. The
// segment is readable by its owner only. An out that names the input itself,
// under any spelling or through a link, is refused before anything runs.
//
// A start past zero that lies at or past the end of the video stream, or
// less than a tenth of a second before it, is media.ErrPastEnd, wrapped,
// and writes nothing (see pastEnd for how it is told).
func (s *Sampler) Cut(ctx context.Context, path string, start, length time.Duration, out string) error {
	if path == "" || out == "" {
		return errors.New("ffmpeg: empty path")
	}
	if sameFile(path, out) {
		return fmt.Errorf("ffmpeg: cut %q into %q: the output is the same file as the input", path, out)
	}
	tmp, err := tempNextTo(out)
	if err != nil {
		return fmt.Errorf("ffmpeg: %w", err)
	}
	args := slices.Concat([]string{"-nostdin", "-hide_banner", "-v", "error", "-y", "-ss", seconds(start)}, input(path))
	if length > 0 {
		args = append(args, "-t", seconds(length))
	}
	args = append(args, "-map", "0:v:0", "-an", "-sn", "-dn", "-c", "copy", "-movflags", "+faststart", "-progress", "pipe:1", local(tmp))
	var progress tailWriter
	err = s.exec(ctx, "ffmpeg", s.cfg.FFmpeg, args, &progress)
	if err == nil && start > 0 && pastEnd(progress.String()) {
		err = fmt.Errorf("ffmpeg: cut %q from %s: %w", path, start, media.ErrPastEnd)
	}
	if err == nil {
		err = os.Rename(tmp, out)
	}
	if err != nil {
		// whatever ffmpeg wrote is not the segment; only this call's own file
		// is removed
		if rmErr := os.Remove(tmp); rmErr != nil && !errors.Is(rmErr, os.ErrNotExist) {
			return errors.Join(err, fmt.Errorf("ffmpeg: remove the partial cut: %w", rmErr))
		}
		return err
	}
	return nil
}

// outTimeKey starts the line of ffmpeg's progress report (-progress) that
// says how far the output reaches: the end of the last packet written, in
// microseconds from the start that was asked for.
const outTimeKey = "out_time_us="

// minCutTail is how much video a cut must hold after its start to count as
// one. The end of the output is known to within a few frames only (it is
// read from decoding timestamps, which run ahead of the pictures in a
// stream with reordering, and a container may start a few milliseconds
// past zero), and a stretch this short has nothing to see that the stretch
// before it did not show.
const minCutTail = 100 * time.Millisecond

// pastEnd reads the end of ffmpeg's progress report for a stream copy and
// reports whether the cut started at or past the end of the video, to
// within minCutTail. Such a copy is not empty: ffmpeg writes the last group
// of pictures before the start again, every packet of it earlier than the
// start, and exits 0. The report gives it away, since the output then ends
// at or before zero. A report without a time (a program that writes none,
// "N/A") says nothing.
func pastEnd(progress string) bool {
	i := strings.LastIndex(progress, outTimeKey)
	if i < 0 {
		return false
	}
	value, _, _ := strings.Cut(progress[i+len(outTimeKey):], "\n")
	us, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
	return err == nil && us < minCutTail.Microseconds()
}

// sameFile reports whether a and b name one file: they are the same file on
// disk (links followed), or, when one of them does not exist yet, the same
// path once both are made absolute and cleaned.
func sameFile(a, b string) bool {
	if ai, err := os.Stat(a); err == nil {
		if bi, err := os.Stat(b); err == nil && os.SameFile(ai, bi) {
			return true
		}
	}
	absA, errA := filepath.Abs(a)
	absB, errB := filepath.Abs(b)
	return errA == nil && errB == nil && absA == absB
}

// tempNextTo creates an empty file next to out, with out's extension (ffmpeg
// picks the container by it) and a name no other call has, and returns its
// path: where a cut is written before it takes out's name.
func tempNextTo(out string) (string, error) {
	f, err := os.CreateTemp(filepath.Dir(out), ".langai-cut-*"+filepath.Ext(out))
	if err != nil {
		return "", err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(f.Name())
		return "", err
	}
	return f.Name(), nil
}

// local writes a file path the way ffmpeg and ffprobe take it for a local
// file whatever its name, as an input and as an output: behind the file:
// protocol, which reads the rest as the name without unescaping anything.
func local(path string) string {
	return "file:" + path
}

// input is the "-i" of a local file, with the input option that keeps
// ffmpeg from opening anything but local files on its behalf: the inputs a
// playlist or a concat list names are opened under the same whitelist.
func input(path string) []string {
	return []string{"-protocol_whitelist", "file", "-i", local(path)}
}

// run runs one of the two programs to its end and returns what it wrote to
// stdout. name says which program it is in the error, which is ctx's when
// ctx ended the run and otherwise ends with the end of the program's log.
func (s *Sampler) run(ctx context.Context, name, bin string, args []string) ([]byte, error) {
	var stdout bytes.Buffer
	if err := s.exec(ctx, name, bin, args, &limitedWriter{buf: &stdout, limit: maxProbeOutput}); err != nil {
		return nil, err
	}
	return stdout.Bytes(), nil
}

// exec runs one of the two programs to its end with its stdout going to
// stdout, which must not block. The error is as run describes it.
func (s *Sampler) exec(ctx context.Context, name, bin string, args []string, stdout io.Writer) error {
	var stderr tailWriter
	cmd := exec.CommandContext(ctx, bin, args...) //nolint:gosec // the binary is the operator's configuration and the arguments are built by the callers, each path as one argv entry, never a shell string
	cmd.Stdout = stdout
	cmd.Stderr = &stderr
	cmd.WaitDelay = s.waitDelay
	if err := cmd.Run(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return fmt.Errorf("%s: %w", name, ctxErr)
		}
		return fmt.Errorf("%s: %w: %s", name, err, stderr.String())
	}
	return nil
}

// probed is what ffprobe says about an input.
type probed struct {
	// duration is the video stream's, or the container's when the stream
	// states none; 0 when neither states one a video can have.
	duration time.Duration
	// image2 says that the image demuxer reads the file. It takes a "%d" in
	// the name for the pattern of a numbered sequence unless told not to.
	image2 bool
}

// literalName is the input option that makes the image demuxer read the
// file named, whatever pattern the name spells. Only that demuxer has it:
// ffmpeg refuses it for any other.
var literalName = []string{"-pattern_type", "none"}

// probeEntries is what probe asks ffprobe for: the first video stream's
// duration, the DURATION tag Matroska notes per stream in its place, and the
// container's demuxer and duration. The answer comes in sections ([STREAM],
// [FORMAT]), which is what tells the two durations apart.
var probeEntries = []string{"-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration:stream_tags=DURATION:format=format_name,duration", "-of", "default"}

// probe asks ffprobe for the video's duration and the container's demuxer.
//
// A name with a percent sign may spell a sequence pattern, and the image
// demuxer would then read other files than the one named. Such a name is
// probed with patterns turned off first; the image demuxer then answers
// without opening the file, so the file is opened here. An ffprobe that
// refuses that option for other demuxers is asked again without it; when
// the answer is then the image demuxer after all, it speaks for the sequence
// and not for the file, and the first error stands.
func (s *Sampler) probe(ctx context.Context, path string) (probed, error) {
	if path == "" {
		return probed{}, errors.New("ffprobe: empty path")
	}
	args := slices.Concat(probeEntries, input(path))
	if !strings.Contains(path, "%") {
		out, err := s.run(ctx, "ffprobe", s.cfg.FFprobe, args)
		return parseProbe(out), err
	}
	f, err := os.Open(path) //nolint:gosec // the path is the video the caller asked for; it is only opened to see that it can be read
	if err != nil {
		return probed{}, fmt.Errorf("ffprobe: %w", err)
	}
	_ = f.Close()
	out, literalErr := s.run(ctx, "ffprobe", s.cfg.FFprobe, slices.Concat(literalName, args))
	if literalErr == nil || ctx.Err() != nil {
		return parseProbe(out), literalErr
	}
	out, err = s.run(ctx, "ffprobe", s.cfg.FFprobe, args)
	if err != nil {
		return probed{}, err
	}
	in := parseProbe(out)
	if in.image2 {
		return probed{}, literalErr
	}
	return in, nil
}

// parseProbe reads ffprobe's answer: key=value lines, those of the video
// stream between [STREAM] and [/STREAM]. The duration is the stream's own,
// else the stream's DURATION tag, else the container's (a duration outside
// a stream section).
func parseProbe(out []byte) probed {
	var in probed
	var video, container time.Duration
	inStream := false
	for line := range strings.Lines(string(out)) {
		key, value, _ := strings.Cut(strings.TrimSpace(line), "=")
		switch {
		case key == "[STREAM]":
			inStream = true
		case key == "[/STREAM]":
			inStream = false
		case key == "format_name":
			in.image2 = slices.Contains(strings.Split(value, ","), "image2")
		case key == "duration" && !inStream:
			container = parseDuration(value)
		case key == "duration" && video == 0:
			video = parseDuration(value)
		case key == "TAG:DURATION" && inStream && video == 0:
			video = parseClock(value)
		}
	}
	in.duration = video
	if video == 0 {
		in.duration = container
	}
	return in
}

// parseDuration reads ffprobe's duration in seconds. Anything but a length
// a video can have is 0, unknown: no number ("N/A", nothing at all, NaN),
// nothing or less, and more than media.MaxDuration, which also keeps the
// number inside what a time.Duration holds.
func parseDuration(out string) time.Duration {
	secs, err := strconv.ParseFloat(strings.TrimSpace(out), 64)
	if err != nil || math.IsNaN(secs) || secs <= 0 || secs > media.MaxDuration.Seconds() {
		return 0
	}
	return time.Duration(secs * float64(time.Second))
}

// parseClock reads a duration written as a clock time, hours:minutes:seconds
// with a fraction, as Matroska's DURATION tag has it; anything else is 0,
// and so is a time parseDuration would not take.
func parseClock(out string) time.Duration {
	parts := strings.Split(strings.TrimSpace(out), ":")
	if len(parts) != 3 {
		return 0
	}
	hours, errH := strconv.ParseUint(parts[0], 10, 32)
	minutes, errM := strconv.ParseUint(parts[1], 10, 32)
	secs, errS := strconv.ParseFloat(parts[2], 64)
	if errH != nil || errM != nil || errS != nil || math.IsNaN(secs) || secs < 0 {
		return 0
	}
	return parseDuration(strconv.FormatFloat(float64(hours)*3600+float64(minutes)*60+secs, 'f', -1, 64))
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
// through (showinfo) and write them to stdout as RGBA images in the PAM
// format, whose header gives each frame's size, stopping at MaxFrames. The
// three filters form one chain, so showinfo's lines come in the order of the
// frames on stdout.
func (s *Sampler) args(path string, in probed) []string {
	gap, maxI := s.gaps(in.duration)
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
	if in.image2 {
		args = append(args, literalName...)
	}
	args = append(args, input(path)...)
	args = append(args, "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", strings.Join(filters, ","), "-fps_mode", "vfr")
	if s.cfg.MaxFrames > 0 {
		args = append(args, "-frames:v", strconv.Itoa(s.cfg.MaxFrames))
	}
	return append(args, "-f", "image2pipe", "-c:v", "pam", "-pix_fmt", "rgba", "pipe:1")
}

// frameInfo is what one showinfo line says about a frame.
type frameInfo struct {
	at   time.Duration
	w, h int
}

// showinfoLine matches the per-frame lines of the showinfo filter, e.g.
// "[Parsed_showinfo_2 @ 0x...] n:   1 pts:  25600 pts_time:2  ... s:320x240 ...".
// It is anchored to the start of the line and to the filter's own log
// prefix: ffmpeg prints what the file says about itself (a title, a chapter
// name) into the same log, indented behind its key, and a title must not be
// able to spell a frame.
var showinfoLine = regexp.MustCompile(`^\[Parsed_showinfo_\d+ @ [^\]]+\] n: *(\d+) .*\bpts_time:(\S+) .*\bs:(\d+)x(\d+)\b`)

// parseShowinfo reads one line of ffmpeg's log: a showinfo line gives the
// number of its frame, counted from 0 by the filter, and the frame's time
// and size; any other line is not a frame's.
func parseShowinfo(line []byte) (n int, info frameInfo, ok bool, err error) {
	m := showinfoLine.FindSubmatch(line)
	if m == nil {
		return 0, frameInfo{}, false, nil
	}
	n, err = strconv.Atoi(string(m[1]))
	if err != nil {
		return 0, frameInfo{}, false, fmt.Errorf("ffmpeg: frame number %q", m[1])
	}
	secs, err := strconv.ParseFloat(string(m[2]), 64)
	if err != nil {
		return 0, frameInfo{}, false, fmt.Errorf("ffmpeg: frame %d has no timestamp (%q)", n, m[2])
	}
	w, _ := strconv.Atoi(string(m[3]))
	h, _ := strconv.Atoi(string(m[4]))
	if w <= 0 || h <= 0 {
		return 0, frameInfo{}, false, fmt.Errorf("ffmpeg: frame %d has size %dx%d", n, w, h)
	}
	return n, frameInfo{at: time.Duration(secs * float64(time.Second)), w: w, h: h}, true, nil
}

// frameLog is where ffmpeg's log goes while it samples: it picks the frame
// lines out as they come, in order, and keeps the end of the log for an
// error message. Nothing else of the log is held, however long it gets. Its
// fields are read once ffmpeg has been waited for.
type frameLog struct {
	tail  tailWriter
	line  []byte // the line being written, up to maxLogLine of it
	infos []frameInfo
	err   error // the first frame line that made no sense
}

// Write implements io.Writer; it never fails and never blocks.
func (l *frameLog) Write(p []byte) (int, error) {
	_, _ = l.tail.Write(p)
	for rest := p; len(rest) > 0; {
		part, after, ended := bytes.Cut(rest, []byte{'\n'})
		l.line = append(l.line, part[:min(len(part), maxLogLine-len(l.line))]...)
		if ended {
			l.endLine()
		}
		rest = after
	}
	return len(p), nil
}

// endLine takes the finished line in, when it is a frame's. The filter
// numbers its frames, and the number is what places a line: frame n's line
// is entry n, a number logged again replaces what stood there, and a number
// that skips one is an error, since the frames after it would get the times
// of others.
func (l *frameLog) endLine() {
	n, info, ok, err := parseShowinfo(l.line)
	l.line = l.line[:0]
	switch {
	case err != nil:
		l.fail(err)
	case !ok:
	case n < len(l.infos):
		l.infos[n] = info
	case n == len(l.infos):
		l.infos = append(l.infos, info)
	default:
		l.fail(fmt.Errorf("ffmpeg: frame %d logged after %d: a frame line is missing", n, len(l.infos)-1))
	}
}

// fail keeps the first error.
func (l *frameLog) fail(err error) {
	if l.err == nil {
		l.err = err
	}
}

// readPAM reads one RGBA image in the PAM format, as ffmpeg's pam encoder
// writes it: a text header and then the pixels, which are read into an
// allocation of their own. io.EOF means the stream ended before the image
// began; an image cut short is io.ErrUnexpectedEOF. maxSide, when positive,
// is the longest side the frame may have.
func readPAM(br *bufio.Reader, maxSide int) (*image.RGBA, error) {
	w, h, err := readPAMHeader(br)
	if err != nil {
		return nil, err
	}
	if maxSide > 0 && (w > maxSide || h > maxSide) {
		return nil, fmt.Errorf("frame is %dx%d, larger than MaxSide %d", w, h, maxSide)
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	if _, err := io.ReadFull(br, img.Pix); err != nil {
		return nil, fmt.Errorf("%dx%d pixels: %w", w, h, cutShort(err))
	}
	return img, nil
}

// maxPAMHeaderLines is more lines than a PAM header has.
const maxPAMHeaderLines = 16

// readPAMHeader reads the header of an 8-bit RGBA PAM image, a line each:
// "P7", WIDTH, HEIGHT, DEPTH 4, MAXVAL 255, TUPLTYPE and ENDHDR. It returns
// the size, or io.EOF when the stream ends before the first byte.
func readPAMHeader(br *bufio.Reader) (w, h int, err error) {
	magic, err := br.ReadSlice('\n')
	if errors.Is(err, io.EOF) && len(magic) == 0 {
		return 0, 0, io.EOF
	}
	if err != nil {
		return 0, 0, fmt.Errorf("PAM header: %w", cutShort(err))
	}
	if string(magic) != "P7\n" {
		return 0, 0, fmt.Errorf("not a PAM image: it starts with %q", magic[:min(len(magic), 16)])
	}
	fields := map[string]int{}
	for range maxPAMHeaderLines {
		line, err := br.ReadSlice('\n')
		if err != nil {
			return 0, 0, fmt.Errorf("PAM header: %w", cutShort(err))
		}
		key, value, _ := strings.Cut(strings.TrimSpace(string(line)), " ")
		if key != "ENDHDR" {
			fields[key], _ = strconv.Atoi(strings.TrimSpace(value))
			continue
		}
		w, h = fields["WIDTH"], fields["HEIGHT"]
		if fields["DEPTH"] != 4 || fields["MAXVAL"] != 255 {
			return 0, 0, fmt.Errorf("PAM image has depth %d and maxval %d, want 8-bit RGBA", fields["DEPTH"], fields["MAXVAL"])
		}
		if w <= 0 || h <= 0 || int64(w)*int64(h) > maxFramePixels {
			return 0, 0, fmt.Errorf("PAM image is %dx%d", w, h)
		}
		return w, h, nil
	}
	return 0, 0, errors.New("PAM header does not end")
}

// cutShort turns the io.EOF of a read inside an image into
// io.ErrUnexpectedEOF: the end of the stream is only clean before an image.
func cutShort(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}

// limitedWriter keeps the first limit bytes written and drops the rest, so
// an output that never ends cannot grow without bound.
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

// tailWriter keeps the last maxLog bytes written and drops what came before,
// so a log that never ends cannot grow without bound and an error message
// still ends with what the program said last. The zero value is ready.
type tailWriter struct {
	buf []byte
}

func (w *tailWriter) Write(p []byte) (int, error) {
	n := len(p)
	p = p[max(0, len(p)-maxLog):]
	if len(w.buf)+len(p) > 2*maxLog {
		w.buf = append(w.buf[:0], w.buf[len(w.buf)-maxLog:]...)
	}
	w.buf = append(w.buf, p...)
	return n, nil
}

// String returns the end of the log, whitespace trimmed.
func (w *tailWriter) String() string {
	return strings.TrimSpace(string(w.buf[max(0, len(w.buf)-maxLog):]))
}

package ffmpeg

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestNew_FillsDefaults(t *testing.T) {
	s, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	want := Config{FFmpeg: "ffmpeg", FFprobe: "ffprobe", SceneThreshold: DefaultSceneThreshold,
		MinInterval: DefaultMinInterval, MaxInterval: DefaultMaxInterval, MaxFrames: DefaultMaxFrames, MaxSide: DefaultMaxSide}
	if s.cfg != want {
		t.Errorf("cfg = %+v, want %+v", s.cfg, want)
	}
	// a negative MaxSide (keep the original size) is a setting, not a zero
	s, _ = New(Config{MaxSide: -1, FFmpeg: "/opt/ffmpeg"})
	if s.cfg.MaxSide != -1 || s.cfg.FFmpeg != "/opt/ffmpeg" {
		t.Errorf("explicit settings overwritten: %+v", s.cfg)
	}
}

func TestNew_RejectsNegativeSettings(t *testing.T) {
	for name, cfg := range map[string]Config{
		"scene":     {SceneThreshold: -0.1},
		"min":       {MinInterval: -time.Second},
		"max":       {MaxInterval: -time.Second},
		"maxframes": {MaxFrames: -1},
	} {
		if _, err := New(cfg); err == nil {
			t.Errorf("%s: want an error for %+v", name, cfg)
		}
	}
}

func TestGaps_RaiseTheMinimumForLongVideos(t *testing.T) {
	s, _ := New(Config{})
	cases := []struct {
		duration  time.Duration
		gap, maxI time.Duration
	}{
		{0, time.Second, 5 * time.Second},                                    // unknown duration: the configured gaps
		{30 * time.Second, time.Second, 5 * time.Second},                     // 30/64 < 1s: the configured gaps
		{10 * time.Minute, 9375 * time.Millisecond, 9375 * time.Millisecond}, // 600/64: both raised
		{2 * time.Hour, 112500 * time.Millisecond, 112500 * time.Millisecond},
	}
	for _, c := range cases {
		gap, maxI := s.gaps(c.duration)
		if gap != c.gap || maxI != c.maxI {
			t.Errorf("gaps(%v) = %v, %v; want %v, %v", c.duration, gap, maxI, c.gap, c.maxI)
		}
	}
	// MaxFrames 0 after defaults cannot happen, but a sampler built by hand
	// with no cap must not divide by zero
	s = &Sampler{cfg: Config{MinInterval: time.Second, MaxInterval: 2 * time.Second}}
	if gap, maxI := s.gaps(time.Hour); gap != time.Second || maxI != 2*time.Second {
		t.Errorf("no cap: gaps = %v, %v", gap, maxI)
	}
}

func TestSelectExpr(t *testing.T) {
	s, _ := New(Config{})
	expr := s.selectExpr(time.Second, 5*time.Second)
	for _, want := range []string{"isnan(prev_selected_t)", "gte(t-prev_selected_t,5.000)", "gt(scene,0.400)*gte(t-prev_selected_t,1.000)"} {
		if !strings.Contains(expr, want) {
			t.Errorf("expr %q lacks %q", expr, want)
		}
	}
	s, _ = New(Config{KeyframesOnly: true})
	expr = s.selectExpr(1500*time.Millisecond, 5*time.Second)
	if expr != "isnan(prev_selected_t)+gte(t-prev_selected_t,1.500)" {
		t.Errorf("keyframes only: expr = %q", expr)
	}
	s, _ = New(Config{SceneThreshold: 1})
	if expr = s.selectExpr(time.Second, 5*time.Second); strings.Contains(expr, "scene") {
		t.Errorf("threshold 1 must drop scene detection, got %q", expr)
	}
}

func TestArgs(t *testing.T) {
	s, _ := New(Config{})
	args := strings.Join(s.args("-odd name.mp4", probed{duration: 10 * time.Second}), " ")
	for _, want := range []string{"-i file:-odd name.mp4 -map 0:v:0", "-vf select='", "',scale=w='min(512,iw)':h='min(512,ih)':force_original_aspect_ratio=decrease,showinfo",
		"-fps_mode vfr -frames:v 64 -f image2pipe -c:v pam -pix_fmt rgba pipe:1", "-nostdin"} {
		if !strings.Contains(args, want) {
			t.Errorf("args %q lack %q", args, want)
		}
	}
	if strings.Contains(args, "-skip_frame") || strings.Contains(args, "-pattern_type") {
		t.Errorf("args %q decode keyframes only or set the image demuxer's option without being asked", args)
	}
	s, _ = New(Config{KeyframesOnly: true, MaxSide: -1})
	args = strings.Join(s.args("clip.mp4", probed{}), " ")
	if !strings.Contains(args, "-skip_frame nokey -protocol_whitelist file -i file:clip.mp4") {
		t.Errorf("keyframes only: args %q lack the decoder option before the input", args)
	}
	// the image demuxer is told to read the file named, not a sequence
	if args := strings.Join(s.args("shot%d.png", probed{image2: true}), " "); !strings.Contains(args, "-pattern_type none -protocol_whitelist file -i file:shot%d.png") {
		t.Errorf("image: args %q lack the literal-name option before the input", args)
	}
	if strings.Contains(args, "scale=") {
		t.Errorf("MaxSide -1: args %q still scale", args)
	}
	if !strings.Contains(args, "select='isnan(prev_selected_t)+gte(t-prev_selected_t,1.000)',showinfo") {
		t.Errorf("keyframes only: args %q", args)
	}
}

func TestParseDuration(t *testing.T) {
	for in, want := range map[string]time.Duration{
		"10.000000\n": 10 * time.Second,
		"0.5":         500 * time.Millisecond,
		"N/A\n":       0,
		"":            0,
		"-3":          0,
	} {
		if got := parseDuration(in); got != want {
			t.Errorf("parseDuration(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestParseProbe(t *testing.T) {
	cases := map[string]probed{
		"format_name=mov,mp4,m4a,3gp,3g2,mj2\nduration=2.000000\n": {duration: 2 * time.Second},
		"format_name=image2\nduration=0.040000\n":                  {duration: 40 * time.Millisecond, image2: true},
		"format_name=png_pipe\r\nduration=N/A\r\n":                 {},
		"format_name=image2pipe\nduration=N/A\n":                   {},
		"":                                                         {},
	}
	for out, want := range cases {
		if got := parseProbe([]byte(out)); got != want {
			t.Errorf("parseProbe(%q) = %+v, want %+v", out, got, want)
		}
	}
}

// recordedLog is what ffmpeg 8 prints for two frames, with the lines a
// real log puts between them.
const recordedLog = `[Parsed_showinfo_2 @ 0x79423c004900] config in time_base: 1/12800, frame_rate: 25/1
[Parsed_showinfo_2 @ 0x79423c004900] config out time_base: 0/0, frame_rate: 0/0
[Parsed_showinfo_2 @ 0x79423c004900] n:   0 pts:      0 pts_time:0       duration:    512 duration_time:0.04    fmt:yuv420p cl:left sar:1/1 s:320x240 i:P iskey:1 type:I checksum:217CA531 plane_checksum:[EFD7F182 22B45F86 B1DB541A] mean:[81 90 240] stdev:[0.0 0.0 0.0]
[Parsed_showinfo_2 @ 0x79423c004900]   side data - H.26[45] User Data Unregistered SEI message: UUID=dc45e9bd-e6d9-48b7-962c-d820d923eeef
[Parsed_showinfo_2 @ 0x79423c004900] color_range:unknown color_space:unknown color_primaries:unknown color_trc:unknown
[Parsed_showinfo_2 @ 0x79423c004900] n:   1 pts:  25600 pts_time:2.04    duration:    512 duration_time:0.04    fmt:yuv420p cl:left sar:1/1 s:160x120 i:P iskey:0 type:B checksum:217CA531 plane_checksum:[EFD7F182 22B45F86 B1DB541A] mean:[81 90 240] stdev:[0.0 0.0 0.0]
`

// TestFrameLog_PicksTheFrameLines: the log arrives in whatever pieces the
// pipe hands over, so the frame lines must come out the same whether it is
// written whole, byte by byte or in chunks that split every line.
func TestFrameLog_PicksTheFrameLines(t *testing.T) {
	want := []frameInfo{{at: 0, w: 320, h: 240}, {at: 2040 * time.Millisecond, w: 160, h: 120}}
	for _, chunk := range []int{len(recordedLog), 1, 7, 100} {
		log := &frameLog{}
		for rest := []byte(recordedLog); len(rest) > 0; rest = rest[min(chunk, len(rest)):] {
			part := rest[:min(chunk, len(rest))]
			if n, err := log.Write(part); n != len(part) || err != nil {
				t.Fatalf("Write = %d, %v", n, err)
			}
		}
		if log.err != nil || !slices.Equal(log.infos, want) {
			t.Errorf("chunks of %d: infos = %+v, %v; want %+v", chunk, log.infos, log.err, want)
		}
	}

	log := &frameLog{}
	_, _ = log.Write([]byte(strings.Replace(recordedLog, "pts_time:2.04 ", "pts_time:NOPTS", 1)))
	if log.err == nil || !strings.Contains(log.err.Error(), "frame 1 has no timestamp") {
		t.Errorf("a frame without a timestamp: err = %v", log.err)
	}
}

// TestFrameLog_TakesOnlyTheFiltersOwnLines: ffmpeg prints what a file says
// about itself (its title, its chapters) into the same log, so a frame line
// is one that starts with the showinfo filter's own prefix, and a frame the
// filter logs stands over whatever claimed its number before.
func TestFrameLog_TakesOnlyTheFiltersOwnLines(t *testing.T) {
	frame := func(n int, at string, size string) string {
		return "[Parsed_showinfo_2 @ 0x1] n:   " + strconv.Itoa(n) + " pts:  0 pts_time:" + at + " duration: 1 fmt:rgba s:" + size + " i:P\n"
	}
	two := []frameInfo{{at: 0, w: 320, h: 240}, {at: 5 * time.Second, w: 320, h: 240}}
	for _, tc := range []struct {
		name    string
		log     string
		want    []frameInfo
		wantErr string
	}{
		{
			name: "a title that spells a frame line",
			log:  "    title           : showinfo n: 0 pts_time:999 s:320x240 x\n" + frame(0, "0", "320x240") + frame(1, "5", "320x240"),
			want: two,
		},
		{
			name: "a title that spells the filter's prefix too",
			log:  "    title           : [Parsed_showinfo_2 @ 0x1] n: 0 pts:0 pts_time:999 s:1x1 x\n" + frame(0, "0", "320x240") + frame(1, "5", "320x240"),
			want: two,
		},
		{
			name: "the prefix in the middle of another filter's line",
			log:  "[Parsed_scale_1 @ 0x2] w:320 [Parsed_showinfo_2 @ 0x1] n: 0 pts_time:999 s:1x1 x\n" + frame(0, "0", "320x240") + frame(1, "5", "320x240"),
			want: two,
		},
		{
			name: "a frame logged twice: the later line stands",
			log:  frame(0, "999", "1x1") + frame(0, "0", "320x240") + frame(1, "5", "320x240"),
			want: two,
		},
		{
			name:    "a frame number out of turn",
			log:     frame(0, "0", "320x240") + frame(2, "5", "320x240"),
			wantErr: "frame 2 logged after 0",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			log := &frameLog{}
			_, _ = log.Write([]byte(tc.log))
			if tc.wantErr != "" {
				if log.err == nil || !strings.Contains(log.err.Error(), tc.wantErr) {
					t.Errorf("err = %v, want one saying %q", log.err, tc.wantErr)
				}
				return
			}
			if log.err != nil || !slices.Equal(log.infos, tc.want) {
				t.Errorf("infos = %+v, %v; want %+v", log.infos, log.err, tc.want)
			}
		})
	}
}

// TestFrameLog_HoldsOnlyTheEndOfALongLog: megabytes of log, one line of them
// endless, leave the frame lines, a bounded line buffer and the last words.
func TestFrameLog_HoldsOnlyTheEndOfALongLog(t *testing.T) {
	log := &frameLog{}
	noise := []byte(strings.Repeat("x", 1000))
	for range 4 << 10 {
		_, _ = log.Write(noise) // 4 MB without a line end
	}
	_, _ = log.Write([]byte("\n" + recordedLog + "the last words\n"))
	if len(log.infos) != 2 || log.err != nil {
		t.Errorf("infos = %+v, %v; want the two frames after the endless line", log.infos, log.err)
	}
	if cap(log.line) > 2*maxLogLine || cap(log.tail.buf) > 4*maxLog {
		t.Errorf("the log holds a %d-byte line and a %d-byte tail", cap(log.line), cap(log.tail.buf))
	}
	if got := log.tail.String(); len(got) > maxLog || !strings.HasSuffix(got, "the last words") {
		t.Errorf("tail = %d bytes ending %q", len(got), got[max(0, len(got)-20):])
	}
}

// pam is a PAM image as ffmpeg's encoder writes it.
func pam(w, h int, pix ...byte) []byte {
	return append(fmt.Appendf(nil, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", w, h), pix...)
}

func TestReadPAM(t *testing.T) {
	// 2x1: red, green; then 1x1: blue
	stream := slices.Concat(pam(2, 1, 255, 0, 0, 255, 0, 255, 0, 255), pam(1, 1, 0, 0, 255, 255))
	br := bufio.NewReader(bytes.NewReader(stream))
	first, err := readPAM(br, 0)
	if err != nil {
		t.Fatal(err)
	}
	if b := first.Bounds(); b.Dx() != 2 || b.Dy() != 1 || len(first.Pix) != 8 {
		t.Fatalf("frame 0 is %v with %d bytes, want 2x1", b, len(first.Pix))
	}
	if r, g, b, _ := first.At(1, 0).RGBA(); r != 0 || g != 0xffff || b != 0 {
		t.Errorf("frame 0 pixel (1,0) = %v %v %v, want green", r, g, b)
	}
	second, err := readPAM(br, 0)
	if err != nil {
		t.Fatal(err)
	}
	if r, _, b, _ := second.At(0, 0).RGBA(); second.Bounds().Dx() != 1 || r != 0 || b != 0xffff {
		t.Errorf("frame 1 = %v, pixel %v %v; want 1x1 blue", second.Bounds(), r, b)
	}
	if _, err := readPAM(br, 0); err != io.EOF {
		t.Errorf("after the last frame: err = %v, want io.EOF", err)
	}
}

func TestReadPAM_Errors(t *testing.T) {
	good := pam(2, 1, 1, 2, 3, 4, 5, 6, 7, 8)
	cases := []struct {
		name    string
		stream  []byte
		maxSide int
		want    string
	}{
		{"pixels cut short", good[:len(good)-3], 0, "unexpected EOF"},
		{"header cut short", good[:20], 0, "unexpected EOF"},
		{"not PAM", []byte("RIFF....AVI \n"), 0, "not a PAM image"},
		{"no line end", []byte("P7"), 0, "unexpected EOF"},
		{"RGB without alpha", bytes.Replace(good, []byte("DEPTH 4"), []byte("DEPTH 3"), 1), 0, "depth 3"},
		{"16 bits", bytes.Replace(good, []byte("MAXVAL 255"), []byte("MAXVAL 65535"), 1), 0, "maxval 65535"},
		{"no size", []byte("P7\nDEPTH 4\nMAXVAL 255\nENDHDR\n"), 0, "0x0"},
		{"absurd size", pam(1<<20, 1<<20), 0, "1048576x1048576"},
		{"larger than asked for", good, 1, "larger than MaxSide 1"},
		{"endless header", []byte("P7\n" + strings.Repeat("# comment\n", 100)), 0, "does not end"},
		{"endless line", []byte("P7\n" + strings.Repeat("x", 1<<16)), 0, "PAM header"},
	}
	for _, tc := range cases {
		_, err := readPAM(bufio.NewReader(bytes.NewReader(tc.stream)), tc.maxSide)
		if err == nil || err == io.EOF || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one that says %q", tc.name, err, tc.want)
		}
	}
}

func TestLimitedWriter(t *testing.T) {
	var buf bytes.Buffer
	w := &limitedWriter{buf: &buf, limit: 5}
	for _, s := range []string{"abc", "def", "ghi"} {
		if n, err := w.Write([]byte(s)); n != 3 || err != nil {
			t.Errorf("Write(%q) = %d, %v", s, n, err)
		}
	}
	if buf.String() != "abcde" {
		t.Errorf("kept %q, want abcde", buf.String())
	}
}

func TestTailWriter(t *testing.T) {
	cases := map[string][]string{
		"short":            {"  one line\n"},
		"many small":       slices.Repeat([]string{"0123456789\n"}, 5000),
		"one huge":         {strings.Repeat("x", 10*maxLog) + "\n"},
		"huge after small": {"a\n", "b\n", strings.Repeat("x", 3*maxLog)},
	}
	for name, writes := range cases {
		var w tailWriter
		for _, s := range writes {
			if n, err := w.Write([]byte(s)); n != len(s) || err != nil {
				t.Fatalf("%s: Write = %d, %v; want %d", name, n, err, len(s))
			}
		}
		all := strings.Join(writes, "")
		want := strings.TrimSpace(all[max(0, len(all)-maxLog):])
		if got := w.String(); got != want {
			t.Errorf("%s: tail = %d bytes ending %q, want the last %d", name, len(got), got[max(0, len(got)-12):], len(want))
		}
		if cap(w.buf) > 4*maxLog {
			t.Errorf("%s: the writer holds %d bytes", name, cap(w.buf))
		}
	}
}

// the Sampler is a search.Fingerprinter, without this package importing search
var _ interface{ Fingerprint() string } = (*Sampler)(nil)

// TestFingerprint_ChangesWithWhatChangesTheFrames: two configurations that
// take other frames have other fingerprints, so a MediaIndex re-indexes its
// videos after the change; two that take the same frames (a default and its
// spelling, a scene threshold that scene detection never reads, the binary's
// path) share one.
func TestFingerprint_ChangesWithWhatChangesTheFrames(t *testing.T) {
	fp := func(cfg Config) string {
		t.Helper()
		s, err := New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		return s.Fingerprint()
	}
	for name, pair := range map[string][2]Config{
		"the defaults spelled out": {{}, {SceneThreshold: DefaultSceneThreshold, MinInterval: DefaultMinInterval, MaxInterval: DefaultMaxInterval, MaxFrames: DefaultMaxFrames, MaxSide: DefaultMaxSide}},
		"another binary":           {{}, {FFmpeg: "/opt/ffmpeg/bin/ffmpeg", FFprobe: "/opt/ffmpeg/bin/ffprobe"}},
		"scene detection off":      {{SceneThreshold: 1}, {SceneThreshold: 3}},
		"keyframes ignore scenes":  {{KeyframesOnly: true}, {KeyframesOnly: true, SceneThreshold: 0.9}},
		"any size kept":            {{MaxSide: -1}, {MaxSide: -7}},
	} {
		if a, b := fp(pair[0]), fp(pair[1]); a != b {
			t.Errorf("%s: %q and %q, want one fingerprint", name, a, b)
		}
	}
	seen := map[string]string{fp(Config{}): "the defaults"}
	for name, cfg := range map[string]Config{
		"scene threshold":    {SceneThreshold: 0.3},
		"scenes off":         {SceneThreshold: 1},
		"min interval":       {MinInterval: 2 * time.Second},
		"max interval":       {MaxInterval: 10 * time.Second},
		"max frames":         {MaxFrames: 16},
		"max side":           {MaxSide: 256},
		"original size":      {MaxSide: -1},
		"keyframes only":     {KeyframesOnly: true},
		"keyframes, min gap": {KeyframesOnly: true, MinInterval: 3 * time.Second},
	} {
		got := fp(cfg)
		if other, dup := seen[got]; dup {
			t.Errorf("%s and %s share the fingerprint %q", name, other, got)
		}
		seen[got] = name
	}
}

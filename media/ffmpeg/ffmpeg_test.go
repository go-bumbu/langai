package ffmpeg

import (
	"bytes"
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
		{0, time.Second, 5 * time.Second},                 // unknown duration: the configured gaps
		{30 * time.Second, time.Second, 5 * time.Second},  // 30/64 < 1s: the configured gaps
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
	args := strings.Join(s.args("-odd name.mp4", 10*time.Second), " ")
	for _, want := range []string{"-i -odd name.mp4 -map 0:v:0", "-vf select='", "',scale=w='min(512,iw)':h='min(512,ih)':force_original_aspect_ratio=decrease,showinfo",
		"-fps_mode vfr -frames:v 64 -f rawvideo -pix_fmt rgba pipe:1", "-nostdin"} {
		if !strings.Contains(args, want) {
			t.Errorf("args %q lack %q", args, want)
		}
	}
	if strings.Contains(args, "-skip_frame") {
		t.Errorf("args %q decode keyframes only without being asked", args)
	}
	s, _ = New(Config{KeyframesOnly: true, MaxSide: -1})
	args = strings.Join(s.args("clip.mp4", 0), " ")
	if !strings.Contains(args, "-skip_frame nokey -i clip.mp4") {
		t.Errorf("keyframes only: args %q lack the decoder option before the input", args)
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

// recordedLog is what ffmpeg 8 prints for two frames, with the lines a
// real log puts between them.
const recordedLog = `[Parsed_showinfo_2 @ 0x79423c004900] config in time_base: 1/12800, frame_rate: 25/1
[Parsed_showinfo_2 @ 0x79423c004900] config out time_base: 0/0, frame_rate: 0/0
[Parsed_showinfo_2 @ 0x79423c004900] n:   0 pts:      0 pts_time:0       duration:    512 duration_time:0.04    fmt:yuv420p cl:left sar:1/1 s:320x240 i:P iskey:1 type:I checksum:217CA531 plane_checksum:[EFD7F182 22B45F86 B1DB541A] mean:[81 90 240] stdev:[0.0 0.0 0.0]
[Parsed_showinfo_2 @ 0x79423c004900]   side data - H.26[45] User Data Unregistered SEI message: UUID=dc45e9bd-e6d9-48b7-962c-d820d923eeef
[Parsed_showinfo_2 @ 0x79423c004900] color_range:unknown color_space:unknown color_primaries:unknown color_trc:unknown
[Parsed_showinfo_2 @ 0x79423c004900] n:   1 pts:  25600 pts_time:2.04    duration:    512 duration_time:0.04    fmt:yuv420p cl:left sar:1/1 s:160x120 i:P iskey:0 type:B checksum:217CA531 plane_checksum:[EFD7F182 22B45F86 B1DB541A] mean:[81 90 240] stdev:[0.0 0.0 0.0]
`

func TestParseShowinfo(t *testing.T) {
	infos, err := parseShowinfo([]byte(recordedLog))
	if err != nil {
		t.Fatal(err)
	}
	want := []frameInfo{{at: 0, w: 320, h: 240}, {at: 2040 * time.Millisecond, w: 160, h: 120}}
	if len(infos) != 2 || infos[0] != want[0] || infos[1] != want[1] {
		t.Errorf("infos = %+v, want %+v", infos, want)
	}
	if infos, err := parseShowinfo(nil); err != nil || len(infos) != 0 {
		t.Errorf("empty log: %v, %v", infos, err)
	}
	bad := strings.Replace(recordedLog, "pts_time:2.04 ", "pts_time:NOPTS", 1)
	if _, err := parseShowinfo([]byte(bad)); err == nil || !strings.Contains(err.Error(), "timestamp") {
		t.Errorf("a frame without a timestamp: err = %v", err)
	}
}

func TestSplitFrames(t *testing.T) {
	infos := []frameInfo{{at: 0, w: 2, h: 1}, {at: time.Second, w: 1, h: 1}}
	data := []byte{255, 0, 0, 255, 0, 255, 0, 255 /* 2x1: red, green */, 0, 0, 255, 255 /* 1x1: blue */}
	frames, err := splitFrames(data, infos)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 2 {
		t.Fatalf("got %d frames, want 2", len(frames))
	}
	if r, g, b, _ := frames[0].Image.At(1, 0).RGBA(); r != 0 || g != 0xffff || b != 0 {
		t.Errorf("frame 0 pixel (1,0) = %v %v %v, want green", r, g, b)
	}
	if frames[1].At != time.Second || frames[1].Image.Bounds().Dx() != 1 {
		t.Errorf("frame 1 = %+v", frames[1])
	}
	if r, _, b, _ := frames[1].Image.At(0, 0).RGBA(); r != 0 || b != 0xffff {
		t.Errorf("frame 1 pixel = %v %v, want blue", r, b)
	}

	// the filters logged a third frame ffmpeg never wrote: fine
	more := append(infos, frameInfo{at: 2 * time.Second, w: 4, h: 4})
	if frames, err := splitFrames(data, more); err != nil || len(frames) != 2 {
		t.Errorf("extra log line: %d frames, %v", len(frames), err)
	}
	if frames, err := splitFrames(nil, infos); err != nil || len(frames) != 0 {
		t.Errorf("no data: %d frames, %v", len(frames), err)
	}
	if _, err := splitFrames(data[:10], infos); err == nil || !strings.Contains(err.Error(), "short") {
		t.Errorf("partial frame: err = %v", err)
	}
	if _, err := splitFrames(append(data, 1, 2, 3, 4), infos); err == nil || !strings.Contains(err.Error(), "beyond") {
		t.Errorf("data past the log: err = %v", err)
	}
}

func TestLimitedWriterAndTail(t *testing.T) {
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
	long := strings.Repeat("x", 3000) + "\nlast line\n"
	if got := tail([]byte(long)); len(got) > 2048 || !strings.HasSuffix(got, "last line") {
		t.Errorf("tail = %d bytes ending %q", len(got), got[max(0, len(got)-20):])
	}
}

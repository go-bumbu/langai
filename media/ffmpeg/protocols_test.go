package ffmpeg

import (
	"context"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// closedPort returns a local address nothing listens on.
func closedPort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	return addr
}

// TestCraftedFilesCannotReachTheNetwork: a playlist, a concat list or a
// session description names other inputs, and ffmpeg would open them
// through whatever protocol they spell. Every program run reads its input
// with the file protocol alone, so such a file fails at once, refused by
// the whitelist itself ("not on whitelist 'file'", narrower than the one
// ffmpeg falls back to), and nothing depends on the connection failing.
func TestCraftedFilesCannotReachTheNetwork(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	addr := closedPort(t)
	dir := t.TempDir()
	files := map[string]string{
		"playlist.m3u8": "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1.0,\nhttp://" + addr + "/seg.ts\n#EXT-X-ENDLIST\n",
		"session.sdp":   "v=0\no=- 0 0 IN IP4 127.0.0.1\ns=x\nc=IN IP4 127.0.0.1\nt=0 0\nm=video " + addr[strings.LastIndex(addr, ":")+1:] + " RTP/AVP 96\na=rtpmap:96 H264/90000\n",
	}
	s, _ := New(Config{})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	for name, body := range files {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		for call, run := range map[string]func() error{
			"Duration": func() error { _, err := s.Duration(ctx, path); return err },
			"Sample":   func() error { _, err := s.Sample(ctx, path); return err },
			"decode":   func() error { _, _, err := s.decode(ctx, path, probed{duration: time.Second}); return err },
			"Cut":      func() error { return s.Cut(ctx, path, 0, time.Second, filepath.Join(dir, "out.mp4")) },
		} {
			start := time.Now()
			err := run()
			if err == nil || !strings.Contains(err.Error(), "not on whitelist 'file'") {
				t.Errorf("%s of %s: err = %v, want the file-only whitelist to refuse the nested input", call, name, err)
			}
			if took := time.Since(start); took > 10*time.Second {
				t.Errorf("%s of %s took %v, want it refused at once", call, name, took)
			}
		}
	}
}

// TestLocalFilesStillWorkUnderTheWhitelist: the whitelist leaves what a
// local video needs: probe, sample, cut, and probe and sample the cut.
func TestLocalFilesStillWorkUnderTheWhitelist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.mp4")
	if err := os.WriteFile(path, tinyVideo(t), 0o600); err != nil {
		t.Fatal(err)
	}
	s, _ := New(Config{})
	ctx := context.Background()
	if d, err := s.Duration(ctx, path); err != nil || d <= 0 {
		t.Fatalf("Duration = %v, %v", d, err)
	}
	if frames, err := s.Sample(ctx, path); err != nil || len(frames) == 0 {
		t.Fatalf("Sample: %d frames, %v", len(frames), err)
	}
	out := filepath.Join(t.TempDir(), "cut.mp4")
	if err := s.Cut(ctx, path, 0, 500*time.Millisecond, out); err != nil {
		t.Fatal(err)
	}
	if frames, err := s.Sample(ctx, out); err != nil || len(frames) == 0 {
		t.Fatalf("Sample of the cut: %d frames, %v", len(frames), err)
	}
}

func TestArgs_WhitelistTheFileProtocolBeforeTheInput(t *testing.T) {
	s, _ := New(Config{})
	args := s.args("/v.mp4", probed{})
	i := strings.Index(strings.Join(args, " "), "-protocol_whitelist file -i file:/v.mp4")
	if i < 0 {
		t.Errorf("args = %q, want the whitelist as the input's option", args)
	}
}

//go:build unix

package ffmpeg

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/go-bumbu/sense/media"
)

// script writes an executable shell script into a fresh directory and
// returns its path.
func script(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "run.sh")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body), 0o700); err != nil {
		t.Fatal(err)
	}
	return path
}

// fakeProbe is an ffprobe that answers at once for any file, for the tests
// that only care about what the ffmpeg after it does.
func fakeProbe(t *testing.T) string {
	return script(t, "echo format_name=mov\necho duration=10\n")
}

// recordPID is the script line that writes the script's own process id,
// which is the id of whatever it then execs, to pidFile.
func recordPID(pidFile string) string {
	return `echo $$ > "` + pidFile + `"` + "\n"
}

// gone checks that the process recorded in pidFile has been reaped: signal 0
// still reaches a zombie, so it only fails with ESRCH once Wait has collected
// the process.
func gone(t *testing.T, pidFile string) {
	t.Helper()
	data, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatalf("the program never started: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		t.Fatal(err)
	}
	if err := syscall.Kill(pid, 0); !errors.Is(err, syscall.ESRCH) {
		t.Errorf("process %d is still there after the call returned (kill -0: %v)", pid, err)
	}
}

// cancelAfter runs call with a context that is cancelled after delay and
// reports how long call took to return once cancelled, and its error.
func cancelAfter(t *testing.T, delay time.Duration, call func(ctx context.Context) error) (time.Duration, error) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cancelled := make(chan time.Time, 1)
	timer := time.AfterFunc(delay, func() { cancelled <- time.Now(); cancel() })
	defer timer.Stop()
	err := call(ctx)
	select {
	case at := <-cancelled:
		return time.Since(at), err
	default:
		t.Fatalf("the call returned before it was cancelled: %v", err)
		return 0, err
	}
}

// TestSample_CancelledMidSampleKillsFFmpeg: -re makes the real ffmpeg read
// the 10 s video at its own speed, so the cancellation lands while frames
// are coming. Sample must return the context's error at once, with ffmpeg
// killed and reaped.
func TestSample_CancelledMidSampleKillsFFmpeg(t *testing.T) {
	path := cutsVideo(t)
	pidFile := filepath.Join(t.TempDir(), "pid")
	slow := script(t, recordPID(pidFile)+`exec ffmpeg -re "$@"`+"\n")
	s, _ := New(Config{FFmpeg: slow, MaxInterval: 500 * time.Millisecond, MinInterval: 100 * time.Millisecond})
	var frames []media.Frame
	took, err := cancelAfter(t, 1500*time.Millisecond, func(ctx context.Context) (err error) {
		frames, err = s.Sample(ctx, path)
		return err
	})
	if !errors.Is(err, context.Canceled) || frames != nil {
		t.Errorf("Sample = %d frames, %v; want context.Canceled and no frames", len(frames), err)
	}
	if took > 2*time.Second {
		t.Errorf("Sample returned %v after the cancellation", took)
	}
	gone(t, pidFile)
}

// TestCancellationKillsEveryProgram: a probe and a cut that hang are killed
// by the context as well.
func TestCancellationKillsEveryProgram(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pid")
	hang := script(t, recordPID(pidFile)+"exec sleep 60\n")
	s, _ := New(Config{FFmpeg: hang, FFprobe: hang})
	out := filepath.Join(t.TempDir(), "out.mp4")
	calls := map[string]func(ctx context.Context) error{
		"Sample":   func(ctx context.Context) error { _, err := s.Sample(ctx, "clip.mp4"); return err },
		"Duration": func(ctx context.Context) error { _, err := s.Duration(ctx, "clip.mp4"); return err },
		"Cut":      func(ctx context.Context) error { return s.Cut(ctx, "clip.mp4", 0, 0, out) },
	}
	for name, call := range calls {
		_ = os.Remove(pidFile)
		took, err := cancelAfter(t, 300*time.Millisecond, call)
		if !errors.Is(err, context.Canceled) {
			t.Errorf("%s: err = %v, want context.Canceled", name, err)
		}
		if took > 2*time.Second {
			t.Errorf("%s returned %v after the cancellation", name, took)
		}
		gone(t, pidFile)
	}
}

// TestCancellation_AHeldPipeCannotBlockForever: the program leaves a child
// behind that keeps its output pipes open, so killing the program alone
// never ends the reads. The wait delay closes the pipes instead.
func TestCancellation_AHeldPipeCannotBlockForever(t *testing.T) {
	dir := t.TempDir()
	pidFile, childFile := filepath.Join(dir, "pid"), filepath.Join(dir, "child")
	leaky := script(t, "sleep 60 &\n"+`echo $! > "`+childFile+`"`+"\n"+recordPID(pidFile)+"exec sleep 60\n")
	t.Cleanup(func() {
		// the child the script left behind, by the pid the script recorded
		if data, err := os.ReadFile(childFile); err == nil {
			if pid, err := strconv.Atoi(strings.TrimSpace(string(data))); err == nil && pid > 1 {
				_ = syscall.Kill(pid, syscall.SIGKILL)
			}
		}
	})
	samplers := map[string]Config{
		"probe":  {FFmpeg: leaky, FFprobe: leaky},
		"sample": {FFmpeg: leaky, FFprobe: fakeProbe(t)},
	}
	for name, cfg := range samplers {
		s, _ := New(cfg)
		s.waitDelay = 200 * time.Millisecond
		_ = os.Remove(pidFile)
		took, err := cancelAfter(t, 300*time.Millisecond, func(ctx context.Context) error {
			_, err := s.Sample(ctx, "clip.mp4")
			return err
		})
		if !errors.Is(err, context.Canceled) {
			t.Errorf("%s: err = %v, want context.Canceled", name, err)
		}
		if took > 3*time.Second {
			t.Errorf("%s: Sample returned %v after the cancellation, held by the child's pipe", name, took)
		}
		gone(t, pidFile)
	}
}

// TestSample_NoFrameIsAnError: media.Sampler promises an error, never an
// empty result, for a file no frame comes out of. The real ffmpeg fails on
// its own for every such file tried, so a program that succeeds without
// writing a frame stands in for it.
func TestSample_NoFrameIsAnError(t *testing.T) {
	s, _ := New(Config{FFmpeg: script(t, "exit 0\n"), FFprobe: fakeProbe(t)})
	frames, err := s.Sample(context.Background(), "clip.mp4")
	if err == nil || frames != nil {
		t.Fatalf("Sample = %d frames, %v; want an error", len(frames), err)
	}
	if !strings.Contains(err.Error(), "no frame") {
		t.Errorf("err = %v, want it to say that there was no frame", err)
	}
}

// TestSample_StopsAtMaxFramesAndKillsFFmpeg: a program that ignores
// -frames:v and never stops writing frames is cut off at MaxFrames, killed
// and reaped, and what was read is returned.
func TestSample_StopsAtMaxFramesAndKillsFFmpeg(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pid")
	// one 1x1 PAM frame and its showinfo line per turn, for ever
	endless := script(t, recordPID(pidFile)+`n=0
while :; do
  echo "[Parsed_showinfo_2 @ 0x1] n:   $n pts:  $n pts_time:$n  duration: 1 fmt:rgba s:1x1 i:P" >&2
  printf 'P7\nWIDTH 1\nHEIGHT 1\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\nabcd'
  n=$((n+1))
done
`)
	s, _ := New(Config{FFmpeg: endless, FFprobe: fakeProbe(t), MaxFrames: 5})
	type result struct {
		frames []media.Frame
		err    error
	}
	done := make(chan result, 1)
	go func() {
		frames, err := s.Sample(context.Background(), "clip.mp4")
		done <- result{frames, err}
	}()
	var res result
	select {
	case res = <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Sample is still reading a program that never stops")
	}
	if res.err != nil || len(res.frames) != 5 {
		t.Fatalf("Sample = %d frames, %v; want the 5 allowed", len(res.frames), res.err)
	}
	if res.frames[4].At != 4*time.Second {
		t.Errorf("frame 4 at %v, want 4 s", res.frames[4].At)
	}
	gone(t, pidFile)
}

// TestSample_FailureCarriesTheEndOfTheLog: a failing run's error ends with
// what the program said last, and stays short however much it said.
func TestSample_FailureCarriesTheEndOfTheLog(t *testing.T) {
	noisy := script(t, `i=0
while [ $i -lt 5000 ]; do echo "noise noise noise noise noise noise noise noise $i" >&2; i=$((i+1)); done
echo "the real reason" >&2
exit 3
`)
	s, _ := New(Config{FFmpeg: noisy, FFprobe: fakeProbe(t)})
	_, err := s.Sample(context.Background(), "clip.mp4")
	if err == nil || !strings.Contains(err.Error(), "exit status 3") || !strings.HasSuffix(err.Error(), "the real reason") {
		t.Fatalf("err = %.300v, want the exit status and the last line", err)
	}
	if len(err.Error()) > 4<<10 {
		t.Errorf("the error is %d bytes long", len(err.Error()))
	}
}

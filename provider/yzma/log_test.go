package yzma

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// TestCancelLine: the error lines an aborted evaluation makes llama.cpp and
// libmtmd log are told from genuine decode failures. A status that says
// aborted is the cancellation's whatever else runs; libmtmd's lines, which
// carry no status, only while an abort is in flight; everything else stays
// an error.
func TestCancelLine(t *testing.T) {
	for _, tc := range []struct {
		msg        string
		cancelling bool
		want       bool
	}{
		{"llama_decode: failed to decode, ret = 2", false, true},
		{"llama_decode: failed to decode, ret = 2", true, true},
		{"process_ubatch: failed to compute graph, compute status: 1", false, true},
		{"graph_compute: ggml_backend_sched_graph_compute_async failed with error 1", false, true},
		{"failed to decode text", true, true},
		{"failed to decode image", true, true},
		{"failed to eval chunk 3", true, true},
		// genuine failures
		{"llama_decode: failed to decode, ret = -1", true, false},
		{"llama_decode: failed to decode, ret = 1", true, false},
		{"llama_decode: failed to decode, ret = 22", true, false},
		{"process_ubatch: failed to compute graph, compute status: -2", true, false},
		{"graph_compute: ggml_backend_sched_graph_compute_async failed with error -1", true, false},
		{"failed to decode text", false, false},
		{"failed to eval chunk 3", false, false},
		{"failed to load model", true, false},
		{"decode: no KV slot for the batch", true, false},
	} {
		if got := cancelLine(tc.msg, tc.cancelling); got != tc.want {
			t.Errorf("cancelLine(%q, cancelling %v) = %v, want %v", tc.msg, tc.cancelling, got, tc.want)
		}
	}
}

// TestForwardLog_DemotesWhatACancellationCauses: the lines classified as a
// cancellation's go out at Debug, marked so, and are gone at Info; a
// genuine failure stays at Error. trackAbort counts an evaluation as
// cancelling from the first time its stop says so until untrackAbort.
func TestForwardLog_DemotesWhatACancellationCauses(t *testing.T) {
	orig := llamaLog.Load()
	t.Cleanup(func() { llamaLog.Store(orig) })
	var debug, info bytes.Buffer
	fail := int32(llama.LogLevelError)

	stopNow := false
	stop := trackAbort(42, func() bool { return stopNow })
	if stop() || cancelling.Load() != 0 {
		t.Fatal("an abort callback that said go on counts as cancelling")
	}
	stopNow = true
	first, again := stop(), stop()
	if !first || !again || cancelling.Load() != 1 {
		t.Fatalf("after stop: cancelling = %d, want 1 however often it is asked", cancelling.Load())
	}
	for _, out := range []*bytes.Buffer{&debug, &info} {
		level := slog.LevelInfo
		if out == &debug {
			level = slog.LevelDebug
		}
		llamaLog.Store(slog.New(slog.NewTextHandler(out, &slog.HandlerOptions{Level: level})))
		forwardLog(fail, "llama_decode: failed to decode, ret = 2\n")
		forwardLog(fail, "failed to decode text\n")
		forwardLog(fail, "llama_decode: failed to decode, ret = -1\n")
	}
	untrackAbort(42)
	if cancelling.Load() != 0 {
		t.Errorf("after untrackAbort: cancelling = %d", cancelling.Load())
	}
	forwardLog(fail, "failed to decode text\n") // to the Info logger: nothing is cancelled now

	if got := debug.String(); strings.Count(got, "level=DEBUG") != 2 || strings.Count(got, "cancelled=true") != 2 || !strings.Contains(got, `level=ERROR msg="llama_decode: failed to decode, ret = -1"`) {
		t.Errorf("at Debug:\n%s", got)
	}
	if got := info.String(); strings.Count(got, "\n") != 2 || !strings.Contains(got, `level=ERROR msg="llama_decode: failed to decode, ret = -1"`) || !strings.Contains(got, `level=ERROR msg="failed to decode text"`) {
		t.Errorf("at Info:\n%s", got)
	}
}

// TestForwardLog_FollowsTheLatestLogger: llama.cpp has one log callback per
// process. The first open must not fix where its lines go: an open without
// a logger drops them, and a later one with a logger gets them, as does a
// second logger after the first.
func TestForwardLog_FollowsTheLatestLogger(t *testing.T) {
	orig := llamaLog.Load()
	llamaLog.Store(nil)
	t.Cleanup(func() { llamaLog.Store(orig) })
	warn, fail, info := int32(llama.LogLevelWarn), int32(llama.LogLevelError), int32(llama.LogLevelInfo)
	var first, second bytes.Buffer
	a := slog.New(slog.NewTextHandler(&first, nil))
	b := slog.New(slog.NewTextHandler(&second, nil))
	f := newFakeAPI()
	rt := &runtimeLoader{}

	// the first opener has no logger: nothing is logged, and nothing breaks
	if err := rt.load(f, testLibDir, nil); err != nil {
		t.Fatal(err)
	}
	forwardLog(warn, "dropped: nobody listens yet\n")

	// a later opener brings one, and gets the lines from then on
	if err := rt.load(f, testLibDir, a); err != nil {
		t.Fatal(err)
	}
	forwardLog(warn, "  slow path taken\n")
	forwardLog(fail, "failed to load model")
	forwardLog(info, "progress, not forwarded")
	forwardLog(warn, "init: embeddings required but some input tokens were not marked as outputs -> overriding")
	forwardLog(warn, " \n")
	got := first.String()
	if strings.Count(got, "\n") != 2 || !strings.Contains(got, `level=WARN msg="slow path taken" component=llama.cpp`) || !strings.Contains(got, `level=ERROR msg="failed to load model" component=llama.cpp`) {
		t.Errorf("the first logger got:\n%s", got)
	}

	// an opener without a logger leaves the destination alone
	if err := rt.load(f, testLibDir, nil); err != nil {
		t.Fatal(err)
	}
	forwardLog(warn, "still to the first")
	if !strings.Contains(first.String(), "still to the first") {
		t.Error("an open without a logger silenced llama.cpp")
	}

	// and another logger takes over
	if err := rt.load(f, testLibDir, b); err != nil {
		t.Fatal(err)
	}
	forwardLog(warn, "to the second")
	if !strings.Contains(second.String(), "to the second") || strings.Contains(first.String(), "to the second") {
		t.Errorf("after a second logger: first %q, second %q", first.String(), second.String())
	}
	if n := f.count("load"); n != 1 {
		t.Errorf("the libraries were loaded %d times", n)
	}
}

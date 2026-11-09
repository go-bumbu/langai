package yzma

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"github.com/hybridgroup/yzma/pkg/llama"
)

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

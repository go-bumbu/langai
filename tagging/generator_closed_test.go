package tagging

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/sensetest"
)

// TestGenerate_DoesNotAskAClosedModelAgain: a Completer that was closed
// answers every prompt with an error wrapping sense.ErrClosed. It used to be
// asked twice for each window of a long text, 22 times for 64 KB. Asking it
// again cannot help: the first such error is the answer.
func TestGenerate_DoesNotAskAClosedModelAgain(t *testing.T) {
	closed := fmt.Errorf("yzma: completer: %w", sense.ErrClosed)
	for _, tc := range []struct {
		name string
		text string
	}{
		{"a short text", bread},
		{"a text of three windows", threeWindows},
		{"a text of 64 KB", strings.TrimSpace("A title\n" + strings.Repeat("word and more words. ", 3000))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			llm, emb := &sensetest.FakeCompleter{Err: closed}, newFlat()
			g := newLongGenerator(t, llm, emb, GeneratorConfig{})
			got, err := g.Generate(context.Background(), tc.text, 0)
			if !errors.Is(err, sense.ErrClosed) || errors.Is(err, ErrPartial) || got != nil {
				t.Fatalf("got %v, %v; want no tags and an error wrapping sense.ErrClosed, not ErrPartial", got, err)
			}
			if n := len(llm.Calls()); n != 1 || len(emb.calls) != 0 {
				t.Errorf("asked %d times and embedded %v, want one question and nothing embedded", n, emb.calls)
			}
		})
	}
}

// TestGenerate_LongTextStopsAtAModelClosedHalfway: the model is closed while
// a long text is read. The windows read before count, as for any window that
// is lost, and no window after it is asked about.
func TestGenerate_LongTextStopsAtAModelClosedHalfway(t *testing.T) {
	closed := fmt.Errorf("yzma: completer: %w", sense.ErrClosed)
	ws := windowsOf(t, threeWindows, 3)
	llm := &sensetest.FakeCompleter{}
	llm.Reply = func(_, user string) (string, error) {
		if user == ws[0] {
			return "alpha", nil
		}
		return "", closed
	}
	g := newLongGenerator(t, llm, newFlat(), GeneratorConfig{})
	got, err := g.Generate(context.Background(), threeWindows, 0)
	if !errors.Is(err, ErrPartial) || !errors.Is(err, sense.ErrClosed) || !strings.Contains(err.Error(), "window 2 of 3") {
		t.Fatalf("err = %v, want one wrapping ErrPartial and sense.ErrClosed, naming window 2 of 3", err)
	}
	if !strings.Contains(err.Error(), "2 of 3 windows") {
		t.Errorf("err = %v, want it to say that 2 of 3 windows are missing", err)
	}
	requireTags(t, got, []Tag{{"alpha", 1}})
	if want := []string{ws[0], ws[1]}; !slices.Equal(users(llm.Calls()), want) {
		t.Errorf("asked %d times, want the first window and the second once", len(llm.Calls()))
	}
}

// TestGenerate_LongTextGivesUpOnAModelThatIsDown: a remote model that is down
// costs a timeout per question. With no window read after
// maxFailedWindowsAtStart of them failed, each asked twice, the rest of the
// text is not asked about: the next windows would time out the same way.
func TestGenerate_LongTextGivesUpOnAModelThatIsDown(t *testing.T) {
	long := strings.TrimSpace("A title\n" + strings.Repeat("word and more words. ", 3000))
	if n := len(Chunk(long, windowBytes)); n < 10 {
		t.Fatalf("the text makes %d windows, want ten or more", n)
	}
	for name, tc := range map[string]struct {
		llm   *sensetest.FakeCompleter
		cause error
	}{
		"the model fails":           {&sensetest.FakeCompleter{Err: errBoom}, errBoom},
		"the model replies nothing": {&sensetest.FakeCompleter{Replies: []string{"\n"}}, ErrEmptyReply},
	} {
		t.Run(name, func(t *testing.T) {
			emb := newFlat()
			g := newLongGenerator(t, tc.llm, emb, GeneratorConfig{})
			got, err := g.Generate(context.Background(), long, 0)
			if !errors.Is(err, tc.cause) || errors.Is(err, ErrPartial) || got != nil {
				t.Fatalf("got %v, %v; want no tags and an error wrapping the cause, not ErrPartial", got, err)
			}
			if n, want := len(tc.llm.Calls()), 2*maxFailedWindowsAtStart; n != want || len(emb.calls) != 0 {
				t.Errorf("asked %d times and embedded %v, want %d questions and nothing embedded", n, emb.calls, want)
			}
		})
	}

	// a window that was read shows the model is up: every later window is
	// asked about, however many of them fail
	t.Run("not once a window was read", func(t *testing.T) {
		ws := Chunk(long, windowBytes)
		llm := &sensetest.FakeCompleter{}
		llm.Reply = func(_, user string) (string, error) {
			if user == ws[0] {
				return "alpha", nil
			}
			return "", errBoom
		}
		g := newLongGenerator(t, llm, newFlat(), GeneratorConfig{})
		got, err := g.Generate(context.Background(), long, 0)
		if !errors.Is(err, ErrPartial) || !errors.Is(err, errBoom) {
			t.Fatalf("err = %v, want one wrapping ErrPartial and the cause", err)
		}
		requireTags(t, got, []Tag{{"alpha", 1}})
		if n, want := len(llm.Calls()), 1+2*(len(ws)-1); n != want {
			t.Errorf("asked %d times, want %d: the first window once and every other twice", n, want)
		}
	})
}

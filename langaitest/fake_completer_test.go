package langaitest_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/go-bumbu/langai/langaitest"
)

// This example scripts a chat model's replies and reads back what it was
// asked.
func ExampleFakeCompleter() {
	ctx := context.Background()
	chat := &langaitest.FakeCompleter{Replies: []string{"bread\nbaking", "travel"}}

	first, _ := chat.Complete(ctx, "Write tags, one per line.", "Baking sourdough bread at home")
	second, _ := chat.Complete(ctx, "Write tags, one per line.", "Three days in Lisbon")
	fmt.Printf("%q %q\n", first, second)

	for _, call := range chat.Calls() {
		fmt.Println(call.User)
	}

	// Output:
	// "bread\nbaking" "travel"
	// Baking sourdough bread at home
	// Three days in Lisbon
}

func TestFakeCompleter_Replies(t *testing.T) {
	boom := errors.New("boom")
	bySystem := func(system, _ string) (string, error) {
		if strings.Contains(system, "tree") {
			return "a / b", nil
		}
		return "a", nil
	}
	cases := []struct {
		name    string
		fake    *langaitest.FakeCompleter
		systems []string // one call each, with the user message "text"
		want    []string
		wantErr error
	}{
		{name: "the zero value answers with nothing", fake: &langaitest.FakeCompleter{},
			systems: []string{"s", "s"}, want: []string{"", ""}},
		{name: "one reply answers every call", fake: &langaitest.FakeCompleter{Replies: []string{"only"}},
			systems: []string{"s", "s", "s"}, want: []string{"only", "only", "only"}},
		{name: "replies come in order and the last one repeats", fake: &langaitest.FakeCompleter{Replies: []string{"one", "two"}},
			systems: []string{"s", "s", "s"}, want: []string{"one", "two", "two"}},
		{name: "the function sees the prompt", fake: &langaitest.FakeCompleter{Reply: bySystem},
			systems: []string{"flat", "tree"}, want: []string{"a", "a / b"}},
		{name: "the function wins over the replies", fake: &langaitest.FakeCompleter{Reply: bySystem, Replies: []string{"unused"}},
			systems: []string{"flat"}, want: []string{"a"}},
		{name: "the function's error comes back", fake: &langaitest.FakeCompleter{Reply: func(string, string) (string, error) { return "", boom }},
			systems: []string{"s"}, want: []string{""}, wantErr: boom},
		{name: "Err fails every call", fake: &langaitest.FakeCompleter{Err: boom, Replies: []string{"unused"}},
			systems: []string{"s", "s"}, want: []string{"", ""}, wantErr: boom},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for i, system := range tc.systems {
				got, err := tc.fake.Complete(context.Background(), system, "text")
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("call %d: err = %v, want %v", i, err, tc.wantErr)
				}
				if got != tc.want[i] {
					t.Errorf("call %d: reply = %q, want %q", i, got, tc.want[i])
				}
			}
			calls := tc.fake.Calls()
			if len(calls) != len(tc.systems) {
				t.Fatalf("recorded %d calls, want %d (failed ones too)", len(calls), len(tc.systems))
			}
			for i, call := range calls {
				if call.System != tc.systems[i] || call.User != "text" {
					t.Errorf("call %d recorded as %+v", i, call)
				}
			}
		})
	}
}

// TestFakeCompleter_ReplyMayReadCalls: the call is recorded before Reply
// runs and no lock is held around it, so a reply function can count the
// calls so far, e.g. to fail the second one.
func TestFakeCompleter_ReplyMayReadCalls(t *testing.T) {
	boom := errors.New("boom")
	fake := &langaitest.FakeCompleter{}
	fake.Reply = func(string, string) (string, error) {
		if len(fake.Calls()) == 2 {
			return "", boom
		}
		return "ok", nil
	}
	var errs []error
	for range 3 {
		_, err := fake.Complete(context.Background(), "s", "u")
		errs = append(errs, err)
	}
	if errs[0] != nil || !errors.Is(errs[1], boom) || errs[2] != nil {
		t.Errorf("errors = %v, want only the second call to fail", errs)
	}
}

// TestFakeCompleter_CancelledContext: like a real model, the fake does not
// answer a caller that has already given up, and still records the call.
func TestFakeCompleter_CancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	fake := &langaitest.FakeCompleter{Replies: []string{"unused"}}
	if got, err := fake.Complete(ctx, "s", "u"); got != "" || !errors.Is(err, context.Canceled) {
		t.Errorf("Complete = %q, %v; want context.Canceled", got, err)
	}
	if len(fake.Calls()) != 1 {
		t.Errorf("recorded %d calls, want 1", len(fake.Calls()))
	}
}

// TestFakeCompleter_CallsIsACopy: what Calls returns is the test's to keep
// and change; the record does not move under it.
func TestFakeCompleter_CallsIsACopy(t *testing.T) {
	fake := &langaitest.FakeCompleter{}
	_, _ = fake.Complete(context.Background(), "s", "first")
	calls := fake.Calls()
	calls[0].User = "changed"
	_, _ = fake.Complete(context.Background(), "s", "second")
	want := []langaitest.CompleterCall{{System: "s", User: "first"}, {System: "s", User: "second"}}
	if got := fake.Calls(); !slices.Equal(got, want) {
		t.Errorf("Calls() = %+v, want %+v", got, want)
	}
}

// TestFakeCompleter_ConcurrentUse runs under -race: every call is recorded
// once and the replies are handed out one each until the last repeats.
func TestFakeCompleter_ConcurrentUse(t *testing.T) {
	const n = 32
	fake := &langaitest.FakeCompleter{Replies: []string{"first", "rest"}}
	replies := make([]string, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Go(func() {
			replies[i], _ = fake.Complete(context.Background(), "s", fmt.Sprint(i))
			_ = fake.Calls()
		})
	}
	wg.Wait()
	if got := len(fake.Calls()); got != n {
		t.Fatalf("recorded %d calls, want %d", got, n)
	}
	firsts := 0
	for _, r := range replies {
		if r == "first" {
			firsts++
		}
	}
	if firsts != 1 {
		t.Errorf("%d calls got the first reply, want exactly 1", firsts)
	}
}

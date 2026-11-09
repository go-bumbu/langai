package langaitest

import (
	"context"
	"slices"
	"sync"

	"github.com/go-bumbu/langai"
)

// CompleterCall is one prompt a FakeCompleter received.
type CompleterCall struct {
	System, User string
}

// FakeCompleter is a scripted Completer for tests: it answers with canned
// replies or with a function of the prompt, fails on demand, and records
// every prompt it was asked. The zero value answers every prompt with an
// empty reply. Set the fields before the first call; Complete and Calls are
// safe for concurrent use.
//
//	chat := &langaitest.FakeCompleter{Replies: []string{"bread\nbaking"}}
//	g, err := tagging.NewGenerator(chat, langaitest.FakeEmbedder{}, tagging.GeneratorConfig{})
//	tags, err := g.Generate(ctx, "Baking sourdough bread at home", 0.3)
//	chat.Calls()[0].User // the text the Generator sent the model
//
// A reply that depends on the prompt takes a function instead:
//
//	chat := &langaitest.FakeCompleter{Reply: func(system, user string) (string, error) {
//		if strings.Contains(system, "JSON") {
//			return `{"tags": []}`, nil
//		}
//		return "", errors.New("unexpected prompt")
//	}}
type FakeCompleter struct {
	// Replies are the answers, one per call in order; the last one answers
	// every call after it. Ignored when Reply is set.
	Replies []string
	// Reply, when set, answers each call from its prompt. The call is
	// recorded before Reply runs, so len(Calls()) inside it is the number of
	// the call being answered, 1 for the first. It may run concurrently.
	Reply func(system, user string) (string, error)
	// Err, when set, fails every call with it, before Reply or Replies are
	// looked at.
	Err error

	mu    sync.Mutex
	calls []CompleterCall
}

// verify the interface is actually satisfied
var _ langai.Completer = (*FakeCompleter)(nil)

// Complete implements langai.Completer. It records the prompt, then fails
// with ctx's error when ctx is done, with Err when that is set, and
// otherwise answers with Reply or the next of Replies.
func (f *FakeCompleter) Complete(ctx context.Context, system, user string) (string, error) {
	f.mu.Lock()
	f.calls = append(f.calls, CompleterCall{System: system, User: user})
	n := len(f.calls)
	f.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return "", err
	}
	if f.Err != nil {
		return "", f.Err
	}
	if f.Reply != nil {
		return f.Reply(system, user)
	}
	if len(f.Replies) == 0 {
		return "", nil
	}
	return f.Replies[min(n, len(f.Replies))-1], nil
}

// Calls returns a copy of the prompts received so far, in the order they
// arrived, the failed ones included.
func (f *FakeCompleter) Calls() []CompleterCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

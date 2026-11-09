// Package langaitest provides fakes of the langai ports for tests: of this
// module's packages and of the code that uses them.
package langaitest

import (
	"context"
	"hash/fnv"
	"strings"
	"sync/atomic"
	"unicode"

	"github.com/go-bumbu/langai"
)

// FakeEmbedder is a deterministic Embedder for tests: it hashes words into a
// fixed number of buckets (a bag-of-words random projection), so texts that
// share words get similar vectors. It has no semantic understanding — it only
// makes word overlap measurable — but that is exactly what a test needs.
//
// The zero value embeds every text and never fails. It is used as a value or
// through a pointer, and may be copied: copies share Calls. Set the fields
// before the first call; Embed is safe for concurrent use when Fail is.
//
//	var calls atomic.Int64
//	emb := langaitest.FakeEmbedder{Calls: &calls, Fail: func(text string) error {
//		if strings.Contains(text, "poison") {
//			return errors.New("the model cannot take this text")
//		}
//		return nil
//	}}
type FakeEmbedder struct {
	// Dims is the vector size; 64 when zero.
	Dims int
	// Err, when set, fails every Embed with it: an embedder that is down.
	Err error
	// Fail, when set, is asked about each text of a call in order, and the
	// first error it returns fails the whole call, as a real embedder fails a
	// batch that holds a text it cannot take: a call with that text alone
	// fails, and one without it works. It may run concurrently.
	Fail func(text string) error
	// Calls, when set, counts the Embed calls, the failed ones included.
	Calls *atomic.Int64
}

// FakeEmbedderModel is the model name FakeEmbedder reports, whatever its
// Dims.
const FakeEmbedderModel = "langaitest/fake-embedder"

// verify the interfaces are actually satisfied
var (
	_ langai.Embedder   = FakeEmbedder{}
	_ langai.Identified = FakeEmbedder{}
)

// Model implements langai.Identified: FakeEmbedderModel.
func (FakeEmbedder) Model() string { return FakeEmbedderModel }

// Dimensions implements Embedder.
func (f FakeEmbedder) Dimensions() int {
	if f.Dims <= 0 {
		return 64
	}
	return f.Dims
}

// Embed implements Embedder. It counts the call in Calls, then fails with
// ctx's error when ctx is done, with Err when that is set, and with the first
// error Fail returns for one of texts; otherwise it embeds them all.
func (f FakeEmbedder) Embed(ctx context.Context, _ langai.Role, texts []string) ([][]float32, error) {
	if f.Calls != nil {
		f.Calls.Add(1)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if f.Err != nil {
		return nil, f.Err
	}
	if f.Fail != nil {
		for _, text := range texts {
			if err := f.Fail(text); err != nil {
				return nil, err
			}
		}
	}
	dims := f.Dimensions()
	out := make([][]float32, len(texts))
	for i, text := range texts {
		vec := make([]float32, dims)
		for _, word := range splitTokens(text) {
			h := fnv.New32a()
			_, _ = h.Write([]byte(word))
			sum := h.Sum32()
			bucket := int(sum % uint32(dims)) //nolint:gosec // dims is a small positive int
			sign := float32(1)
			if sum&0x80000000 != 0 {
				sign = -1
			}
			vec[bucket] += sign
		}
		langai.Normalize(vec)
		out[i] = vec
	}
	return out, nil
}

func splitTokens(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
}

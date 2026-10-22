package semsearch

import (
	"context"
	"hash/fnv"
	"strings"
	"unicode"
)

// FakeEmbedder is a deterministic Embedder for tests: it hashes words into a
// fixed number of buckets (a bag-of-words random projection), so texts that
// share words get similar vectors. It has no semantic understanding — it only
// makes word overlap measurable — but that is exactly what a test needs.
type FakeEmbedder struct {
	// Dims is the vector size; 64 when zero.
	Dims int
}

// Dimensions implements Embedder.
func (f FakeEmbedder) Dimensions() int {
	if f.Dims <= 0 {
		return 64
	}
	return f.Dims
}

// Embed implements Embedder.
func (f FakeEmbedder) Embed(_ context.Context, _ Role, texts []string) ([][]float32, error) {
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
		Normalize(vec)
		out[i] = vec
	}
	return out, nil
}

func splitTokens(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
}

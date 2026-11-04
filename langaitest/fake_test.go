package langaitest_test

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
)

func TestFakeEmbedder_Dimensions(t *testing.T) {
	if got := (langaitest.FakeEmbedder{}).Dimensions(); got != 64 {
		t.Errorf("zero value Dimensions() = %d, want 64", got)
	}
	if got := (langaitest.FakeEmbedder{Dims: 8}).Dimensions(); got != 8 {
		t.Errorf("Dimensions() with Dims 8 = %d, want 8", got)
	}
}

// TestFakeEmbedder_SharedWordsScoreHigher: the fake only makes word overlap
// measurable, so the same words in another case or with other punctuation give
// the same unit vector, and a text that shares words with another sits closer
// to it than one that shares none.
func TestFakeEmbedder_SharedWordsScoreHigher(t *testing.T) {
	emb := langaitest.FakeEmbedder{}
	texts := []string{"Go channels and goroutines", "go CHANNELS, and goroutines!", "goroutines in Go", "tomato pasta sauce"}
	vecs, err := emb.Embed(context.Background(), langai.RoleDocument, texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != len(texts) {
		t.Fatalf("got %d vectors for %d texts", len(vecs), len(texts))
	}
	for i, v := range vecs {
		if len(v) != emb.Dimensions() {
			t.Fatalf("vector %d has %d dimensions, want %d", i, len(v), emb.Dimensions())
		}
		if n := dot(v, v); math.Abs(n-1) > 1e-6 {
			t.Errorf("vector %d has squared length %f, want 1", i, n)
		}
	}
	if !slices.Equal(vecs[0], vecs[1]) {
		t.Error("case and punctuation changed the vector")
	}
	if related, unrelated := dot(vecs[0], vecs[2]), dot(vecs[0], vecs[3]); related <= unrelated {
		t.Errorf("shared words scored %f, no shared words %f, want the first higher", related, unrelated)
	}
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

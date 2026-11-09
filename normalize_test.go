package sense

import (
	"math"
	"testing"
)

// TestNormalize_ScalesToUnitLength: the vector keeps its direction and comes
// back at length 1, so cosine and dot product rank alike.
func TestNormalize_ScalesToUnitLength(t *testing.T) {
	v := []float32{3, 4}
	Normalize(v)
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Fatalf("Normalize([3 4]) = %v, want [0.6 0.8]", v)
	}
}

// TestNormalize_ZeroVectorStaysZero covers the divide-by-zero guard: an
// all-zero vector has no direction to scale, so it must be left alone rather
// than turned into NaNs.
func TestNormalize_ZeroVectorStaysZero(t *testing.T) {
	v := []float32{0, 0, 0}
	Normalize(v)
	for i, x := range v {
		if x != 0 {
			t.Fatalf("zero vector was modified at %d: %v", i, v)
		}
	}
}

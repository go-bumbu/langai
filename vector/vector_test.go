package vector_test

import (
	"errors"
	"math"
	"testing"

	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/vector/vectortest"
)

// TestMemStore_Conformance runs the suite against MemStore, which is also
// what lets `make coverage` measure vectortest.
func TestMemStore_Conformance(t *testing.T) {
	vectortest.Run(t, func(*testing.T) vector.Store { return vector.NewMemStore() })
	vectortest.RunShared(t, func(_ *testing.T, n int) []vector.Store {
		s := vector.NewMemStore()
		stores := make([]vector.Store, n)
		for i := range stores {
			stores[i] = s
		}
		return stores
	})
}

// BenchmarkMemStoreSearchVectors: a search with 1, 16 and 64 query vectors,
// the shape of a "more like this" search, over 20,000 chunks of 256
// dimensions.
func BenchmarkMemStoreSearchVectors(b *testing.B) {
	vectortest.BenchSearch(b, func(*testing.B) vector.Store { return vector.NewMemStore() })
}

func TestValidate(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(-1))
	for _, tc := range []struct {
		name string
		vec  []float32
		dims int
		want error
	}{
		{"unit", []float32{0, 1, 0}, 3, nil},
		{"not unit", []float32{3, 4, 0}, 3, nil},
		{"small", []float32{1e-15, 0, 0}, 3, nil},
		{"large", []float32{1e19, 1e19, 0}, 3, nil},
		// float32 squares, as the backends compute them, underflow to 0 or
		// overflow to infinity, though float64 ones would not
		{"tiny", []float32{1e-30, 0, 0}, 3, vector.ErrInvalidVector},
		{"tiny pair", []float32{1e-25, 1e-25, 0}, 3, vector.ErrInvalidVector},
		{"huge", []float32{1e20, 1e20, 0}, 3, vector.ErrInvalidVector},
		{"max", []float32{3.4028235e38, 0, 0}, 3, vector.ErrInvalidVector},
		{"short", []float32{1, 0}, 3, vector.ErrDimensions},
		{"long", []float32{1, 0, 0, 0}, 3, vector.ErrDimensions},
		{"empty", nil, 3, vector.ErrDimensions},
		{"zero", []float32{0, 0, 0}, 3, vector.ErrInvalidVector},
		{"NaN", []float32{1, nan, 0}, 3, vector.ErrInvalidVector},
		{"Inf", []float32{inf, 0, 0}, 3, vector.ErrInvalidVector},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := vector.Validate(tc.vec, tc.dims)
			if tc.want == nil && err != nil {
				t.Fatalf("Validate(%v, %d) = %v, want nil", tc.vec, tc.dims, err)
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Fatalf("Validate(%v, %d) = %v, want %v", tc.vec, tc.dims, err, tc.want)
			}
		})
	}
}

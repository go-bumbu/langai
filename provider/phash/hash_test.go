package phash

import "testing"

func TestHashStringAndBits(t *testing.T) {
	h := Hash{alg: AlgorithmPHash, bits: 64}
	h.set(0)
	h.set(15)
	h.set(63)
	if got, want := h.String(), "8001000000000001"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
	for i := range 64 {
		if want := i == 0 || i == 15 || i == 63; h.bit(i) != want {
			t.Errorf("bit(%d) = %v, want %v", i, h.bit(i), want)
		}
	}
	wide := Hash{alg: AlgorithmDHash, bits: 256}
	wide.set(64)
	wide.set(255)
	if got, want := wide.String(), "0000000000000000"+"8000000000000000"+"0000000000000000"+"0000000000000001"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
	if got := (Hash{}).String(); got != "" {
		t.Errorf("zero Hash: String() = %q, want empty", got)
	}
}

func TestHashDistance(t *testing.T) {
	a := Hash{alg: AlgorithmPHash, bits: 64}
	b := a
	if a != b || a.Distance(b) != 0 {
		t.Errorf("equal hashes: == is %v, Distance %d", a == b, a.Distance(b))
	}
	b.set(3)
	b.set(40)
	a.set(40)
	a.set(41)
	if a.Distance(b) != 2 || b.Distance(a) != 2 {
		t.Errorf("Distance = %d and %d, want 2 both ways", a.Distance(b), b.Distance(a))
	}
	if a == b {
		t.Error("different hashes compare equal")
	}

	wide := Hash{alg: AlgorithmPHash, bits: 256}
	for i := range 256 {
		wide.set(i)
	}
	if got := wide.Distance(Hash{alg: AlgorithmPHash, bits: 256}); got != 256 {
		t.Errorf("a hash and its negative: Distance = %d, want 256", got)
	}
}

// Hashes of two configurations are two vector spaces: never equal, and as
// far apart as the larger allows, even when their bits agree.
func TestHashesOfDifferentSpacesDoNotCompare(t *testing.T) {
	tests := []struct {
		name string
		a, b Hash
		want int
	}{
		{"algorithm", Hash{alg: AlgorithmPHash, bits: 64}, Hash{alg: AlgorithmDHash, bits: 64}, 64},
		{"size", Hash{alg: AlgorithmPHash, bits: 64}, Hash{alg: AlgorithmPHash, bits: 256}, 256},
		{"zero value", Hash{}, Hash{alg: AlgorithmDHash, bits: 64}, 64},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.a == tt.b {
				t.Error("hashes of different spaces compare equal")
			}
			if got := tt.a.Distance(tt.b); got != tt.want {
				t.Errorf("Distance = %d, want %d", got, tt.want)
			}
			if got := tt.b.Distance(tt.a); got != tt.want {
				t.Errorf("Distance the other way = %d, want %d", got, tt.want)
			}
		})
	}
}

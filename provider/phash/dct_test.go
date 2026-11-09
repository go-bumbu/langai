package phash

import (
	"math"
	"testing"
)

func TestQuarterIsTheRoundedCosine(t *testing.T) {
	for k, got := range quarter {
		exact := math.Cos(float64(k)*math.Pi/128) * 65536
		if want := int64(math.Round(exact)); got != want {
			t.Errorf("quarter[%d] = %d, want %d", k, got, want)
		}
	}
}

func TestCosineFollowsTheWave(t *testing.T) {
	for m := range 1024 {
		want := math.Cos(float64(m)*math.Pi/128) * 65536
		if got := float64(cosine(m)); math.Abs(got-want) > 0.5 {
			t.Errorf("cosine(%d) = %v, want %v", m, got, want)
		}
	}
}

// The cosines of every frequency but zero cancel exactly over the samples:
// what makes a picture that is constant along an axis come out with exact
// zeros instead of rounding residue.
func TestBasisCancelsAboveFrequencyZero(t *testing.T) {
	for _, side := range []int{8, 16} {
		n := 4 * side
		basis := dctBasis(side, n)
		for k := range side {
			var sum int64
			for x := range n {
				sum += basis[k*n+x]
			}
			want := int64(0)
			if k == 0 {
				want = int64(n) * 65536
			}
			if sum != want {
				t.Errorf("%d samples, frequency %d: the cosines add up to %d, want %d", n, k, sum, want)
			}
		}
	}
}

func TestLowFrequenciesMatchTheFloatDCT(t *testing.T) {
	for _, side := range []int{8, 16} {
		n := 4 * side
		r := rng(uint64(side))
		grey := make([]int64, n*n)
		for i := range grey {
			grey[i] = int64(r.intn(256))
		}
		got := lowFrequencies(grey, n, side, dctBasis(side, n))
		for u := range side {
			for v := range side {
				var want float64
				for y := range n {
					for x := range n {
						want += float64(grey[y*n+x]) *
							math.Cos(float64(2*y+1)*float64(u)*math.Pi/float64(2*n)) *
							math.Cos(float64(2*x+1)*float64(v)*math.Pi/float64(2*n))
					}
				}
				// 16 fractional bits per pass; the table's rounding is at
				// most half a unit on each of the n*n terms of up to 255
				scaled := float64(got[u*side+v]) / (65536 * 65536)
				if math.Abs(scaled-want) > 0.02*float64(n) {
					t.Errorf("%d×%d: coefficient (%d,%d) = %v, want %v", n, n, u, v, scaled, want)
				}
			}
		}
	}
}

func TestLowFrequenciesOfAOneDimensionalPictureAreExactZeros(t *testing.T) {
	const side, n = 8, 32
	basis := dctBasis(side, n)
	r := rng(7)
	stripes := make([]int64, n)
	for i := range stripes {
		stripes[i] = int64(r.intn(256))
	}
	columns, rows := make([]int64, n*n), make([]int64, n*n)
	for y := range n {
		for x := range n {
			columns[y*n+x] = stripes[x] // constant from top to bottom
			rows[y*n+x] = stripes[y]    // constant from left to right
		}
	}
	across, down := lowFrequencies(columns, n, side, basis), lowFrequencies(rows, n, side, basis)
	for u := range side {
		for v := range side {
			if u > 0 && across[u*side+v] != 0 {
				t.Errorf("vertical stripes: coefficient (%d,%d) = %d, want 0", u, v, across[u*side+v])
			}
			if v > 0 && down[u*side+v] != 0 {
				t.Errorf("horizontal stripes: coefficient (%d,%d) = %d, want 0", u, v, down[u*side+v])
			}
		}
	}
}

func TestMedian(t *testing.T) {
	values := []int64{5, -3, 9, 0, 5}
	if got := median(values); got != 5 {
		t.Errorf("median = %d, want 5", got)
	}
	if values[1] != -3 {
		t.Error("median reordered its argument")
	}
}

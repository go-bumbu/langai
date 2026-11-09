package phash

import "slices"

// quarter is cos(k·π/128) for k = 0 to 64 as 16.16 fixed point, rounded to
// nearest: the first quarter of the cosine wave, from which cosine reads the
// rest. The table is written out rather than computed with math.Cos so that
// no floating point is involved in a hash at all: the bits cannot depend on
// the platform's math library or on whether the compiler fuses a multiply and
// an add. dct_test.go checks it against math.Cos.
var quarter = [65]int64{
	65536, 65516, 65457, 65358, 65220, 65043, 64827, 64571,
	64277, 63944, 63572, 63162, 62714, 62228, 61705, 61145,
	60547, 59914, 59244, 58538, 57798, 57022, 56212, 55368,
	54491, 53581, 52639, 51665, 50660, 49624, 48559, 47464,
	46341, 45190, 44011, 42806, 41576, 40320, 39040, 37736,
	36410, 35062, 33692, 32303, 30893, 29466, 28020, 26558,
	25080, 23586, 22078, 20557, 19024, 17479, 15924, 14359,
	12785, 11204, 9616, 8022, 6424, 4821, 3216, 1608,
	0,
}

// cosine is cos(m·π/128) in 16.16 fixed point for any m >= 0, read from
// quarter by the wave's symmetries, which therefore hold exactly: the cosines
// of a frequency above zero add up to exactly zero over the samples, so a
// picture that does not change along an axis has coefficients of exactly
// zero for every frequency along it, not rounding noise.
func cosine(m int) int64 {
	m %= 256
	if m > 128 {
		m = 256 - m
	}
	if m > 64 {
		return -quarter[128-m]
	}
	return quarter[m]
}

// dctBasis returns the cosines of the DCT-II for the freqs lowest
// frequencies over n samples, n being 32 or 64: entry k*n+x is
// cos((2x+1)·k·π/2n).
func dctBasis(freqs, n int) []int64 {
	step := 64 / n // (2x+1)·k·π/2n in units of π/128
	basis := make([]int64, freqs*n)
	for k := range freqs {
		for x := range n {
			basis[k*n+x] = cosine((2*x + 1) * k * step)
		}
	}
	return basis
}

// lowFrequencies returns the freqs×freqs lowest-frequency coefficients of
// the two-dimensional DCT-II of the n×n grid grey, row by row: entry
// u*freqs+v is the coefficient of vertical frequency u and horizontal
// frequency v, entry 0 the DC term. The coefficients are not normalized:
// every one but the DC term would be scaled by the same factor, which no
// comparison between them notices. Levels up to 255, 64 samples and 16
// fractional bits per pass keep every sum under 2^52.
func lowFrequencies(grey []int64, n, freqs int, basis []int64) []int64 {
	// rows first: rowwise[y*freqs+v] is row y's coefficient of frequency v
	rowwise := make([]int64, n*freqs)
	for y := range n {
		row := grey[y*n : (y+1)*n]
		for v := range freqs {
			var sum int64
			for x, level := range row {
				sum += level * basis[v*n+x]
			}
			rowwise[y*freqs+v] = sum
		}
	}
	out := make([]int64, freqs*freqs)
	for u := range freqs {
		for v := range freqs {
			var sum int64
			for y := range n {
				sum += rowwise[y*freqs+v] * basis[u*n+y]
			}
			out[u*freqs+v] = sum
		}
	}
	return out
}

// median is the middle value of an odd number of values.
func median(values []int64) int64 {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	return sorted[len(sorted)/2]
}

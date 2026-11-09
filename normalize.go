package sense

import "math"

// Normalize scales vec to unit length in place, so cosine similarity and dot
// product rank alike. The model providers call it on what their models
// return. An all-zero vector has no direction and stays zero; the vector
// stores refuse one (see vector.Validate).
func Normalize(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return
	}
	inv := 1 / math.Sqrt(sum)
	for i := range vec {
		vec[i] = float32(float64(vec[i]) * inv)
	}
}

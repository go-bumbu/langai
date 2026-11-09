package phash

import (
	"fmt"
	"math/bits"
	"strings"
)

// maxWords is the 64-bit words of the largest hash, 256 bits.
const maxWords = 4

// Hash is the perceptual hash of one picture: the bits an Embedder's vector
// is made of, for callers that group duplicates without a vector store. It
// is comparable: two pictures with equal hashes (==, or as map keys) are
// exact duplicates as far as the algorithm sees. A Hash remembers the
// algorithm and size that made it, so hashes of two different Embedder
// configurations are never equal.
type Hash struct {
	alg   Algorithm
	bits  int
	words [maxWords]uint64 // bit i is bit 63-i%64 of words[i/64], so String reads row by row
}

// set turns bit i on.
func (h *Hash) set(i int) {
	h.words[i/64] |= 1 << (63 - i%64)
}

// bit reports whether bit i is on.
func (h Hash) bit(i int) bool {
	return h.words[i/64]>>(63-i%64)&1 == 1
}

// Distance is the Hamming distance to other: the number of bits that differ,
// 0 for exact duplicates and about half the bits for unrelated pictures. The
// cosine of the two pictures' vectors is 1 - 2*Distance/bits. Hashes of
// different algorithms or sizes do not compare: for those Distance is the
// larger of the two sizes, the distance between a picture and its negative,
// so that no threshold takes them for duplicates.
func (h Hash) Distance(other Hash) int {
	if h.alg != other.alg || h.bits != other.bits {
		return max(h.bits, other.bits)
	}
	var d int
	for i := range h.words {
		d += bits.OnesCount64(h.words[i] ^ other.words[i])
	}
	return d
}

// String is the hash in hexadecimal, 16 digits for 64 bits and 64 for 256,
// the first bit first: for a 64-bit hash every two digits are one row of the
// 8×8 grid the bits come from. The zero Hash is the empty string.
func (h Hash) String() string {
	var sb strings.Builder
	for i := range h.bits / 64 {
		fmt.Fprintf(&sb, "%016x", h.words[i])
	}
	return sb.String()
}

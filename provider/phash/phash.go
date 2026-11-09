// Package phash hashes pictures perceptually, as a langai.ImageEmbedder
// without a model: a picture becomes 64 (or 256) bits that survive a
// re-encode, a resize or a small edit, and the bits become a unit-length
// vector of +1/sqrt(bits) and -1/sqrt(bits) entries. The cosine of two such
// vectors is 1 - 2*hamming/bits, so search.MediaIndex and every vector.Store
// work on them unchanged. It is pure Go and loads and downloads nothing: for
// de-duplication jobs that should not pay for a model.
//
// It finds exact and near-exact duplicates: the same picture saved again as
// a JPEG, scaled up or down, slightly brightened or darkened, or with a
// small mark stamped on it. It does not find a crop, a rotated or mirrored
// copy, or a picture that merely looks similar (another photo of the same
// dog): to the hash those are different pictures, and they are what the
// model embedders (provider/onnx, provider/yzma) are for. A picture with
// almost no structure has almost nothing to hash: every flat picture,
// whatever its colour, has the all-zero hash, and on a plain background a
// small mark is most of what there is.
//
// AlgorithmPHash, the default, averages the picture into a 32×32 grid of
// grey levels, takes the 8×8 lowest frequencies of its DCT and sets a bit
// for each that is above their median. AlgorithmDHash averages it into a 9×8
// grid and sets a bit wherever a cell is brighter than its left neighbour.
// Every pHash bit looks at the whole picture and the median sets about half
// of them, so unrelated pictures agree by chance least often. Every dHash bit
// looks at two neighbouring cells, so a local edit flips only the bits under
// it, and unrelated pictures come closer. Config.Bits selects 256 bits
// instead of 64 (a 64×64 grid and 16×16 frequencies, or a 17×16 grid), which
// puts duplicates and unrelated pictures much further apart. Reading the
// pixels is nearly all of the cost, the same for all four: about 2 ms for a
// 1920×1080 RGBA and 5 ms for a decoded JPEG of that size on one core of a
// laptop (BenchmarkHash).
//
// A score is a Hamming distance read another way, 1 - 2*distance/bits: with
// 64 bits a score of 1 is distance 0, 0.875 is 4, 0.75 is 8, 0.6875 is 10,
// and 0 is 32, where unrelated pictures sit on average. These are the most
// bits a copy moved on this package's test pictures (smooth colour fields
// with shapes on them, see phash_test.go), plain copies being the ones
// saved as a JPEG of quality 60, scaled to half or double size or
// brightened, and how close the closest of 79,800 unrelated pairs came:
//
//	            plain copy   with a mark   closest unrelated pair
//	phash-64    2            10            14
//	dhash-64    2            2             11
//	phash-256   6            46            80
//	dhash-256   9            5             56
//
// So with the 64-bit pHash a floor of 0.875 (search.WithMediaMinScore; 4
// bits) keeps the plain copies and 0.6875 (10 bits) the marked ones too,
// while under about 0.6 a large collection starts to return unrelated
// pictures. With 256 bits one floor of 0.6 (51 bits) keeps every copy
// measured and is far from any unrelated pair (80 bits is 0.375). Photos are
// not these pictures: measure on yours before trusting a floor. A caller
// that only groups duplicates can skip the store and compare Hash values.
//
// Everything is integer arithmetic over defined steps, so the same pixels
// give the same bits on every OS and architecture: transparency is composed
// on white, colours are reduced to 8-bit sRGB, the picture is shrunk by an
// exact area average, a cell's grey level is the Rec. 601 luma of its mean
// colour, and the DCT reads a fixed-point cosine table. An Embedder has
// nothing to load or release, so there is no Close, and it is safe for
// concurrent use.
package phash

import (
	"context"
	"fmt"
	"image"

	"github.com/go-bumbu/langai"
)

// Algorithm selects how a picture becomes bits.
type Algorithm string

const (
	// AlgorithmPHash is the DCT hash, and the zero value's meaning: the
	// picture's luma is averaged into a grid four times the hash's side (32×32
	// for 64 bits), and the bits are the lowest frequencies of the grid's
	// two-dimensional DCT-II (8×8), row by row, each set when its coefficient
	// is above the median. The DC term, the picture's mean brightness, says
	// nothing about its structure: it is left out of the median and its bit,
	// the first, is always 0. The median of the remaining odd number of
	// coefficients is one of them; a coefficient equal to it is 0, so only
	// those strictly above set a bit.
	AlgorithmPHash Algorithm = "phash"
	// AlgorithmDHash is the gradient hash: the picture's luma is averaged
	// into a grid one cell wider than the hash's side (9×8 for 64 bits), and
	// a bit is set where a cell is strictly brighter than its left
	// neighbour, row by row. Equal neighbours are 0.
	AlgorithmDHash Algorithm = "dhash"
)

// algorithms lists the accepted values, for error messages.
var algorithms = []Algorithm{AlgorithmPHash, AlgorithmDHash}

// defaultBits is the hash size when Config.Bits is not set.
const defaultBits = 64

// Config selects the hash. The zero value is the 64-bit pHash.
type Config struct {
	// Algorithm is how a picture becomes bits; "" means AlgorithmPHash.
	Algorithm Algorithm
	// Bits is the size of the hash and of its vectors: 64 or 256; 0 means
	// 64. Both are squares of a power of two, which makes 1/sqrt(Bits) exact
	// in a float32 and the cosine of two vectors exactly 1 - 2*hamming/Bits.
	Bits int
}

// Embedder implements langai.ImageEmbedder by hashing. It is immutable, so
// one serves any number of goroutines.
type Embedder struct {
	alg   Algorithm
	side  int     // the hash is side×side bits: 8 or 16
	model string  // what Model returns
	basis []int64 // pHash: the DCT cosines for side frequencies over 4*side samples
}

// verify the interfaces are actually satisfied
var (
	_ langai.ImageEmbedder = (*Embedder)(nil)
	_ langai.Identified    = (*Embedder)(nil)
)

// New returns the Embedder cfg describes, or an error naming what is wrong
// with cfg. There is nothing to load: it never blocks and needs no Close.
func New(cfg Config) (*Embedder, error) {
	e := &Embedder{alg: cfg.Algorithm}
	switch cfg.Algorithm {
	case "":
		e.alg = AlgorithmPHash
	case AlgorithmPHash, AlgorithmDHash:
	default:
		return nil, fmt.Errorf("phash: unknown algorithm %q: use one of %v", cfg.Algorithm, algorithms)
	}
	switch cfg.Bits {
	case 0, defaultBits:
		e.side = 8
	case 256:
		e.side = 16
	default:
		return nil, fmt.Errorf("phash: Bits %d is not supported: use 64 or 256", cfg.Bits)
	}
	e.model = fmt.Sprintf("%s-%d", e.alg, e.side*e.side)
	if e.alg == AlgorithmPHash {
		e.basis = dctBasis(e.side, 4*e.side)
	}
	return e, nil
}

// Model implements langai.Identified: the algorithm and the size, "phash-64",
// "dhash-64", "phash-256" or "dhash-256". Each is a vector space of its own,
// which a store keeps apart from the others and from any model's.
func (e *Embedder) Model() string { return e.model }

// Dimensions implements langai.ImageEmbedder: one entry per bit of the hash.
func (e *Embedder) Dimensions() int { return e.side * e.side }

// EmbedImages implements langai.ImageEmbedder: each picture's hash as a
// unit-length vector, +1/sqrt(bits) where a bit is set and -1/sqrt(bits)
// where it is not, so no vector is ever zero. A nil picture, one without
// pixels or one of more than 256 megapixels is an error naming its index.
// ctx is checked before each picture.
func (e *Embedder) EmbedImages(ctx context.Context, images []image.Image) ([][]float32, error) {
	out := make([][]float32, len(images))
	for i, img := range images {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := check(img); err != nil {
			return nil, fmt.Errorf("phash: image %d %w", i, err)
		}
		out[i] = e.vector(e.hash(img))
	}
	return out, nil
}

// Hash returns the picture's hash itself, the bits EmbedImages turns into a
// vector, for grouping duplicates without a vector store. A nil picture, one
// without pixels or one of more than 256 megapixels is an error.
func (e *Embedder) Hash(img image.Image) (Hash, error) {
	if err := check(img); err != nil {
		return Hash{}, fmt.Errorf("phash: image %w", err)
	}
	return e.hash(img), nil
}

// hash hashes a picture that passed check.
func (e *Embedder) hash(img image.Image) Hash {
	h := Hash{alg: e.alg, bits: e.side * e.side}
	if e.alg == AlgorithmDHash {
		cols := e.side + 1
		grey := grid(img, cols, e.side)
		for r := range e.side {
			for c := range e.side {
				if grey[r*cols+c+1] > grey[r*cols+c] {
					h.set(r*e.side + c)
				}
			}
		}
		return h
	}
	n := 4 * e.side
	coeffs := lowFrequencies(grid(img, n, n), n, e.side, e.basis)
	// coeffs[0] is the DC term: not in the median, and its bit stays 0
	mid := median(coeffs[1:])
	for i, c := range coeffs[1:] {
		if c > mid {
			h.set(i + 1)
		}
	}
	return h
}

// vector is h as a unit-length vector. 1/side is 1/sqrt(bits), a power of
// two, so the entries and every dot product of two such vectors are exact.
func (e *Embedder) vector(h Hash) []float32 {
	unit := 1 / float32(e.side)
	vec := make([]float32, h.bits)
	for i := range vec {
		if h.bit(i) {
			vec[i] = unit
		} else {
			vec[i] = -unit
		}
	}
	return vec
}

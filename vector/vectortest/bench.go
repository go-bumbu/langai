package vectortest

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/go-bumbu/langai/vector"
)

// BenchDims is the number of dimensions of every vector BenchSearch writes;
// the store its newStore returns must take vectors of this size.
const BenchDims = 256

// BenchSearch fills a collection with benchDocs documents of benchChunks
// chunks each.
const (
	benchDocs   = 2000
	benchChunks = 10
)

// BenchSearch measures the searches a "more like this" query makes, one
// Search with 1, 16 and 64 query vectors, over a collection of 20,000
// chunks of BenchDims dimensions. newStore returns an empty store for
// vectors of BenchDims dimensions.
func BenchSearch(b *testing.B, newStore func(b *testing.B) vector.Store) {
	ctx := context.Background()
	s := newStore(b)
	rng := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // seeded test data, the same on every run; nothing secret
	vec := func() []float32 {
		v := make([]float32, BenchDims)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	c := vector.Coord{Index: "bench", Partition: "p", Collection: "c"}
	batch := make([]vector.Doc, 0, 100)
	for d := range benchDocs {
		doc := vector.Doc{ID: fmt.Sprintf("doc%05d", d), Hash: "h"}
		for i := range benchChunks {
			doc.Chunks = append(doc.Chunks, vector.Chunk{Seq: i, Text: "chunk", Vector: vec()})
		}
		batch = append(batch, doc)
		if len(batch) == cap(batch) {
			if err := s.Upsert(ctx, c, "m", batch...); err != nil {
				b.Fatal(err)
			}
			batch = batch[:0]
		}
	}
	queries := make([][]float32, 64)
	for i := range queries {
		queries[i] = vec()
	}
	for _, n := range []int{1, 16, 64} {
		b.Run(fmt.Sprintf("vectors=%d", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := s.Search(ctx, c, vector.Query{Vectors: queries[:n], Limit: 10, Model: "m"}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

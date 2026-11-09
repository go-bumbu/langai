package phash

import (
	"image"
	"testing"
)

// BenchmarkHash hashes a 1920×1080 picture with each algorithm, as the RGBA a
// PNG decodes to and as the YCbCr a JPEG decodes to. Reading the pixels is
// nearly all of the cost, so the algorithms and sizes differ little.
func BenchmarkHash(b *testing.B) {
	rgba := picture(1, 1920, 1080)
	sources := []struct {
		name string
		img  image.Image
	}{
		{"rgba", rgba},
		{"ycbcr", reencoded(b, rgba, 90)},
	}
	for _, cfg := range configs {
		e := mustNew(b, cfg)
		for _, src := range sources {
			b.Run(e.Model()+"/"+src.name, func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := e.Hash(src.img); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

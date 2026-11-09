package search_test

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"path/filepath"

	"github.com/go-bumbu/sense/search"
	"github.com/go-bumbu/sense/sensetest"
	"github.com/go-bumbu/sense/vector"
)

// This example is the README's semantic search with fakes in place of the
// model and the store: index, search, "more like this", list, delete and
// reset.
func ExampleNew() {
	ctx := context.Background()
	emb := sensetest.FakeEmbedder{} // an app passes yzma.New's Embedder
	ix, err := search.New(emb, vector.NewMemStore(), "tenant-1", "space-7", "notes", search.WithMinScore(0.3))
	if err != nil {
		log.Fatal(err)
	}
	if err := ix.Index(ctx, "pasta", "Boil the spaghetti for nine minutes in salted water."); err != nil {
		log.Fatal(err)
	}
	res, err := ix.IndexBatch(ctx, []search.Doc{
		{ID: "bread", Text: "Knead the sourdough and let it rise overnight."},
		{ID: "noodles", Text: "Boil the noodles for four minutes."},
		{ID: "pasta", Text: "Boil the spaghetti for nine minutes in salted water."},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("indexed", res.Indexed, "unchanged", res.Unchanged)

	hits, err := ix.Search(ctx, "how long to boil spaghetti", 10)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("best match:", hits[0].DocID)
	more, err := ix.SearchIndexed(ctx, "pasta", 10)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("more like pasta:", more[0].DocID)

	if err := ix.Delete(ctx, "bread"); err != nil {
		log.Fatal(err)
	}
	for d, err := range ix.Docs(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("indexed:", d.ID)
	}

	// Reset empties the collection, e.g. to rebuild it with a new model
	if err := ix.Reset(ctx); err != nil {
		log.Fatal(err)
	}
	left := 0
	for range ix.Docs(ctx) {
		left++
	}
	fmt.Println("after Reset:", left)
	// Output:
	// indexed [bread noodles] unchanged [pasta]
	// best match: pasta
	// more like pasta: noodles
	// indexed: noodles
	// indexed: pasta
	// after Reset: 0
}

// picture is a picture of one colour.
func picture(c color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, 32, 24))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = c.R, c.G, c.B, c.A
	}
	return img
}

// solid encodes a picture of one colour as PNG.
func solid(c color.RGBA) *bytes.Reader {
	var buf bytes.Buffer
	if err := png.Encode(&buf, picture(c)); err != nil {
		log.Fatal(err)
	}
	return bytes.NewReader(buf.Bytes())
}

// This example indexes an image and a video, whose segments a video
// embedder makes, and finds the stretch of the video a photo shows. The
// fakes share one picture space, as yzma's Vision is all three embedders.
func ExampleNewMediaIndex() {
	ctx := context.Background()
	red, green := color.RGBA{200, 30, 30, 255}, color.RGBA{30, 200, 30, 255}
	video := &sensetest.FakeVideoEmbedder{Clips: map[string][]image.Image{
		// the clip's two 10 s segments show red, then green
		"clip.mp4": {picture(red), picture(green)},
	}}
	// the index hashes the file to skip it while it is unchanged; the fake
	// reads nothing but its name
	dir, err := os.MkdirTemp("", "media")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()
	clip := filepath.Join(dir, "clip.mp4")
	if err := os.WriteFile(clip, []byte("not really a video"), 0o600); err != nil {
		log.Fatal(err)
	}
	ix, err := search.NewMediaIndex(sensetest.FakeImageEmbedder{}, vector.NewMemStore(), "tenant-1", "space-7", "media",
		search.WithVideoEmbedder(video), search.WithMediaMinScore(0.9))
	if err != nil {
		log.Fatal(err)
	}
	if err := ix.IndexImage(ctx, "photo-42", solid(red)); err != nil {
		log.Fatal(err)
	}
	if err := ix.IndexVideo(ctx, "clip-7", clip); err != nil {
		log.Fatal(err)
	}
	hits, err := ix.SearchImage(ctx, solid(green), 10)
	if err != nil {
		log.Fatal(err)
	}
	for _, h := range hits {
		fmt.Printf("%s at %v\n", h.DocID, h.At)
	}
	// Output:
	// clip-7 at 10s
}

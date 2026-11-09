package langai_test

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/langaitest"
)

// This example defers a model's load to its first use. The identity and the
// dimensions come from the configuration, as a provider's spec gives them
// before anything loads (yzma's ModelSpec.ModelID and Dims), so a store can
// be set up first; the loaded model must then report exactly that identity.
func ExampleNewLazyEmbedder() {
	ctx := context.Background()
	emb := langai.NewLazyEmbedder(langaitest.FakeEmbedderModel, 64, func(context.Context) (langai.Embedder, error) {
		return langaitest.FakeEmbedder{}, nil // an app returns yzma.New(ctx, cfg)
	})
	state, _ := emb.State()
	fmt.Println(emb.Model(), emb.Dimensions(), state)

	vecs, err := emb.Embed(ctx, langai.RoleQuery, []string{"hello"})
	if err != nil {
		log.Fatal(err)
	}
	state, _ = emb.State()
	fmt.Println(len(vecs[0]), state)

	if err := emb.Close(); err != nil {
		log.Fatal(err)
	}
	_, err = emb.Embed(ctx, langai.RoleQuery, []string{"hello"})
	fmt.Println(errors.Is(err, langai.ErrClosed))
	// Output:
	// langaitest/fake-embedder 64 not started
	// 64 loaded
	// true
}

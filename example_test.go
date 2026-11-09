package sense_test

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/go-bumbu/sense"
	"github.com/go-bumbu/sense/sensetest"
)

// This example defers a model's load to its first use. The identity and the
// dimensions come from the configuration, as a provider's spec gives them
// before anything loads (yzma's ModelSpec.ModelID and Dims), so a store can
// be set up first; the loaded model must then report exactly that identity.
func ExampleNewLazyEmbedder() {
	ctx := context.Background()
	emb := sense.NewLazyEmbedder(sensetest.FakeEmbedderModel, 64, func(context.Context) (sense.Embedder, error) {
		return sensetest.FakeEmbedder{}, nil // an app returns yzma.New(ctx, cfg)
	})
	state, _ := emb.State()
	fmt.Println(emb.Model(), emb.Dimensions(), state)

	vecs, err := emb.Embed(ctx, sense.RoleQuery, []string{"hello"})
	if err != nil {
		log.Fatal(err)
	}
	state, _ = emb.State()
	fmt.Println(len(vecs[0]), state)

	if err := emb.Close(); err != nil {
		log.Fatal(err)
	}
	_, err = emb.Embed(ctx, sense.RoleQuery, []string{"hello"})
	fmt.Println(errors.Is(err, sense.ErrClosed))
	// Output:
	// sensetest/fake-embedder 64 not started
	// 64 loaded
	// true
}

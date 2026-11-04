package tagging_test

import (
	"context"
	"fmt"
	"log"

	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/tagging"
)

// cannedModel answers every prompt with the same reply.
type cannedModel string

func (m cannedModel) Complete(context.Context, string, string) (string, error) {
	return string(m), nil
}

// This example has a model write tags for a text, then keeps the ones that
// fit it.
func ExampleGenerator() {
	ctx := context.Background()

	// A canned model and FakeEmbedder keep the example self-contained; an app
	// would use the yzma Completer and Embedder. FakeEmbedder only counts
	// shared words, so its scores run lower than a real model's, and the cutoff
	// is lowered to match.
	model := cannedModel("sourdough bread\nbaking\ntravel\n")
	g, err := tagging.NewGenerator(model, langaitest.FakeEmbedder{}, tagging.GeneratorConfig{})
	if err != nil {
		log.Fatal(err)
	}

	// The model proposes three tags; "travel" has nothing to do with the text
	// and scores under the cutoff. What is left comes best score first.
	tags, err := g.Generate(ctx, "Baking sourdough bread at home", 0.3)
	if err != nil {
		log.Fatal(err)
	}
	for _, tag := range tags {
		fmt.Printf("%-16s %.2f\n", tag.Name, tag.Score)
	}

	// Output:
	// sourdough bread  0.63
	// baking           0.45
}

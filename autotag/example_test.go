package autotag_test

import (
	"context"
	"fmt"
	"log"

	"github.com/go-bumbu/langai/autotag"
	"github.com/go-bumbu/langai/semsearch"
)

// This example suggests tags for items by the names of the tags, then teaches
// a tag new words by tagging a few items with it by hand.
func Example() {
	ctx := context.Background()

	// FakeEmbedder and MemStore keep the example self-contained; an app would
	// use the yzma embedder and a Store that keeps its data in a database.
	// FakeEmbedder only counts shared words, so its scores run lower than a
	// real model's, and the two minimum scores are lowered to match. The
	// defaults fit the yzma model.
	tagger, err := autotag.New(semsearch.FakeEmbedder{}, autotag.NewMemStore(), autotag.Config{
		Model:          "fake",
		NameMinScore:   0.3,
		MemberMinScore: 0.5,
	})
	if err != nil {
		log.Fatal(err)
	}

	// The tags of one space of one tenant.
	space := tagger.Scope("tenant-1", "space-7")
	if err := space.SetTag(ctx, "bread", "cooking / bread"); err != nil {
		log.Fatal(err)
	}
	if err := space.SetTag(ctx, "japan", "travel / japan"); err != nil {
		log.Fatal(err)
	}

	// suggest stores an item with its text and prints the tags suggested for it.
	suggest := func(id, doc string) {
		if err := space.SetItem(ctx, id, doc); err != nil {
			log.Fatal(err)
		}
		ids, err := space.Suggest(ctx, id)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%-37s -> %v\n", doc, ids)
	}

	// No tag has examples yet, so each one matches by its name. An item gets
	// every tag that fits it, best fit first, or none.
	suggest("guide", "Baking bread at home: a cooking guide")
	suggest("diary", "A travel diary from Kyoto, Japan")
	suggest("trip", "Cooking bread in Japan")
	suggest("croissant", "Croissant dough with cold butter")

	// Someone tags three croissant recipes with "cooking / bread" by hand. The
	// tag learns their words, and now it fits the croissant item too.
	for _, recipe := range []struct{ id, doc string }{
		{"recipe-1", "Croissant dough: folding cold butter into layers"},
		{"recipe-2", "Butter croissant dough, step by step"},
		{"recipe-3", "Shaping croissant dough with butter"},
	} {
		if err := space.SetItem(ctx, recipe.id, recipe.doc); err != nil {
			log.Fatal(err)
		}
		if err := space.TagByHand(ctx, recipe.id, "bread"); err != nil {
			log.Fatal(err)
		}
	}
	suggest("croissant", "Croissant dough with cold butter")

	// Output:
	// Baking bread at home: a cooking guide -> [bread]
	// A travel diary from Kyoto, Japan      -> [japan]
	// Cooking bread in Japan                -> [bread japan]
	// Croissant dough with cold butter      -> []
	// Croissant dough with cold butter      -> [bread]
}

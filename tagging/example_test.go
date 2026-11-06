package tagging_test

import (
	"context"
	"fmt"
	"log"
	"maps"
	"slices"
	"strconv"

	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/tagging"
)

// notebook is an application's own tags and hand-made links, kept in memory
// for the example; an app reads them from its database. It has a single
// space, so it ignores the index and the partition. rev is the space's
// revision, which goes up with every change; each tag has a revision of its
// own, which goes up when its members change.
type notebook struct {
	names   map[string]string   // tag id -> name
	members map[string][]string // tag id -> texts of the items tagged with it by hand
	tagRev  map[string]int
	rev     int
}

func (n *notebook) Tags(context.Context, string, string) ([]tagging.CatalogTag, error) {
	var out []tagging.CatalogTag
	for _, id := range slices.Sorted(maps.Keys(n.names)) {
		out = append(out, tagging.CatalogTag{ID: id, Name: n.names[id], Revision: strconv.Itoa(n.tagRev[id])})
	}
	return out, nil
}

func (n *notebook) Members(_ context.Context, _, _ string, tagIDs []string) (map[string][]string, error) {
	out := map[string][]string{}
	for _, id := range tagIDs {
		out[id] = n.members[id]
	}
	return out, nil
}

// tagByHand records that a person tagged the item with text with the tag: the
// tag's members change, so its revision and the space's go up.
func (n *notebook) tagByHand(tagID, text string) {
	n.members[tagID] = append(n.members[tagID], text)
	n.tagRev[tagID]++
	n.rev++
}

// This example suggests tags for items by the names of the tags, then teaches
// a tag new words by tagging a few items with it by hand.
func ExampleTagger() {
	ctx := context.Background()

	// The application's tags of one space of one tenant.
	notes := &notebook{
		names:   map[string]string{"bread": "cooking / bread", "japan": "travel / japan"},
		members: map[string][]string{},
		tagRev:  map[string]int{"bread": 1, "japan": 1},
		rev:     1,
	}

	// FakeEmbedder and MemCache keep the example self-contained; an app would
	// use the yzma embedder and a cache that keeps the embeddings in a
	// database (sqlitecache or pgcache). FakeEmbedder only counts shared
	// words, so its scores run lower than a real model's, and the two minimum
	// scores are lowered to match. The defaults fit the yzma model.
	tagger, err := tagging.NewTagger(langaitest.FakeEmbedder{}, notes, tagging.NewMemCache(), tagging.TaggerConfig{
		Model:          "fake",
		NameMinScore:   0.3,
		MemberMinScore: 0.4,
	})
	if err != nil {
		log.Fatal(err)
	}
	space := tagger.Scope("tenant-1", "space-7")

	// suggest prints the tags suggested for an item's text at the space's
	// current revision.
	suggest := func(doc string) {
		ids, err := space.Suggest(ctx, strconv.Itoa(notes.rev), doc)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%-37s -> %v\n", doc, ids)
	}

	// No tag has examples yet, so each one matches by its name. An item gets
	// every tag that fits it, best fit first, or none.
	suggest("Baking bread at home: a cooking guide")
	suggest("A travel diary from Kyoto, Japan")
	suggest("Cooking bread in Japan")
	suggest("Croissant dough with cold butter")

	// Someone tags three croissant recipes with "cooking / bread" by hand. The
	// tag learns their words, and now it fits the croissant item too.
	for _, recipe := range []string{
		"Croissant dough: folding cold butter into layers",
		"Butter croissant dough, step by step",
		"Shaping croissant dough with butter",
	} {
		notes.tagByHand("bread", recipe)
	}
	suggest("Croissant dough with cold butter")

	// Output:
	// Baking bread at home: a cooking guide -> [bread]
	// A travel diary from Kyoto, Japan      -> [japan]
	// Cooking bread in Japan                -> [bread japan]
	// Croissant dough with cold butter      -> []
	// Croissant dough with cold butter      -> [bread]
}

package sqlitecache_test

import (
	"context"
	"database/sql"
	"log"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/tagging/sqlitecache"
)

// This example mirrors the README's automatic tagging: a Tagger over the
// application's Catalog with its embeddings cached in SQLite, and the
// scheduled Prune. It has no output, so it is compiled and not run: emb and
// catalog are the application's.
func ExampleNew() {
	ctx := context.Background()
	var (
		emb interface {
			langai.Embedder
			langai.Identified
		} // e.g. yzma.New's Embedder
		catalog tagging.Catalog // the application's tags and hand-made links
	)
	db, err := sql.Open("sqlite3", "autotag.db?_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		log.Fatal(err)
	}
	cache, err := sqlitecache.New(ctx, db)
	if err != nil {
		log.Fatal(err)
	}
	tagger, err := tagging.NewTagger(emb, catalog, cache, tagging.TaggerConfig{Model: emb.Model()})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = tagger.Close() }()
	space := tagger.Scope(tagging.Space{Index: "tenant-1", Partition: "space-7"})
	ids, err := space.Suggest(ctx, "rev-12", "Baking sourdough bread at home")
	if err != nil {
		log.Fatal(err)
	}
	log.Println(ids)

	spaces := []tagging.Space{{Index: "tenant-1", Partition: "space-7"}} // every space still in use
	removed, err := tagger.Prune(ctx, time.Now().AddDate(0, 0, -30), spaces...)
	log.Println(removed, err)
}

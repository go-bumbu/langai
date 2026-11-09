package mediasearch

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai/internal/tablename"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/vector/sqlitevec"
)

// legacyLibrary is a Library over a SQLite store whose index "examples" is
// in the tables of the schema before this version, as a database built by
// v0.1.1 is, told how to drop them.
func legacyLibrary(t *testing.T) (*Library, *fixture) {
	t.Helper()
	ctx := t.Context()
	f := newFixture(t)
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "index.sqlite")+"?_busy_timeout=5000&_journal_mode=WAL")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	// the old schema's marker: a vec table per index
	vec, err := tablename.For("", "examples", "vec")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `CREATE TABLE `+vec+` (x)`); err != nil {
		t.Fatal(err)
	}
	emb := &countingEmbedder{}
	store, err := sqlitevec.New(ctx, db, emb.Dimensions())
	if err != nil {
		t.Fatal(err)
	}
	coord := vector.Coord{Index: "examples", Partition: "local", Collection: "media"}
	ix, err := search.NewMediaIndex(emb, store, coord.Index, coord.Partition, coord.Collection, search.WithSampler(f.sampler))
	if err != nil {
		t.Fatal(err)
	}
	cache := t.TempDir()
	lib, err := New(Config{
		Dir: f.dir, Index: ix, Sampler: f.sampler, CatalogFile: filepath.Join(cache, "catalog.json"), CacheDir: filepath.Join(cache, "thumbs"),
		Info:      Info{Model: "fake", Dimensions: emb.Dimensions(), Provider: "cpu"},
		DropIndex: func(ctx context.Context) error { return store.DeleteIndex(ctx, coord.Index) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return lib, f
}

// TestScan_TheOldSchemaStopsTheScanUntilReset: an index in the old schema's
// tables stops the scan at its first file, as another model's does, instead
// of failing every file alike; the reset drops the tables, and the next scan
// indexes everything.
func TestScan_TheOldSchemaStopsTheScanUntilReset(t *testing.T) {
	ctx := context.Background()
	lib, _ := legacyLibrary(t)
	stats, err := lib.Scan(ctx)
	if !errors.Is(err, vector.ErrLegacySchema) {
		t.Fatalf("scan over the old schema: %v", err)
	}
	if !stats.Legacy || stats.Mismatch || stats.Error == "" || stats.Indexed != 0 || stats.Failed != 0 {
		t.Errorf("stats: %+v", stats)
	}
	if items := lib.Items(); len(items) != 4 {
		t.Errorf("catalog holds %d items, want all 4", len(items))
	}

	if err := lib.Reset(ctx); err != nil {
		t.Fatalf("reset of the old schema: %v", err)
	}
	stats, err = lib.Scan(ctx)
	if err != nil || stats.Legacy || stats.Indexed != 3 || stats.Failed != 1 {
		t.Errorf("scan after the reset: %+v, %v", stats, err)
	}
}

// TestReset_WithoutDropIndexLeavesTheOldSchema: with no way to drop the
// tables, the reset reports the store's refusal.
func TestReset_WithoutDropIndexLeavesTheOldSchema(t *testing.T) {
	lib, _ := legacyLibrary(t)
	lib.cfg.DropIndex = nil
	if err := lib.Reset(context.Background()); !errors.Is(err, vector.ErrLegacySchema) {
		t.Errorf("reset without DropIndex: %v, want vector.ErrLegacySchema", err)
	}
}

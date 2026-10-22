package sqlitevec

import (
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai/semsearch"
)

// newStore returns a Store backed by a sqlite file in a per-test temp dir.
func newStore(t *testing.T, dims int) *Store {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "semsearch.db"))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	store, err := New(context.Background(), db, dims)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return store
}

func TestUpsertAndSearch(t *testing.T) {
	store := newStore(t, 3)
	ctx := context.Background()

	err := store.Upsert(ctx, "idx", "p", "c", "a", "hash-a", []semsearch.Chunk{
		{Seq: 0, Text: "north", Vector: []float32{1, 0, 0}},
		{Seq: 1, Text: "east", Vector: []float32{0, 1, 0}},
	})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	err = store.Upsert(ctx, "idx", "p", "c", "b", "hash-b", []semsearch.Chunk{
		{Seq: 0, Text: "up", Vector: []float32{0, 0, 1}},
	})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	got, err := store.Search(ctx, "idx", "p", "c", []float32{1, 0.1, 0}, 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 matches, got %d: %#v", len(got), got)
	}
	if got[0].DocID != "a" || got[0].Text != "north" || got[0].Seq != 0 {
		t.Fatalf("unexpected best match: %#v", got[0])
	}
	if got[0].Score < 0.9 {
		t.Errorf("expected near-1 cosine score for close vectors, got %f", got[0].Score)
	}
	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Fatalf("results not ordered by descending score: %#v", got)
		}
	}
}

func TestSearchLimit(t *testing.T) {
	store := newStore(t, 3)
	ctx := context.Background()
	for i, v := range [][]float32{{1, 0, 0}, {0.9, 0.1, 0}, {0.8, 0.2, 0}} {
		err := store.Upsert(ctx, "idx", "p", "c", string(rune('a'+i)), "h", []semsearch.Chunk{{Seq: 0, Text: "t", Vector: v}})
		if err != nil {
			t.Fatalf("Upsert: %v", err)
		}
	}
	got, err := store.Search(ctx, "idx", "p", "c", []float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(got))
	}
}

func TestUpsertReplacesChunks(t *testing.T) {
	store := newStore(t, 3)
	ctx := context.Background()

	err := store.Upsert(ctx, "idx", "p", "c", "doc", "h1", []semsearch.Chunk{
		{Seq: 0, Text: "old-0", Vector: []float32{1, 0, 0}},
		{Seq: 1, Text: "old-1", Vector: []float32{0, 1, 0}},
		{Seq: 2, Text: "old-2", Vector: []float32{0, 0, 1}},
	})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	err = store.Upsert(ctx, "idx", "p", "c", "doc", "h2", []semsearch.Chunk{
		{Seq: 0, Text: "new-0", Vector: []float32{0, 1, 0}},
	})
	if err != nil {
		t.Fatalf("Upsert (replace): %v", err)
	}

	got, err := store.Search(ctx, "idx", "p", "c", []float32{1, 1, 1}, 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 || got[0].Text != "new-0" {
		t.Fatalf("expected only the replacement chunk, got %#v", got)
	}
}

func TestContentHash(t *testing.T) {
	store := newStore(t, 3)
	ctx := context.Background()

	hash, err := store.ContentHash(ctx, "idx", "p", "c", "missing")
	if err != nil {
		t.Fatalf("ContentHash: %v", err)
	}
	if hash != "" {
		t.Fatalf("expected empty hash for unknown doc, got %q", hash)
	}

	// zero chunks is valid: hash recorded, nothing searchable
	if err := store.Upsert(ctx, "idx", "p", "c", "doc", "the-hash", nil); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	hash, err = store.ContentHash(ctx, "idx", "p", "c", "doc")
	if err != nil {
		t.Fatalf("ContentHash: %v", err)
	}
	if hash != "the-hash" {
		t.Fatalf("expected recorded hash, got %q", hash)
	}
}

func TestDelete(t *testing.T) {
	store := newStore(t, 3)
	ctx := context.Background()

	err := store.Upsert(ctx, "idx", "p", "c", "doc", "h", []semsearch.Chunk{{Seq: 0, Text: "t", Vector: []float32{1, 0, 0}}})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := store.Delete(ctx, "idx", "p", "c", "doc"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	got, err := store.Search(ctx, "idx", "p", "c", []float32{1, 0, 0}, 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no matches after delete, got %#v", got)
	}
	hash, err := store.ContentHash(ctx, "idx", "p", "c", "doc")
	if err != nil {
		t.Fatalf("ContentHash: %v", err)
	}
	if hash != "" {
		t.Fatalf("expected hash removed on delete, got %q", hash)
	}
	// deleting an unknown doc is a no-op
	if err := store.Delete(ctx, "idx", "p", "c", "doc"); err != nil {
		t.Fatalf("Delete (again): %v", err)
	}
}

func TestDimensionMismatch(t *testing.T) {
	store := newStore(t, 3)
	ctx := context.Background()

	err := store.Upsert(ctx, "idx", "p", "c", "doc", "h", []semsearch.Chunk{{Seq: 0, Text: "t", Vector: []float32{1, 0}}})
	if err == nil {
		t.Fatal("expected error for wrong chunk dimensions")
	}
	if _, err := store.Search(ctx, "idx", "p", "c", []float32{1, 0}, 10); err == nil {
		t.Fatal("expected error for wrong query dimensions")
	}
}

// openTestDB returns a sqlite *sql.DB backed by a file in a per-test temp
// dir, for tests that construct the Store themselves via New.
func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "semsearch.db"))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func ids(ms []semsearch.Match) []string {
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.DocID
	}
	return out
}

func sameIDs(ms []semsearch.Match, want ...string) bool {
	if len(ms) != len(want) {
		return false
	}
	have := map[string]bool{}
	for _, m := range ms {
		have[m.DocID] = true
	}
	for _, w := range want {
		if !have[w] {
			return false
		}
	}
	return true
}

func TestSqliteVec_IndexIsolationAndScope(t *testing.T) {
	ctx := context.Background()
	db := openTestDB(t)
	s, err := New(ctx, db, 4)
	if err != nil {
		t.Fatal(err)
	}
	up := func(index, part, coll, id string, ax int) {
		v := make([]float32, 4)
		v[ax] = 1
		if err := s.Upsert(ctx, index, part, coll, id, "h", []semsearch.Chunk{{Seq: 0, Text: id, Vector: v}}); err != nil {
			t.Fatal(err)
		}
	}
	up("A", "10", "tags", "a", 0)
	up("B", "10", "tags", "b", 0) // same coords, different index
	up("A", "10", "bookmarks", "c", 0)
	up("A", "20", "tags", "d", 0)

	q := []float32{1, 0, 0, 0}
	// index isolation: A's exact search never sees B
	got, _ := s.Search(ctx, "A", "10", "tags", q, 10)
	if len(got) != 1 || got[0].DocID != "a" {
		t.Fatalf("exact/isolation wrong: %v", ids(got))
	}
	// partition-only within A -> a + c
	got, _ = s.SearchScope(ctx, "A", semsearch.ScopeFilter{Partition: "10"}, q, 10)
	if !sameIDs(got, "a", "c") {
		t.Fatalf("partition-only wrong: %v", ids(got))
	}
	// collection-only within A -> a + d
	got, _ = s.SearchScope(ctx, "A", semsearch.ScopeFilter{Collection: "tags"}, q, 10)
	if !sameIDs(got, "a", "d") {
		t.Fatalf("collection-only wrong: %v", ids(got))
	}
}

// TestSqliteVec_IndexCaseIsolation guards against a case-collision bug: SQL
// identifiers are case-insensitive, so if the index string were interpolated
// directly into the table name, indexes "A" and "a" would resolve to the
// same physical tables (ss_A_* / ss_a_* on a case-sensitive filesystem, but
// identical to SQLite's identifier matching) — a data leak across indexes,
// i.e. across tenants when the index is the tenant.
func TestSqliteVec_IndexCaseIsolation(t *testing.T) {
	ctx := context.Background()
	db := openTestDB(t)
	s, err := New(ctx, db, 3)
	if err != nil {
		t.Fatal(err)
	}
	// same partition/collection/vector, only the index case differs
	v := []float32{1, 0, 0}
	if err := s.Upsert(ctx, "A", "p", "c", "upper-doc", "h", []semsearch.Chunk{
		{Seq: 0, Text: "upper", Vector: v},
	}); err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(ctx, "a", "p", "c", "lower-doc", "h", []semsearch.Chunk{
		{Seq: 0, Text: "lower", Vector: v},
	}); err != nil {
		t.Fatal(err)
	}

	got, err := s.Search(ctx, "A", "p", "c", v, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !sameIDs(got, "upper-doc") {
		t.Fatalf(`Search("A", ...) not isolated from "a": %v`, ids(got))
	}

	got, err = s.Search(ctx, "a", "p", "c", v, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !sameIDs(got, "lower-doc") {
		t.Fatalf(`Search("a", ...) not isolated from "A": %v`, ids(got))
	}
}

// TestSqliteVec_GenerationsAreIsolated: one store per generation on the same
// database is what a migration runs while the next model's vectors are built
// next to the current ones. The same index, coordinates and document id stay
// apart in every operation, even when the two models' vectors differ in size.
func TestSqliteVec_GenerationsAreIsolated(t *testing.T) {
	ctx := context.Background()
	db := openTestDB(t)
	current, err := New(ctx, db, 3)
	if err != nil {
		t.Fatal(err)
	}
	next, err := New(ctx, db, 4, WithGeneration("g2"))
	if err != nil {
		t.Fatal(err)
	}
	if err := current.Upsert(ctx, "idx", "p", "c", "doc", "hash-current", []semsearch.Chunk{
		{Seq: 0, Text: "current", Vector: []float32{1, 0, 0}},
	}); err != nil {
		t.Fatalf("Upsert current: %v", err)
	}
	if err := next.Upsert(ctx, "idx", "p", "c", "doc", "hash-next", []semsearch.Chunk{
		{Seq: 0, Text: "next", Vector: []float32{1, 0, 0, 0}},
	}); err != nil {
		t.Fatalf("Upsert next: %v", err)
	}

	got, err := next.Search(ctx, "idx", "p", "c", []float32{1, 0, 0, 0}, 10)
	if err != nil {
		t.Fatalf("Search next: %v", err)
	}
	if len(got) != 1 || got[0].Text != "next" {
		t.Fatalf("next generation found %#v, want only its own chunk", got)
	}
	if h, err := next.ContentHash(ctx, "idx", "p", "c", "doc"); err != nil || h != "hash-next" {
		t.Fatalf("next ContentHash = %q, %v; want hash-next", h, err)
	}

	// retiring the document in one generation leaves the other's copy
	if err := next.Delete(ctx, "idx", "p", "c", "doc"); err != nil {
		t.Fatalf("Delete next: %v", err)
	}
	got, err = current.Search(ctx, "idx", "p", "c", []float32{1, 0, 0}, 10)
	if err != nil {
		t.Fatalf("Search current: %v", err)
	}
	if len(got) != 1 || got[0].Text != "current" {
		t.Fatalf("current generation found %#v after the next one deleted its copy, want its own chunk", got)
	}
	if h, err := current.ContentHash(ctx, "idx", "p", "c", "doc"); err != nil || h != "hash-current" {
		t.Fatalf("current ContentHash = %q, %v; want hash-current", h, err)
	}
}

// TestNew_RejectsMalformedGeneration: a generation that cannot name tables
// is refused when the store is built, not on its first write.
func TestNew_RejectsMalformedGeneration(t *testing.T) {
	if _, err := New(context.Background(), openTestDB(t), 3, WithGeneration("G2")); err == nil {
		t.Fatal(`New accepted generation "G2"`)
	}
}

// TestSqliteVec_GenerationObjectsCarryItsPrefix: every table and index a
// generation's store creates, sqlite-vec's shadow tables included, is named
// ss_<generation>_*, so a retired generation can be dropped by name without
// listing indexes. SQLite's own sqlite_* objects are not the store's.
func TestSqliteVec_GenerationObjectsCarryItsPrefix(t *testing.T) {
	ctx := context.Background()
	db := openTestDB(t)
	s, err := New(ctx, db, 3, WithGeneration("g2"))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(ctx, "idx", "p", "c", "doc", "h", []semsearch.Chunk{
		{Seq: 0, Text: "t", Vector: []float32{1, 0, 0}},
	}); err != nil {
		t.Fatal(err)
	}
	rows, err := db.QueryContext(ctx, `SELECT name FROM sqlite_master WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite\_%' ESCAPE '\'`)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rows.Close() }()
	var names []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		names = append(names, name)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(names) == 0 {
		t.Fatal("the store created no tables")
	}
	for _, name := range names {
		if !strings.HasPrefix(name, "ss_g2_") {
			t.Errorf("%s is not named after generation g2", name)
		}
	}
}

func TestEndToEndWithIndex(t *testing.T) {
	store := newStore(t, 64)
	ix, err := semsearch.New(semsearch.FakeEmbedder{}, store, "idx", "p", "c")
	if err != nil {
		t.Fatalf("semsearch.New: %v", err)
	}
	ctx := context.Background()

	if err := ix.Index(ctx, "golang", "# Go\n\ngoroutines channels compiler"); err != nil {
		t.Fatalf("Index: %v", err)
	}
	if err := ix.Index(ctx, "cooking", "# Cooking\n\npasta pizza tomato sauce"); err != nil {
		t.Fatalf("Index: %v", err)
	}

	got, err := ix.Search(ctx, "goroutines channels", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) == 0 || got[0].DocID != "golang" {
		t.Fatalf("expected golang first, got %#v", got)
	}
}

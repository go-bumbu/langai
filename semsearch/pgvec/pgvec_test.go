package pgvec

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/go-bumbu/testdbs"

	"github.com/go-bumbu/langai/internal/tablename"
	"github.com/go-bumbu/langai/semsearch"
)

// TestMain registers a pgvector container as a testdbs slow DB. A plain
// go test selects none and every test here skips; TESTDBS_ALL=1 (needs
// Docker) starts the container once for the whole package:
//
//	TESTDBS_ALL=1 go test ./semsearch/pgvec/
func TestMain(m *testing.M) {
	testdbs.InitCustomDbs(nil, []testdbs.TargetDb{testdbs.NewPostgres("pgvector/pgvector:pg18")})
	code := m.Run()
	if err := testdbs.Clean(); err != nil {
		fmt.Fprintf(os.Stderr, "testdbs cleanup: %v\n", err)
		os.Exit(1)
	}
	os.Exit(code)
}

// newStore returns a Store on a database of its own in the pgvector
// container, so tests never see each other's tables, and skips the test when
// TESTDBS_ALL is unset. Postgres scopes extensions per database; New creates
// the vector extension, so each fresh database gets it.
func newStore(t *testing.T, dims int) *Store {
	t.Helper()
	dbs := testdbs.DBs()
	if len(dbs) == 0 {
		t.Skip("set TESTDBS_ALL=1 (needs Docker) to run the Postgres tests")
	}
	db, err := dbs[0].ConnDbName(t.Name()).DB()
	if err != nil {
		t.Fatalf("db handle: %v", err)
	}
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

func TestUpsertReplacesChunks(t *testing.T) {
	store := newStore(t, 3)
	ctx := context.Background()

	err := store.Upsert(ctx, "idx", "p", "c", "doc", "h1", []semsearch.Chunk{
		{Seq: 0, Text: "old-0", Vector: []float32{1, 0, 0}},
		{Seq: 1, Text: "old-1", Vector: []float32{0, 1, 0}},
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

func TestContentHashAndDelete(t *testing.T) {
	store := newStore(t, 3)
	ctx := context.Background()

	hash, err := store.ContentHash(ctx, "idx", "p", "c", "missing")
	if err != nil {
		t.Fatalf("ContentHash: %v", err)
	}
	if hash != "" {
		t.Fatalf("expected empty hash for unknown doc, got %q", hash)
	}

	err = store.Upsert(ctx, "idx", "p", "c", "doc", "the-hash", []semsearch.Chunk{
		{Seq: 0, Text: "t", Vector: []float32{1, 0, 0}},
	})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	hash, err = store.ContentHash(ctx, "idx", "p", "c", "doc")
	if err != nil {
		t.Fatalf("ContentHash: %v", err)
	}
	if hash != "the-hash" {
		t.Fatalf("expected recorded hash, got %q", hash)
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
	hash, err = store.ContentHash(ctx, "idx", "p", "c", "doc")
	if err != nil {
		t.Fatalf("ContentHash: %v", err)
	}
	if hash != "" {
		t.Fatalf("expected hash removed on delete, got %q", hash)
	}
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

func TestPgVec_IndexIsolationAndScope(t *testing.T) {
	ctx := context.Background()
	s := newStore(t, 4)
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
	got, err := s.Search(ctx, "A", "10", "tags", q, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].DocID != "a" {
		t.Fatalf("exact/isolation wrong: %v", ids(got))
	}
	// partition-only within A -> a + c
	got, err = s.SearchScope(ctx, "A", semsearch.ScopeFilter{Partition: "10"}, q, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !sameIDs(got, "a", "c") {
		t.Fatalf("partition-only wrong: %v", ids(got))
	}
	// collection-only within A -> a + d
	got, err = s.SearchScope(ctx, "A", semsearch.ScopeFilter{Collection: "tags"}, q, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !sameIDs(got, "a", "d") {
		t.Fatalf("collection-only wrong: %v", ids(got))
	}
}

// axisQuery is the query seedNearAndFar's partitions are placed around.
var axisQuery = []float32{1, 0, 0, 0, 0, 0, 0, 0}

// seedNearAndFar stores n 8-dimensional chunks close to axisQuery under
// partition "near" and n pointing elsewhere under partition "far", all in one
// index, then analyzes the table as autovacuum would, so the planner chooses
// as it does on a live database. Every nearest neighbour of axisQuery is then
// one a search scoped to "far" must skip.
func seedNearAndFar(t *testing.T, s *Store, index string, n int) {
	t.Helper()
	ctx := context.Background()
	rng := rand.New(rand.NewSource(1))
	for axis, part := range []string{"near", "far"} {
		chunks := make([]semsearch.Chunk, n)
		for i := range chunks {
			v := make([]float32, 8)
			for j := range v {
				v[j] = (rng.Float32()*2 - 1) / 10
			}
			v[axis] = 1
			chunks[i] = semsearch.Chunk{Seq: i, Text: part, Vector: v}
		}
		if err := s.Upsert(ctx, index, part, "c", part+"-doc", "h", chunks); err != nil {
			t.Fatalf("Upsert %s: %v", part, err)
		}
	}
	chunksT, err := tablename.For("", index, "chunks")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.db.ExecContext(ctx, "ANALYZE "+chunksT); err != nil {
		t.Fatalf("analyze: %v", err)
	}
}

// TestSearchScopeFillsLimitBehindCloserPartitions: a scoped search returns the
// best rows of its scope however many closer rows other partitions of the
// index hold. An HNSW index scan filters only after collecting its ef_search
// (40) candidates, all of them "near" here, and so answers with nothing.
func TestSearchScopeFillsLimitBehindCloserPartitions(t *testing.T) {
	s := newStore(t, 8)
	seedNearAndFar(t, s, "idx", 500)
	got, err := s.Search(context.Background(), "idx", "far", "c", axisQuery, 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 10 {
		t.Fatalf("expected 10 matches from partition far, got %d", len(got))
	}
	for _, m := range got {
		if m.Partition != "far" {
			t.Fatalf("match outside the scope: %#v", m)
		}
	}
}

// TestSearchIgnoresLegacyHNSWIndex: databases created before searches went
// exact carry an HNSW index on the chunk vectors. A Store opened on one after
// the upgrade must not leave it for the planner to pick, or scoped searches
// keep losing rows.
func TestSearchIgnoresLegacyHNSWIndex(t *testing.T) {
	ctx := context.Background()
	s := newStore(t, 8)
	seedNearAndFar(t, s, "idx", 500)
	chunksT, err := tablename.For("", "idx", "chunks")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.db.ExecContext(ctx, fmt.Sprintf(
		`CREATE INDEX IF NOT EXISTS %s_embedding_idx ON %s USING hnsw (embedding vector_cosine_ops)`, chunksT, chunksT)); err != nil {
		t.Fatalf("create legacy index: %v", err)
	}
	upgraded, err := New(ctx, s.db, 8)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err := upgraded.Search(ctx, "idx", "far", "c", axisQuery, 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 10 {
		t.Fatalf("expected 10 matches from partition far, got %d", len(got))
	}
}

// TestPgVec_GenerationsAreIsolated: one store per generation on the same
// database is what a migration runs while the next model's vectors are built
// next to the current ones. The same index, coordinates and document id stay
// apart in every operation, even when the two models' vectors differ in size.
func TestPgVec_GenerationsAreIsolated(t *testing.T) {
	ctx := context.Background()
	current := newStore(t, 3)
	next, err := New(ctx, current.db, 4, WithGeneration("g2"))
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
	db := newStore(t, 3).db
	if _, err := New(context.Background(), db, 3, WithGeneration("G2")); err == nil {
		t.Fatal(`New accepted generation "G2"`)
	}
}

// TestPgVec_GenerationObjectsCarryItsPrefix: every table, index and sequence
// a generation's store creates, the ones Postgres names after its tables
// included, is named ss_<generation>_*, so a retired generation can be
// dropped by name without listing indexes. Each test has a database of its
// own, so everything in its schema is the store's.
func TestPgVec_GenerationObjectsCarryItsPrefix(t *testing.T) {
	ctx := context.Background()
	s, err := New(ctx, newStore(t, 3).db, 3, WithGeneration("g2"))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(ctx, "idx", "p", "c", "doc", "h", []semsearch.Chunk{
		{Seq: 0, Text: "t", Vector: []float32{1, 0, 0}},
	}); err != nil {
		t.Fatal(err)
	}
	rows, err := s.db.QueryContext(ctx, `SELECT c.relname FROM pg_class c
		JOIN pg_namespace n ON n.oid = c.relnamespace
		WHERE n.nspname = current_schema() AND c.relkind IN ('r', 'i', 'S')`)
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

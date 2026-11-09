package pgvec

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/go-bumbu/testdbs"

	"github.com/go-bumbu/langai/internal/tablename"
	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/vector/vectortest"
)

// TestMain registers a pgvector container as a testdbs slow DB. A plain
// go test selects none and every test here skips; TESTDBS_ALL=1 (needs
// Docker) starts the container once for the whole package:
//
//	TESTDBS_ALL=1 go test ./vector/pgvec/
func TestMain(m *testing.M) {
	testdbs.InitCustomDbs(nil, []testdbs.TargetDb{testdbs.NewPostgres("pgvector/pgvector:pg18")})
	code := m.Run()
	if err := testdbs.Clean(); err != nil {
		fmt.Fprintf(os.Stderr, "testdbs cleanup: %v\n", err)
		os.Exit(1)
	}
	os.Exit(code)
}

// newDB returns a handle on a database of its own in the pgvector container,
// so tests never see each other's tables, and skips the test when
// TESTDBS_ALL is unset. Postgres scopes extensions per database; New creates
// the vector extension, so each fresh database gets it.
func newDB(t *testing.T) *sql.DB {
	t.Helper()
	dbs := testdbs.DBs()
	if len(dbs) == 0 {
		t.Skip("set TESTDBS_ALL=1 (needs Docker) to run the Postgres tests")
	}
	db, err := dbs[0].ConnDbName(t.Name()).DB()
	if err != nil {
		t.Fatalf("db handle: %v", err)
	}
	return db
}

// newStore returns a Store on a database of its own.
func newStore(t *testing.T, dims int, opts ...Option) (*Store, *sql.DB) {
	t.Helper()
	db := newDB(t)
	s, err := New(t.Context(), db, dims, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s, db
}

func TestConformance(t *testing.T) {
	vectortest.Run(t, func(t *testing.T) vector.Store {
		s, _ := newStore(t, vectortest.Dims)
		return s
	})
}

// tableNames lists the tables, indexes and sequences of the test database.
func tableNames(t *testing.T, db *sql.DB) []string {
	t.Helper()
	rows, err := db.QueryContext(t.Context(), `SELECT c.relname FROM pg_class c
		JOIN pg_namespace n ON n.oid = c.relnamespace
		WHERE n.nspname = current_schema() AND c.relkind IN ('r', 'i', 'S') ORDER BY c.relname`)
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
	return names
}

// axisQuery is the query seedNearAndFar's partitions are placed around.
var axisQuery = []float32{1, 0, 0, 0, 0, 0, 0, 0}

// seedNearAndFar stores n 8-dimensional documents close to axisQuery under
// partition "near" and n pointing elsewhere under partition "far", all in one
// index, then analyzes the table as autovacuum would, so the planner chooses
// as it does on a live database. Every nearest neighbour of axisQuery is then
// one a search scoped to "far" must skip.
func seedNearAndFar(t *testing.T, s *Store, db *sql.DB, index string, n int) {
	t.Helper()
	ctx := t.Context()
	rng := rand.New(rand.NewSource(1))
	for axis, part := range []string{"near", "far"} {
		docs := make([]vector.Doc, n)
		for i := range docs {
			v := make([]float32, 8)
			for j := range v {
				v[j] = (rng.Float32()*2 - 1) / 10
			}
			v[axis] = 1
			docs[i] = vector.Doc{ID: fmt.Sprintf("%s-%d", part, i), Chunks: []vector.Chunk{{Text: part, Vector: v}}}
		}
		if err := s.Upsert(ctx, vector.Coord{Index: index, Partition: part, Collection: "c"}, "", docs...); err != nil {
			t.Fatalf("Upsert %s: %v", part, err)
		}
	}
	chunksT, err := tablename.For("", index, "chunks")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, "ANALYZE "+chunksT); err != nil {
		t.Fatalf("analyze: %v", err)
	}
}

// TestSearchFillsLimitBehindCloserPartitions: a search returns the best
// documents of its partition however many closer ones other partitions of
// the index hold. An HNSW index scan filters only after collecting its
// ef_search (40) candidates, all of them "near" here, and so would answer
// with nothing.
func TestSearchFillsLimitBehindCloserPartitions(t *testing.T) {
	s, db := newStore(t, 8)
	seedNearAndFar(t, s, db, "idx", 500)
	got, err := s.Search(t.Context(), vector.Coord{Index: "idx", Partition: "far", Collection: "c"}, vector.Query{Vector: axisQuery, Limit: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 10 {
		t.Fatalf("expected 10 matches from partition far, got %d", len(got))
	}
	for _, m := range got {
		if m.Coord.Partition != "far" {
			t.Fatalf("match outside the scope: %#v", m)
		}
	}
}

// TestReadOnlyRole: a role that may only read the tables searches an index
// another role created, and reads of an index that does not exist neither
// create tables nor fail for want of the right to.
func TestReadOnlyRole(t *testing.T) {
	ctx := t.Context()
	s, db := newStore(t, 3)
	c := vector.Coord{Index: "idx", Partition: "p", Collection: "c"}
	if err := s.Upsert(ctx, c, "m", vector.Doc{ID: "a", Hash: "h", Chunks: []vector.Chunk{{Vector: []float32{1, 0, 0}}}}); err != nil {
		t.Fatal(err)
	}
	before := tableNames(t, db)
	// every connection opened to this test's database from now on runs as a
	// role that may only read
	role := fmt.Sprintf("reader_%d", rand.Int63())
	for _, st := range []string{
		`CREATE ROLE ` + role,
		`GRANT SELECT ON ALL TABLES IN SCHEMA public TO ` + role,
		`DO $$ BEGIN EXECUTE format('ALTER ROLE %I IN DATABASE %I SET role = ` + role + `', current_user, current_database()); END $$`,
	} {
		if _, err := db.ExecContext(ctx, st); err != nil {
			t.Fatalf("%s: %v", st, err)
		}
	}
	db.SetMaxIdleConns(0) // close the connections opened before
	var current string
	if err := db.QueryRowContext(ctx, `SELECT current_role`).Scan(&current); err != nil || current != role {
		t.Fatalf("connection runs as %q, %v; want %q", current, err, role)
	}
	reader, err := New(ctx, db, 3)
	if err != nil {
		t.Fatalf("New as a reader: %v", err)
	}
	got, err := reader.Search(ctx, c, vector.Query{Vector: []float32{1, 0, 0}, Limit: 10, Model: "m"})
	if err != nil || len(got) != 1 {
		t.Fatalf("Search as a reader = %v, %v", got, err)
	}
	if err := reader.Upsert(ctx, c, "m", vector.Doc{ID: "b"}); err == nil {
		t.Fatal("the reader wrote: the role is not read-only")
	}
	unknown := vector.Coord{Index: "never", Partition: "p", Collection: "c"}
	if got, err := reader.Search(ctx, unknown, vector.Query{Vector: []float32{1, 0, 0}, Limit: 10}); err != nil || len(got) != 0 {
		t.Fatalf("Search of an unknown index as a reader = %v, %v", got, err)
	}
	if h, err := reader.Hashes(ctx, unknown, "a"); err != nil || len(h) != 0 {
		t.Fatalf("Hashes of an unknown index as a reader = %v, %v", h, err)
	}
	if after := tableNames(t, db); strings.Join(after, ",") != strings.Join(before, ",") {
		t.Fatalf("reads changed the tables from %v to %v", before, after)
	}
}

// TestReadsCreateNoTables: searching, listing and deleting in an index that
// was never written leave the database as it was.
func TestReadsCreateNoTables(t *testing.T) {
	ctx := t.Context()
	s, db := newStore(t, 3)
	c := vector.Coord{Index: "never", Partition: "p", Collection: "c"}
	if _, err := s.Search(ctx, c, vector.Query{Vector: []float32{1, 0, 0}, Limit: 10, AllCollections: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Docs(ctx, c, "", 0); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Chunks(ctx, c, "a"); err != nil {
		t.Fatal(err)
	}
	if err := s.Delete(ctx, c, "a"); err != nil {
		t.Fatal(err)
	}
	if names := tableNames(t, db); len(names) != 0 {
		t.Fatalf("reads created %v", names)
	}
}

// TestNew_OtherDimensions: a store opened for other dimensions than an
// index's tables hold refuses that index and names the way out.
func TestNew_OtherDimensions(t *testing.T) {
	ctx := t.Context()
	eight, db := newStore(t, 8)
	c := vector.Coord{Index: "idx", Partition: "p", Collection: "c"}
	if err := eight.Upsert(ctx, c, "", vector.Doc{ID: "a", Chunks: []vector.Chunk{{Vector: axisQuery}}}); err != nil {
		t.Fatal(err)
	}
	sixteen, err := New(ctx, db, 16)
	if err != nil {
		t.Fatal(err)
	}
	q := make([]float32, 16)
	q[0] = 1
	_, err = sixteen.Search(ctx, c, vector.Query{Vector: q, Limit: 10})
	if !errors.Is(err, vector.ErrDimensions) || !strings.Contains(err.Error(), "WithGeneration") {
		t.Fatalf("Search with a 16-d store over 8-d tables: %v, want ErrDimensions naming WithGeneration", err)
	}
	if err := sixteen.Upsert(ctx, c, "", vector.Doc{ID: "b", Chunks: []vector.Chunk{{Vector: q}}}); !errors.Is(err, vector.ErrDimensions) {
		t.Fatalf("Upsert with a 16-d store over 8-d tables: %v, want ErrDimensions", err)
	}
}

// TestLegacySchemaIsRefused: the tables of the schema before the meta table
// are refused with an error that says what to do, and DeleteIndex clears
// them.
func TestLegacySchemaIsRefused(t *testing.T) {
	ctx := t.Context()
	s, db := newStore(t, 3)
	docs, err := tablename.For("", "idx", "docs")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `CREATE TABLE `+docs+` (x TEXT)`); err != nil {
		t.Fatal(err)
	}
	c := vector.Coord{Index: "idx", Partition: "p", Collection: "c"}
	if _, err := s.Hashes(ctx, c, "a"); err == nil || !strings.Contains(err.Error(), "schema before") {
		t.Fatalf("Hashes over legacy tables: %v, want the schema error", err)
	}
	if err := s.DeleteIndex(ctx, "idx"); err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(ctx, c, "", vector.Doc{ID: "a", Chunks: []vector.Chunk{{Vector: []float32{1, 0, 0}}}}); err != nil {
		t.Fatalf("Upsert after DeleteIndex dropped the legacy tables: %v", err)
	}
}

// TestGenerationsAreIsolated: one store per generation on the same database
// is what a migration runs while the next model's vectors are built next to
// the current ones. The same index, coordinates and document id stay apart,
// even when the two models' vectors differ in size.
func TestGenerationsAreIsolated(t *testing.T) {
	ctx := t.Context()
	current, db := newStore(t, 3)
	next, err := New(ctx, db, 4, WithGeneration("g2"))
	if err != nil {
		t.Fatal(err)
	}
	c := vector.Coord{Index: "idx", Partition: "p", Collection: "c"}
	if err := current.Upsert(ctx, c, "m1", vector.Doc{ID: "doc", Hash: "hash-current", Chunks: []vector.Chunk{{Text: "current", Vector: []float32{1, 0, 0}}}}); err != nil {
		t.Fatalf("Upsert current: %v", err)
	}
	if err := next.Upsert(ctx, c, "m2", vector.Doc{ID: "doc", Hash: "hash-next", Chunks: []vector.Chunk{{Text: "next", Vector: []float32{1, 0, 0, 0}}}}); err != nil {
		t.Fatalf("Upsert next: %v", err)
	}
	got, err := next.Search(ctx, c, vector.Query{Vector: []float32{1, 0, 0, 0}, Limit: 10})
	if err != nil || len(got) != 1 || got[0].Text != "next" {
		t.Fatalf("next generation found %+v, %v; want only its own chunk", got, err)
	}
	if err := next.Delete(ctx, c, "doc"); err != nil {
		t.Fatalf("Delete next: %v", err)
	}
	if h, err := current.Hashes(ctx, c, "doc"); err != nil || h["doc"] != "hash-current" {
		t.Fatalf("current hashes = %v, %v; want hash-current", h, err)
	}
}

// TestNew_RejectsMalformedGeneration: a generation that cannot name tables
// is refused when the store is built, not on its first write.
func TestNew_RejectsMalformedGeneration(t *testing.T) {
	if _, err := New(t.Context(), newDB(t), 3, WithGeneration("G2")); err == nil {
		t.Fatal(`New accepted generation "G2"`)
	}
}

// TestGenerationObjectsCarryItsPrefix: every table, index and sequence a
// generation's store creates, the ones Postgres names after its tables
// included, is named ss_<generation>_*, so DropGeneration can find them by
// name. Each test has a database of its own, so everything in its schema is
// the store's.
func TestGenerationObjectsCarryItsPrefix(t *testing.T) {
	ctx := t.Context()
	s, db := newStore(t, 3, WithGeneration("g2"))
	if err := s.Upsert(ctx, vector.Coord{Index: "idx"}, "m", vector.Doc{ID: "doc", Chunks: []vector.Chunk{{Vector: []float32{1, 0, 0}}}}); err != nil {
		t.Fatal(err)
	}
	names := tableNames(t, db)
	if len(names) == 0 {
		t.Fatal("the store created no tables")
	}
	for _, name := range names {
		if !strings.HasPrefix(name, "ss_g2_") {
			t.Errorf("%s is not named after generation g2", name)
		}
	}
}

// TestDropGeneration: after a migration, the new generation's store drops
// the old generation's tables, every index's, sequences included, and keeps
// its own; it refuses to drop its own generation.
func TestDropGeneration(t *testing.T) {
	ctx := t.Context()
	old, db := newStore(t, 3)
	next, err := New(ctx, db, 3, WithGeneration("g2"))
	if err != nil {
		t.Fatal(err)
	}
	doc := vector.Doc{ID: "doc", Chunks: []vector.Chunk{{Vector: []float32{1, 0, 0}}}}
	for _, s := range []*Store{old, next} {
		for _, index := range []string{"a", "b"} {
			if err := s.Upsert(ctx, vector.Coord{Index: index}, "", doc); err != nil {
				t.Fatal(err)
			}
		}
	}
	if _, err := db.ExecContext(ctx, `CREATE TABLE ss_mine (x TEXT)`); err != nil {
		t.Fatal(err)
	}
	if err := next.DropGeneration(ctx, "g2"); err == nil {
		t.Fatal("DropGeneration dropped the store's own generation")
	}
	if err := next.DropGeneration(ctx, ""); err != nil {
		t.Fatalf("DropGeneration: %v", err)
	}
	for _, name := range tableNames(t, db) {
		if !strings.HasPrefix(name, "ss_g2_") && name != "ss_mine" {
			t.Errorf("%s survived DropGeneration", name)
		}
	}
	got, err := next.Search(ctx, vector.Coord{Index: "a"}, vector.Query{Vector: []float32{1, 0, 0}, Limit: 10})
	if err != nil || len(got) != 1 {
		t.Fatalf("the new generation lost its vectors: %v, %v", got, err)
	}
}

func TestEndToEndWithIndex(t *testing.T) {
	s, _ := newStore(t, 64)
	ix, err := search.New(langaitest.FakeEmbedder{}, s, "idx", "p", "c")
	if err != nil {
		t.Fatalf("search.New: %v", err)
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

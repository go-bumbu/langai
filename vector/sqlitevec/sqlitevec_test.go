package sqlitevec

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai/internal/tablename"
	"github.com/go-bumbu/langai/langaitest"
	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/vector/vectortest"
)

// recommendedDSN is the DSN the package doc recommends, after the file name.
const recommendedDSN = "?_busy_timeout=5000&_journal_mode=WAL"

// openTestDB returns a sqlite *sql.DB backed by a file in a per-test temp
// dir, so every connection the pool opens sees the same database, opened
// with dsn after the file name.
func openTestDB(t *testing.T, dsn string) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "search.db")+dsn)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

// newStore returns a Store on a database of its own, opened the way the
// package doc recommends.
func newStore(t *testing.T, dims int, opts ...Option) *Store {
	t.Helper()
	s, err := New(t.Context(), openTestDB(t, recommendedDSN), dims, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func TestConformance(t *testing.T) {
	vectortest.Run(t, func(t *testing.T) vector.Store { return newStore(t, vectortest.Dims) })
}

// TestConformanceShared runs the shared cases with every Store on a handle
// of its own to one database file, as separate processes would be.
func TestConformanceShared(t *testing.T) {
	vectortest.RunShared(t, func(t *testing.T, n int) []vector.Store {
		path := filepath.Join(t.TempDir(), "search.db")
		stores := make([]vector.Store, n)
		for i := range stores {
			stores[i] = storeAt(t, path)
		}
		return stores
	})
}

// storeAt returns a Store of vectortest.Dims dimensions on a handle of its
// own to the database file at path, opened the way the package doc
// recommends.
func storeAt(t *testing.T, path string) *Store {
	t.Helper()
	db, err := sql.Open("sqlite3", path+recommendedDSN)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	s, err := New(t.Context(), db, vectortest.Dims)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// TestOpenDoesNotStallOtherIndexes: while the first write of a new index
// waits for SQLite's write lock, which another process holds, a read of an
// index already in use goes ahead at once, and a read of the new index gives
// up when its context does, instead of all of them queueing behind the
// creation for up to the busy timeout.
func TestOpenDoesNotStallOtherIndexes(t *testing.T) {
	ctx := t.Context()
	path := filepath.Join(t.TempDir(), "search.db")
	s := storeAt(t, path)
	ready := vector.Coord{Index: "ready", Partition: "p", Collection: "c"}
	v := []float32{1, 0, 0, 0}
	if err := s.Upsert(ctx, ready, "", vector.Doc{ID: "x", Chunks: []vector.Chunk{{Vector: v}}}); err != nil {
		t.Fatal(err)
	}
	other, err := sql.Open("sqlite3", path+recommendedDSN)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = other.Close() })
	conn, err := other.Conn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	if _, err := conn.ExecContext(ctx, "BEGIN IMMEDIATE"); err != nil {
		t.Fatal(err)
	}
	fresh := vector.Coord{Index: "fresh", Partition: "p", Collection: "c"}
	created := make(chan error, 1)
	go func() {
		created <- s.Upsert(ctx, fresh, "", vector.Doc{ID: "y", Chunks: []vector.Chunk{{Vector: v}}})
	}()
	time.Sleep(200 * time.Millisecond) // the creation is waiting for the lock by now
	for _, tc := range []struct {
		name    string
		c       vector.Coord
		wantErr error
	}{
		{"ready index", ready, nil},
		{"index being created", fresh, context.DeadlineExceeded},
	} {
		short, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
		start := time.Now()
		_, err := s.Hashes(short, tc.c, "x")
		took := time.Since(start)
		cancel()
		if !errors.Is(err, tc.wantErr) {
			t.Errorf("%s: Hashes = %v, want %v", tc.name, err, tc.wantErr)
		}
		if took > time.Second {
			t.Errorf("%s: Hashes took %v behind another index's creation", tc.name, took)
		}
	}
	if _, err := conn.ExecContext(ctx, "COMMIT"); err != nil {
		t.Fatal(err)
	}
	if err := <-created; err != nil {
		t.Fatalf("the creation failed: %v", err)
	}
}

// tableNames lists the tables and indexes of db that are not SQLite's own.
func tableNames(t *testing.T, db *sql.DB) []string {
	t.Helper()
	rows, err := db.QueryContext(t.Context(), `SELECT name FROM sqlite_master WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite\_%' ESCAPE '\' ORDER BY name`)
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

// TestConcurrentUpserts_BusyTimeoutIsEnough: every write transaction starts
// with a write, so a busy timeout alone lets writers queue, in WAL mode and
// in the default rollback-journal mode; before, a transaction that read
// first failed with "database is locked" in most calls.
func TestConcurrentUpserts_BusyTimeoutIsEnough(t *testing.T) {
	for _, dsn := range []string{recommendedDSN, "?_busy_timeout=5000"} {
		t.Run(dsn, func(t *testing.T) {
			s, err := New(t.Context(), openTestDB(t, dsn), 3)
			if err != nil {
				t.Fatal(err)
			}
			c := vector.Coord{Index: "i", Partition: "p", Collection: "c"}
			var wg sync.WaitGroup
			errs := make(chan error, 80)
			for w := range 8 {
				wg.Go(func() {
					for i := range 10 {
						errs <- s.Upsert(t.Context(), c, "m", vector.Doc{ID: fmt.Sprintf("%d-%d", w, i), Hash: "h",
							Chunks: []vector.Chunk{{Vector: []float32{1, float32(i), 0}}}})
					}
				})
			}
			wg.Wait()
			close(errs)
			failed := 0
			for err := range errs {
				if err != nil {
					failed++
					t.Log(err)
				}
			}
			if failed > 0 {
				t.Fatalf("%d of 80 concurrent upserts failed", failed)
			}
		})
	}
}

// BenchmarkSearchVectors: a search with 1, 16 and 64 query vectors, the
// shape of a "more like this" search, over 20,000 chunks of 256 dimensions.
func BenchmarkSearchVectors(b *testing.B) {
	vectortest.BenchSearch(b, func(b *testing.B) vector.Store {
		db, err := sql.Open("sqlite3", filepath.Join(b.TempDir(), "search.db")+recommendedDSN)
		if err != nil {
			b.Fatal(err)
		}
		b.Cleanup(func() { _ = db.Close() })
		s, err := New(b.Context(), db, vectortest.BenchDims)
		if err != nil {
			b.Fatal(err)
		}
		return s
	})
}

// TestDriverErrorsAreWrapped: an error of the driver or of database/sql
// comes back prefixed with "vector: sqlite" and the operation, and still
// matches what it wraps.
func TestDriverErrorsAreWrapped(t *testing.T) {
	ctx := t.Context()
	db := openTestDB(t, recommendedDSN)
	s, err := New(ctx, db, 3)
	if err != nil {
		t.Fatal(err)
	}
	c := vector.Coord{Index: "idx", Partition: "p", Collection: "c"}
	if err := s.Upsert(ctx, c, "", vector.Doc{ID: "a", Chunks: []vector.Chunk{{Vector: []float32{1, 0, 0}}}}); err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	for _, tc := range []struct {
		op   string
		call func() error
	}{
		{"upsert", func() error {
			return s.Upsert(cancelled, c, "", vector.Doc{ID: "b", Chunks: []vector.Chunk{{Vector: []float32{1, 0, 0}}}})
		}},
		{"delete", func() error { return s.Delete(cancelled, c, "a") }},
		{"delete index", func() error { return s.DeleteIndex(cancelled, "idx") }},
		{"hashes", func() error { _, err := s.Hashes(cancelled, c, "a"); return err }},
		{"docs", func() error { _, err := s.Docs(cancelled, c, "", 0); return err }},
		{"chunks", func() error { _, err := s.Chunks(cancelled, c, "a"); return err }},
		{"search", func() error {
			_, err := s.Search(cancelled, c, vector.Query{Vectors: [][]float32{[]float32{1, 0, 0}}, Limit: 1})
			return err
		}},
	} {
		err := tc.call()
		if !errors.Is(err, context.Canceled) || !strings.HasPrefix(err.Error(), "vector: sqlite "+tc.op+": ") {
			t.Errorf("%s with a cancelled context: %v, want context.Canceled prefixed with vector: sqlite %s", tc.op, err, tc.op)
		}
	}
}

// TestReadsCreateNoTables: searching, listing and deleting in an index that
// was never written leave the database as it was.
func TestReadsCreateNoTables(t *testing.T) {
	ctx := t.Context()
	db := openTestDB(t, recommendedDSN)
	s, err := New(ctx, db, 3)
	if err != nil {
		t.Fatal(err)
	}
	c := vector.Coord{Index: "never", Partition: "p", Collection: "c"}
	if _, err := s.Search(ctx, c, vector.Query{Vectors: [][]float32{[]float32{1, 0, 0}}, Limit: 10, AllCollections: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Hashes(ctx, c, "a"); err != nil {
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
	if err := s.DeleteCollection(ctx, c); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Model(ctx, c); err != nil {
		t.Fatal(err)
	}
	if names := tableNames(t, db); len(names) != 0 {
		t.Fatalf("reads created %v", names)
	}
}

// TestNew_OtherDimensions: a store opened for other dimensions than an
// index's tables hold refuses that index, on reads and writes, and names the
// way out.
func TestNew_OtherDimensions(t *testing.T) {
	ctx := t.Context()
	db := openTestDB(t, recommendedDSN)
	eight, err := New(ctx, db, 8)
	if err != nil {
		t.Fatal(err)
	}
	c := vector.Coord{Index: "idx", Partition: "p", Collection: "c"}
	if err := eight.Upsert(ctx, c, "", vector.Doc{ID: "a", Chunks: []vector.Chunk{{Vector: []float32{1, 0, 0, 0, 0, 0, 0, 0}}}}); err != nil {
		t.Fatal(err)
	}
	sixteen, err := New(ctx, db, 16)
	if err != nil {
		t.Fatal(err)
	}
	q := make([]float32, 16)
	q[0] = 1
	_, err = sixteen.Search(ctx, c, vector.Query{Vectors: [][]float32{q}, Limit: 10})
	if !errors.Is(err, vector.ErrDimensions) || !strings.Contains(err.Error(), "WithGeneration") {
		t.Fatalf("Search with a 16-d store over 8-d tables: %v, want ErrDimensions naming WithGeneration", err)
	}
	err = sixteen.Upsert(ctx, c, "", vector.Doc{ID: "b", Chunks: []vector.Chunk{{Vector: q}}})
	if !errors.Is(err, vector.ErrDimensions) {
		t.Fatalf("Upsert with a 16-d store over 8-d tables: %v, want ErrDimensions", err)
	}
}

// TestLegacySchemaIsRefused: the tables of the schema before the meta table
// (a vec0 table per index) are refused with an error that says what to do,
// instead of a driver error on the first statement that touches them.
func TestLegacySchemaIsRefused(t *testing.T) {
	ctx := t.Context()
	db := openTestDB(t, recommendedDSN)
	vec, err := tablename.For("", "idx", "vec")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `CREATE TABLE `+vec+` (x)`); err != nil {
		t.Fatal(err)
	}
	s, err := New(ctx, db, 3)
	if err != nil {
		t.Fatal(err)
	}
	c := vector.Coord{Index: "idx", Partition: "p", Collection: "c"}
	_, err = s.Search(ctx, c, vector.Query{Vectors: [][]float32{[]float32{1, 0, 0}}, Limit: 10})
	if err == nil || !strings.Contains(err.Error(), "schema before") {
		t.Fatalf("Search over legacy tables: %v, want the schema error", err)
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
// the current ones. The same index, coordinates and document id stay apart
// in every operation, even when the two models' vectors differ in size.
func TestGenerationsAreIsolated(t *testing.T) {
	ctx := t.Context()
	db := openTestDB(t, recommendedDSN)
	current, err := New(ctx, db, 3)
	if err != nil {
		t.Fatal(err)
	}
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
	got, err := next.Search(ctx, c, vector.Query{Vectors: [][]float32{[]float32{1, 0, 0, 0}}, Limit: 10})
	if err != nil || len(got) != 1 || got[0].Text != "next" {
		t.Fatalf("next generation found %+v, %v; want only its own chunk", got, err)
	}
	// retiring the document in one generation leaves the other's copy
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
	if _, err := New(t.Context(), openTestDB(t, ""), 3, WithGeneration("G2")); err == nil {
		t.Fatal(`New accepted generation "G2"`)
	}
}

// TestGenerationObjectsCarryItsPrefix: every table and index a generation's
// store creates is named ss_<generation>_*, so DropGeneration can find them
// by name. SQLite's own sqlite_* objects are not the store's.
func TestGenerationObjectsCarryItsPrefix(t *testing.T) {
	ctx := t.Context()
	db := openTestDB(t, recommendedDSN)
	s, err := New(ctx, db, 3, WithGeneration("g2"))
	if err != nil {
		t.Fatal(err)
	}
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
// the old generation's tables, every index's, and keeps its own; it refuses
// to drop its own generation.
func TestDropGeneration(t *testing.T) {
	ctx := t.Context()
	db := openTestDB(t, recommendedDSN)
	old, err := New(ctx, db, 3)
	if err != nil {
		t.Fatal(err)
	}
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
	// a table of the application's that only looks like one of the store's
	if _, err := db.ExecContext(ctx, `CREATE TABLE ss_mine (x)`); err != nil {
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
	got, err := next.Search(ctx, vector.Coord{Index: "a"}, vector.Query{Vectors: [][]float32{[]float32{1, 0, 0}}, Limit: 10})
	if err != nil || len(got) != 1 {
		t.Fatalf("the new generation lost its vectors: %v, %v", got, err)
	}
}

// TestBackendsAgree: on random vectors the exact KNN of sqlite-vec ranks as
// MemStore's brute force does.
func TestBackendsAgree(t *testing.T) {
	ctx := t.Context()
	const dims = 8
	mem := vector.NewMemStore()
	sv := newStore(t, dims)
	vecs := langaitest.FakeEmbedder{Dims: dims}
	texts := make([]string, 50)
	for i := range texts {
		texts[i] = fmt.Sprintf("word%d word%d word%d", i, i*7%13, i*3%11)
	}
	embedded, err := vecs.Embed(ctx, 0, texts)
	if err != nil {
		t.Fatal(err)
	}
	for _, index := range []string{"A", "B"} {
		c := vector.Coord{Index: index, Partition: "10", Collection: "tags"}
		for i, v := range embedded {
			d := vector.Doc{ID: fmt.Sprintf("%s%02d", index, i), Chunks: []vector.Chunk{{Vector: v}}}
			if err := mem.Upsert(ctx, c, "", d); err != nil {
				t.Fatal(err)
			}
			if err := sv.Upsert(ctx, c, "", d); err != nil {
				t.Fatal(err)
			}
		}
	}
	c := vector.Coord{Index: "A", Partition: "10", Collection: "tags"}
	q := vector.Query{Vectors: [][]float32{embedded[3]}, Limit: 10}
	memHits, err := mem.Search(ctx, c, q)
	if err != nil {
		t.Fatal(err)
	}
	svHits, err := sv.Search(ctx, c, q)
	if err != nil {
		t.Fatal(err)
	}
	ids := func(ms []vector.Match) []string {
		var out []string
		for _, m := range ms {
			out = append(out, m.DocID)
		}
		return out
	}
	if len(memHits) != 10 || !slices.Equal(ids(memHits), ids(svHits)) {
		t.Fatalf("rankings differ:\nmem %v\nsv  %v", ids(memHits), ids(svHits))
	}
}

func TestEndToEndWithIndex(t *testing.T) {
	ix, err := search.New(langaitest.FakeEmbedder{}, newStore(t, 64), "idx", "p", "c")
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

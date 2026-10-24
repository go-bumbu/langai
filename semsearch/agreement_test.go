package semsearch_test

import (
	"context"
	"database/sql"
	"math/rand"
	"path/filepath"
	"strconv"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai/semsearch"
	"github.com/go-bumbu/langai/semsearch/sqlitevec"
)

// openSqliteForTest returns a CGO sqlite *sql.DB backed by a file in a
// per-test temp dir (rather than ":memory:") so every connection the pool
// opens sees the same database; blank-importing mattn/go-sqlite3 registers
// the "sqlite3" driver, and sqlitevec's own import of the sqlite-vec cgo
// bindings registers the auto-extension on every future connection.
func openSqliteForTest(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "agreement.db"))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestBackendsAgree(t *testing.T) {
	ctx := context.Background()
	const dims = 8
	mem := semsearch.NewMemStore()
	db := openSqliteForTest(t) // helper: CGO sqlite *sql.DB
	sv, err := sqlitevec.New(ctx, db, dims)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	vec := func() []float32 {
		v := make([]float32, dims)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	// seed the same data into both, two indexes
	for _, idx := range []string{"A", "B"} {
		for i := 0; i < 25; i++ {
			id := idx + strconv.Itoa(i)
			v := vec()
			ch := []semsearch.Chunk{{Seq: 0, Text: id, Vector: v}}
			if err := mem.Upsert(ctx, idx, "10", "tags", id, "h", ch); err != nil {
				t.Fatal(err)
			}
			if err := sv.Upsert(ctx, idx, "10", "tags", id, "h", ch); err != nil {
				t.Fatal(err)
			}
		}
	}
	q := vec()
	memHits, err := mem.Search(ctx, "A", "10", "tags", q, 5)
	if err != nil {
		t.Fatal(err)
	}
	svHits, err := sv.Search(ctx, "A", "10", "tags", q, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(memHits) != 5 {
		t.Fatalf("expected 5 seeded hits, got %d", len(memHits))
	}
	if len(memHits) != len(svHits) {
		t.Fatalf("hit count differs: mem=%d sv=%d", len(memHits), len(svHits))
	}
	for i := range memHits {
		if memHits[i].DocID != svHits[i].DocID {
			t.Fatalf("rank %d differs: mem=%s sv=%s", i, memHits[i].DocID, svHits[i].DocID)
		}
		if memHits[i].DocID[0] != 'A' { // isolation: never a 'B' doc
			t.Fatalf("index leak: %s", memHits[i].DocID)
		}
	}
}

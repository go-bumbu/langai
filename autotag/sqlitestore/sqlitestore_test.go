package sqlitestore

import (
	"database/sql"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai/autotag"
	"github.com/go-bumbu/langai/autotag/storetest"
)

// openTestDB opens a SQLite file in a per-test temp dir, the way the package
// doc recommends for several goroutines writing: the suite's concurrent Links
// would fail with "database is locked" otherwise.
func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "autotag.db")+"?_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestConformance(t *testing.T) {
	storetest.Run(t, func(t *testing.T) autotag.Store {
		s, err := New(t.Context(), openTestDB(t))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return s
	})
}

func TestNew_RejectsANilHandle(t *testing.T) {
	if _, err := New(t.Context(), nil); err == nil {
		t.Fatal("New accepted a nil database handle")
	}
}

// TestNew_FailsWhenSQLiteAnswersNothing: New asks the database for its version
// to know the handle works, so a closed one is refused when the store is built.
func TestNew_FailsWhenSQLiteAnswersNothing(t *testing.T) {
	db := openTestDB(t)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := New(t.Context(), db); err == nil {
		t.Fatal("New accepted a closed database handle")
	}
}

// TestWithGeneration_RejectsABadToken: a generation that cannot name tables is
// refused when the store is built, not on its first write.
func TestWithGeneration_RejectsABadToken(t *testing.T) {
	if _, err := New(t.Context(), openTestDB(t), WithGeneration("G-2")); err == nil {
		t.Fatal(`New accepted generation "G-2"`)
	}
}

// TestWithGeneration_KeepsTablesApart: stores of two generations on one
// database do not see each other's tags.
func TestWithGeneration_KeepsTablesApart(t *testing.T) {
	ctx := t.Context()
	db := openTestDB(t)
	current, err := New(ctx, db)
	if err != nil {
		t.Fatal(err)
	}
	next, err := New(ctx, db, WithGeneration("g2"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := next.PutTag(ctx, "idx", "p", "t1", "a"); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := current.Tag(ctx, "idx", "p", "t1"); ok || err != nil {
		t.Errorf("the default generation found a tag of g2: %v, %v", ok, err)
	}
	if _, ok, err := next.Tag(ctx, "idx", "p", "t1"); !ok || err != nil {
		t.Errorf("g2 lost its own tag: %v, %v", ok, err)
	}
}

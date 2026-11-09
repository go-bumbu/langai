package sqlitecache

import (
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/sense/tagging"
	"github.com/go-bumbu/sense/tagging/cachetest"
)

// openTestDB opens a SQLite file in a per-test temp dir, the way the package
// doc recommends for several goroutines writing: the suite's concurrent Puts
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
	cachetest.Run(t, func(t *testing.T, now func() time.Time) tagging.EmbeddingCache {
		c, err := open(t.Context(), openTestDB(t), now)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return c
	})
}

func TestNew_RejectsANilHandle(t *testing.T) {
	if _, err := New(t.Context(), nil); err == nil {
		t.Fatal("New accepted a nil database handle")
	}
}

// TestNew_FailsWhenSQLiteAnswersNothing: New asks the database for its version
// to know the handle works, so a closed one is refused when the cache is built.
func TestNew_FailsWhenSQLiteAnswersNothing(t *testing.T) {
	db := openTestDB(t)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := New(t.Context(), db); err == nil {
		t.Fatal("New accepted a closed database handle")
	}
}

package sqlstore_test

import (
	"database/sql"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai/vector"
	"github.com/go-bumbu/langai/vector/sqlitevec"
	"github.com/go-bumbu/langai/vector/vectortest"
)

// TestConformance_SQLite runs the suite through sqlitevec, the Dialect that
// needs no container, so that `make coverage` measures this package; pgvec
// runs it through the Postgres Dialect from its own tests.
func TestConformance_SQLite(t *testing.T) {
	vectortest.Run(t, func(t *testing.T) vector.Store {
		db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "search.db")+"?_busy_timeout=5000&_journal_mode=WAL")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = db.Close() })
		s, err := sqlitevec.New(t.Context(), db, vectortest.Dims)
		if err != nil {
			t.Fatal(err)
		}
		return s
	})
}

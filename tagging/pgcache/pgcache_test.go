package pgcache

import (
	"database/sql"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/go-bumbu/testdbs"

	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/tagging/cachetest"
)

// TestMain registers a Postgres container as a testdbs slow DB. A plain go test
// selects none and every test that needs the database skips; TESTDBS_ALL=1
// (needs Docker) starts the container once for the whole package:
//
//	TESTDBS_ALL=1 go test ./tagging/pgcache/
//
// The cache needs no extension. The image is the one the pgvec tests run, so a
// machine that has run those already holds it.
func TestMain(m *testing.M) {
	testdbs.InitCustomDbs(nil, []testdbs.TargetDb{testdbs.NewPostgres("pgvector/pgvector:pg18")})
	code := m.Run()
	if err := testdbs.Clean(); err != nil {
		fmt.Fprintf(os.Stderr, "testdbs cleanup: %v\n", err)
		os.Exit(1)
	}
	os.Exit(code)
}

// newDB returns a handle on a database of its own in the container, so tests
// never see each other's tables, and skips the test when TESTDBS_ALL is unset.
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

func TestConformance(t *testing.T) {
	cachetest.Run(t, func(t *testing.T, now func() time.Time) tagging.EmbeddingCache {
		c, err := open(t.Context(), newDB(t), now)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return c
	})
}

// TestNew_ConcurrentStartsAllSucceed: processes that start together each
// create the table if it is missing, and Postgres lets two CREATE TABLE IF NOT
// EXISTS collide on its catalog ("duplicate key value violates unique
// constraint pg_type_typname_nsp_index"). Every New must still succeed.
func TestNew_ConcurrentStartsAllSucceed(t *testing.T) {
	db := newDB(t)
	const rounds, starters = 5, 16
	for round := range rounds {
		if _, err := db.ExecContext(t.Context(), `DROP TABLE IF EXISTS autotag_embeddings`); err != nil {
			t.Fatal(err)
		}
		errs := make(chan error, starters)
		var wg sync.WaitGroup
		for range starters {
			wg.Go(func() {
				_, err := New(t.Context(), db)
				errs <- err
			})
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Fatalf("round %d: New: %v", round, err)
			}
		}
	}
}

func TestNew_RejectsANilHandle(t *testing.T) {
	if _, err := New(t.Context(), nil); err == nil {
		t.Fatal("New accepted a nil database handle")
	}
}

// TestNew_FailsWhenPostgresAnswersNothing: New asks the database for its
// version to know the handle works, so a closed one is refused when the cache
// is built.
func TestNew_FailsWhenPostgresAnswersNothing(t *testing.T) {
	db := newDB(t)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := New(t.Context(), db); err == nil {
		t.Fatal("New accepted a closed database handle")
	}
}

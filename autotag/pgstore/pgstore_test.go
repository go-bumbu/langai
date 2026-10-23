package pgstore

import (
	"database/sql"
	"fmt"
	"os"
	"testing"

	"github.com/go-bumbu/testdbs"

	"github.com/go-bumbu/langai/autotag"
	"github.com/go-bumbu/langai/autotag/storetest"
)

// TestMain registers a Postgres container as a testdbs slow DB. A plain go test
// selects none and every test that needs the database skips; TESTDBS_ALL=1
// (needs Docker) starts the container once for the whole package:
//
//	TESTDBS_ALL=1 go test ./autotag/pgstore/
//
// The store needs no extension. The image is the one the pgvec tests run, so a
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

func newStore(t *testing.T) autotag.Store {
	t.Helper()
	s, err := New(t.Context(), newDB(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func TestConformance(t *testing.T) { storetest.Run(t, newStore) }

func TestNew_RejectsANilHandle(t *testing.T) {
	if _, err := New(t.Context(), nil); err == nil {
		t.Fatal("New accepted a nil database handle")
	}
}

// TestNew_FailsWhenPostgresAnswersNothing: New asks the database for its
// version to know the handle works, so a closed one is refused when the store
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

// TestWithGeneration_RejectsABadToken: a generation that cannot name tables is
// refused when the store is built, not on its first write.
func TestWithGeneration_RejectsABadToken(t *testing.T) {
	if _, err := New(t.Context(), newDB(t), WithGeneration("G-2")); err == nil {
		t.Fatal(`New accepted generation "G-2"`)
	}
}

// TestWithGeneration_KeepsTablesApart: stores of two generations on one
// database do not see each other's tags.
func TestWithGeneration_KeepsTablesApart(t *testing.T) {
	ctx := t.Context()
	db := newDB(t)
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

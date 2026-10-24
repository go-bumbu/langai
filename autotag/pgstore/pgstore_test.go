package pgstore

import (
	"database/sql"
	"fmt"
	"os"
	"testing"

	"github.com/go-bumbu/testdbs"

	"github.com/go-bumbu/langai/autotag"
	"github.com/go-bumbu/langai/autotag/storetest"
	"github.com/go-bumbu/langai/internal/tablename"
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

// TestVersionColumn_HoldsTheLargestStartExactly: a new tag starts at a random
// Version of up to 2^62, so the BIGINT version column must keep it exactly,
// through a read, a bump and the Version check of SetTagVectors. 2^62 and
// 2^62+1 are one and the same float64, so a column or a driver that went
// through floating point would lose the bump.
func TestVersionColumn_HoldsTheLargestStartExactly(t *testing.T) {
	ctx := t.Context()
	db := newDB(t)
	s, err := New(ctx, db)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.PutTag(ctx, "idx", "p", "t1", "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PutItem(ctx, "idx", "p", autotag.ItemRecord{ID: "i1", Text: "x", Vec: []float32{1}, Model: "m"}); err != nil {
		t.Fatal(err)
	}
	tags, err := tablename.ForPrefix("autotag", "", "idx", "tags") // the store's own name for the table
	if err != nil {
		t.Fatal(err)
	}
	const top = int64(1) << 62
	if _, err := db.ExecContext(ctx, `UPDATE `+tags+` SET version = $1 WHERE partition = 'p' AND tag_id = 't1'`, top); err != nil {
		t.Fatal(err)
	}
	version := func() int64 {
		t.Helper()
		rec, ok, err := s.Tag(ctx, "idx", "p", "t1")
		if err != nil || !ok {
			t.Fatalf("Tag(t1) = %+v, %v, %v", rec, ok, err)
		}
		return rec.Version
	}
	if got := version(); got != top {
		t.Fatalf("Version of t1 = %d, want %d", got, top)
	}
	if _, err := s.Link(ctx, "idx", "p", "i1", "t1"); err != nil {
		t.Fatal(err)
	}
	if got := version(); got != top+1 {
		t.Fatalf("Version of t1 after a Link = %d, want %d", got, top+1)
	}
	for _, c := range []struct {
		version int64
		want    bool
	}{{top, false}, {top + 1, true}} {
		stored, err := s.SetTagVectors(ctx, "idx", "p", "t1", c.version, autotag.TagVectors{NameVec: []float32{1}, Vec: []float32{1}, Model: "m"})
		if err != nil || stored != c.want {
			t.Errorf("SetTagVectors at Version %d of a tag at %d = %v, %v; want %v, nil", c.version, top+1, stored, err, c.want)
		}
	}
}

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

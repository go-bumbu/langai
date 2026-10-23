package sqlstore

import (
	"bytes"
	"database/sql"
	"path/filepath"
	"slices"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"github.com/go-bumbu/langai/autotag"
	"github.com/go-bumbu/langai/autotag/storetest"
	"github.com/go-bumbu/langai/internal/tablename"
)

// openSQLite opens a SQLite file in a per-test temp dir. SQLite allows one
// writer at a time: the busy timeout and the immediate transactions are what
// let the suite's concurrent Links wait for each other instead of failing with
// "database is locked" (see the package doc of sqlitestore).
func openSQLite(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "autotag.db")+"?_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

// tableOf returns the name of the table a store of the generation keeps for
// the index, as tablename derives it.
func tableOf(t *testing.T, generation, index, suffix string) string {
	t.Helper()
	name, err := tablename.ForPrefix(prefix, generation, index, suffix)
	if err != nil {
		t.Fatal(err)
	}
	return name
}

func TestSQLiteConformance(t *testing.T) {
	storetest.Run(t, func(t *testing.T) autotag.Store {
		s, err := New(openSQLite(t), SQLite, "")
		if err != nil {
			t.Fatal(err)
		}
		return s
	})
}

// TestCodec_RoundTripsAndRejectsTornBlobs: a vector is stored as the
// little-endian bytes of its float32s, with nil as NULL, so a blob whose size
// is not a multiple of four was cut short, and reading it is an error rather
// than a wrong vector.
func TestCodec_RoundTripsAndRejectsTornBlobs(t *testing.T) {
	v1 := []float32{1.5, -2.25, 3.4028235e38, 1.1754944e-38, 0}
	got, err := decode(encode(v1))
	if err != nil || !slices.Equal(got, v1) {
		t.Errorf("decode(encode(%v)) = %v, %v; want it back", v1, got, err)
	}

	// the byte layout is the stored format, so it must not change
	if got, want := encode([]float32{1}), []byte{0x00, 0x00, 0x80, 0x3f}; !bytes.Equal(got, want) {
		t.Errorf("encode(1) = %#v, want the little-endian float32 %#v", got, want)
	}

	if got := encode(nil); got != nil {
		t.Errorf("encode(nil) = %#v, want nil, which is stored as NULL", got)
	}
	for _, blob := range [][]byte{nil, {}} {
		if got, err := decode(blob); got != nil || err != nil {
			t.Errorf("decode(%#v) = %v, %v; want nil, nil", blob, got, err)
		}
	}
	if got, err := decode([]byte{1, 2, 3}); err == nil {
		t.Errorf("decode of a blob cut short = %v, nil; want an error", got)
	}
}

func TestRebind(t *testing.T) {
	cases := []struct{ query, want string }{
		{"a = ? AND b = ?", "a = $1 AND b = $2"},
		{"SELECT 1", "SELECT 1"},
		{"(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", "($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"},
	}
	for _, c := range cases {
		if got := rebind(c.query); got != c.want {
			t.Errorf("rebind(%q) = %q, want %q", c.query, got, c.want)
		}
	}
}

func TestNew_RejectsNilDBAndBadGeneration(t *testing.T) {
	if _, err := New(nil, SQLite, ""); err == nil {
		t.Error("New accepted a nil database handle")
	}
	if _, err := New(openSQLite(t), SQLite, "G2"); err == nil {
		t.Error(`New accepted generation "G2"`)
	}
}

// TestGenerationsAreIsolated: one store per generation on the same database is
// what a migration to another embedding model runs while the next model's
// vectors are built next to the current ones. The same index, partition and
// ids stay apart in every table.
func TestGenerationsAreIsolated(t *testing.T) {
	ctx := t.Context()
	db := openSQLite(t)
	current, err := New(db, SQLite, "")
	if err != nil {
		t.Fatal(err)
	}
	next, err := New(db, SQLite, "g2")
	if err != nil {
		t.Fatal(err)
	}

	if _, err := current.PutTag(ctx, "idx", "p", "t1", "current tag"); err != nil {
		t.Fatalf("PutTag in the current generation: %v", err)
	}
	if _, ok, err := next.Tag(ctx, "idx", "p", "t1"); ok || err != nil {
		t.Errorf("the next generation found the current one's tag: %v, %v", ok, err)
	}

	// the same ids in the next generation are other rows, in every table
	if _, err := next.PutTag(ctx, "idx", "p", "t1", "next tag"); err != nil {
		t.Fatalf("PutTag in the next generation: %v", err)
	}
	for _, s := range []autotag.Store{current, next} {
		if _, err := s.PutItem(ctx, "idx", "p", autotag.ItemRecord{ID: "i1", Text: "x", Vec: []float32{1}, Model: "m"}); err != nil {
			t.Fatalf("PutItem: %v", err)
		}
	}
	if _, err := next.Link(ctx, "idx", "p", "i1", "t1"); err != nil {
		t.Fatalf("Link in the next generation: %v", err)
	}

	for _, c := range []struct {
		store       autotag.Store
		name        string
		wantMembers int
	}{{current, "current tag", 0}, {next, "next tag", 1}} {
		rec, ok, err := c.store.Tag(ctx, "idx", "p", "t1")
		if err != nil || !ok || rec.Name != c.name {
			t.Errorf("Tag = %+v, %v, %v; want the name %q", rec, ok, err, c.name)
		}
		if members, err := c.store.Members(ctx, "idx", "p", "t1"); err != nil || len(members) != c.wantMembers {
			t.Errorf("%s: Members = %+v, %v; want %d", c.name, members, err, c.wantMembers)
		}
	}
}

// TestGenerationObjectsCarryItsPrefix: every table and index a generation's
// store creates is named by tablename, autotag_<generation>_<hash>_*, so a
// retired generation can be dropped by name. SQLite's own sqlite_* objects are
// not the store's.
func TestGenerationObjectsCarryItsPrefix(t *testing.T) {
	ctx := t.Context()
	db := openSQLite(t)
	s, err := New(db, SQLite, "g2")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Tags(ctx, "idx", "p"); err != nil {
		t.Fatal(err)
	}

	rows, err := db.QueryContext(ctx, `SELECT name FROM sqlite_master WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite\_%' ESCAPE '\' ORDER BY name`)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rows.Close() }()
	var got []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		got = append(got, name)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}

	want := []string{
		tableOf(t, "g2", "idx", "items"),
		tableOf(t, "g2", "idx", "links"),
		tableOf(t, "g2", "idx", "links") + "_tag",
		tableOf(t, "g2", "idx", "tags"),
	}
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Errorf("the store created %v, want %v", got, want)
	}
}

// TestStoresOnOneDatabaseShareTheirData: what a store wrote is what another
// store on the same database reads, as after a restart, and the second one's
// creating the tables again changes nothing.
func TestStoresOnOneDatabaseShareTheirData(t *testing.T) {
	ctx := t.Context()
	db := openSQLite(t)
	first, err := New(db, SQLite, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.PutTag(ctx, "idx", "p", "t1", "a"); err != nil {
		t.Fatal(err)
	}
	item := autotag.ItemRecord{ID: "i1", Text: "x", Vec: []float32{0.5, -1}, Model: "m"}
	if _, err := first.PutItem(ctx, "idx", "p", item); err != nil {
		t.Fatal(err)
	}
	if _, err := first.Link(ctx, "idx", "p", "i1", "t1"); err != nil {
		t.Fatal(err)
	}

	second, err := New(db, SQLite, "")
	if err != nil {
		t.Fatal(err)
	}
	if rec, ok, err := second.Tag(ctx, "idx", "p", "t1"); err != nil || !ok || rec.Name != "a" || rec.Version != 2 {
		t.Errorf("Tag = %+v, %v, %v; want the tag first wrote, at Version 2 after its Link", rec, ok, err)
	}
	if got, ok, err := second.Item(ctx, "idx", "p", "i1"); err != nil || !ok || got.Text != item.Text || !slices.Equal(got.Vec, item.Vec) {
		t.Errorf("Item = %+v, %v, %v; want %+v", got, ok, err, item)
	}
	if members, err := second.Members(ctx, "idx", "p", "t1"); err != nil || len(members) != 1 || members[0].ItemID != "i1" {
		t.Errorf("Members = %+v, %v; want i1", members, err)
	}
}

// TestNilVectorsStayNil: the Store contract says a vector never set stays nil,
// so an item without one is stored as NULL, and every way an item reaches the
// store, and every way it is read, gives nil back.
func TestNilVectorsStayNil(t *testing.T) {
	ctx := t.Context()
	s, err := New(openSQLite(t), SQLite, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.PutTag(ctx, "idx", "p", "t1", "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PutItem(ctx, "idx", "p", autotag.ItemRecord{ID: "i1", Text: "x", Model: "m"}); err != nil {
		t.Fatalf("PutItem without a vector: %v", err)
	}
	if _, err := s.Link(ctx, "idx", "p", "i1", "t1"); err != nil {
		t.Fatal(err)
	}
	check := func(when string) {
		t.Helper()
		item, ok, err := s.Item(ctx, "idx", "p", "i1")
		if err != nil || !ok || item.Vec != nil {
			t.Errorf("Item %s = %+v, %v, %v; want it with a nil Vec", when, item, ok, err)
		}
		members, err := s.Members(ctx, "idx", "p", "t1")
		if err != nil || len(members) != 1 || members[0].Vec != nil {
			t.Errorf("Members %s = %+v, %v; want one with a nil Vec", when, members, err)
		}
	}
	check("after PutItem")

	if _, stored, err := s.SetItemVector(ctx, "idx", "p", "i1", "x", []float32{1}, "m"); err != nil || !stored {
		t.Fatalf("SetItemVector = %v, %v; want it stored", stored, err)
	}
	if _, stored, err := s.SetItemVector(ctx, "idx", "p", "i1", "x", nil, "m"); err != nil || !stored {
		t.Fatalf("SetItemVector of a nil vector = %v, %v; want it stored", stored, err)
	}
	check("after SetItemVector of nil")
}

// TestReads_FailOnATornBlob: a vector cut short in the database is an error in
// every read that returns one, not a wrong or an empty vector, and the reads
// that do not return it still work.
func TestReads_FailOnATornBlob(t *testing.T) {
	tags, items := tableOf(t, "", "idx", "tags"), tableOf(t, "", "idx", "items")
	cases := []struct {
		name    string
		tear    string   // the statement that cuts a blob short
		failing []string // the reads that return that blob
	}{
		{"name vector", `UPDATE ` + tags + ` SET name_vec = x'010203'`, []string{"Tag", "Tags"}},
		{"tag vector", `UPDATE ` + tags + ` SET vec = x'010203'`, []string{"Tag", "Tags"}},
		{"item vector", `UPDATE ` + items + ` SET vec = x'010203'`, []string{"Item", "Members"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ctx := t.Context()
			db := openSQLite(t)
			s, err := New(db, SQLite, "")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := s.PutTag(ctx, "idx", "p", "t1", "a"); err != nil {
				t.Fatal(err)
			}
			if _, err := s.PutItem(ctx, "idx", "p", autotag.ItemRecord{ID: "i1", Text: "x", Vec: []float32{1}, Model: "m"}); err != nil {
				t.Fatal(err)
			}
			if _, err := s.Link(ctx, "idx", "p", "i1", "t1"); err != nil {
				t.Fatal(err)
			}
			if _, err := db.ExecContext(ctx, c.tear); err != nil {
				t.Fatalf("tearing the %s: %v", c.name, err)
			}

			reads := map[string]func() error{
				"Tag":     func() error { _, _, err := s.Tag(ctx, "idx", "p", "t1"); return err },
				"Tags":    func() error { _, err := s.Tags(ctx, "idx", "p"); return err },
				"Item":    func() error { _, _, err := s.Item(ctx, "idx", "p", "i1"); return err },
				"Members": func() error { _, err := s.Members(ctx, "idx", "p", "t1"); return err },
			}
			for name, read := range reads {
				if err, want := read(), slices.Contains(c.failing, name); (err != nil) != want {
					t.Errorf("%s with a torn %s: err = %v, want an error: %v", name, c.name, err, want)
				}
			}
		})
	}
}

// Package vectortest is the conformance suite for vector.Store. Every backend
// runs it from its own tests, so that all of them keep the contract the same
// way:
//
//	func TestConformance(t *testing.T) {
//		vectortest.Run(t, func(t *testing.T) vector.Store { return open(t, vectortest.Dims) })
//	}
package vectortest

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"sync"
	"testing"

	"github.com/go-bumbu/langai/vector"
)

// Dims is the number of dimensions of every vector the suite writes; the
// store newStore returns must take vectors of this size.
const Dims = 4

// coord is where most cases write; other is another collection of its index.
var (
	coord = vector.Coord{Index: "tenant", Partition: "p", Collection: "c"}
	other = vector.Coord{Index: "tenant", Partition: "p", Collection: "other"}
)

// Run runs the suite against the store newStore returns. Each subtest calls
// newStore with its own t, so a backend can tie what it opens to t.Cleanup,
// and the store it returns must hold nothing yet.
func Run(t *testing.T, newStore func(t *testing.T) vector.Store) {
	cases := []struct {
		name string
		run  func(t *testing.T, s vector.Store)
	}{
		{"UpsertReplaces", upsertReplaces},
		{"BatchUpsertIsAtomic", batchUpsertIsAtomic},
		{"DeleteRemovesIDs", deleteRemovesIDs},
		{"HashesOfKnownIDs", hashesOfKnownIDs},
		{"DocsListsByID", docsListsByID},
		{"ChunksRoundTrip", chunksRoundTrip},
		{"SlicesAreCopies", slicesAreCopies},
		{"IndexesAreIsolated", indexesAreIsolated},
		{"CoordsMatchExactly", coordsMatchExactly},
		{"ScopeOpensCoords", scopeOpensCoords},
		{"OnlyAndExclude", onlyAndExclude},
		{"OneMatchPerDoc", oneMatchPerDoc},
		{"LargeLimit", largeLimit},
		{"ScoresAreCosine", scoresAreCosine},
		{"InvalidVectors", invalidVectors},
		{"DimensionMismatch", dimensionMismatch},
		{"ModelMismatch", modelMismatch},
		{"DeleteIndex", deleteIndex},
		{"TiesAreOrdered", tiesAreOrdered},
		{"ConcurrentSameDoc", concurrentSameDoc},
		{"ConcurrentDocs", concurrentDocs},
		{"UnknownIndexReads", unknownIndexReads},
		{"EmptyIndexIsAnError", emptyIndexIsAnError},
		{"OddIDs", oddIDs},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) { c.run(t, newStore(t)) })
	}
}

// axis returns the unit vector along axis i, so cosine scores are easy to
// predict: 1 with itself, 0 with another axis.
func axis(i int) []float32 {
	v := make([]float32, Dims)
	v[i] = 1
	return v
}

// near returns a unit-ish vector close to axis i, nudged towards axis j by
// eps; a larger eps scores lower against axis i.
func near(i, j int, eps float32) []float32 {
	v := axis(i)
	v[j] += eps
	return v
}

// doc returns a document of one chunk with the given vector, its text the id.
func doc(id string, v []float32) vector.Doc {
	return vector.Doc{ID: id, Hash: "h-" + id, Chunks: []vector.Chunk{{Seq: 0, Text: id, Vector: v}}}
}

func upsert(t *testing.T, s vector.Store, c vector.Coord, model string, docs ...vector.Doc) {
	t.Helper()
	if err := s.Upsert(context.Background(), c, model, docs...); err != nil {
		t.Fatalf("Upsert %v: %v", c, err)
	}
}

func search(t *testing.T, s vector.Store, c vector.Coord, q vector.Query) []vector.Match {
	t.Helper()
	got, err := s.Search(context.Background(), c, q)
	if err != nil {
		t.Fatalf("Search %v: %v", c, err)
	}
	return got
}

func hashes(t *testing.T, s vector.Store, c vector.Coord, ids ...string) map[string]string {
	t.Helper()
	got, err := s.Hashes(context.Background(), c, ids...)
	if err != nil {
		t.Fatalf("Hashes %v: %v", c, err)
	}
	return got
}

// ids returns the document ids of ms in order.
func ids(ms []vector.Match) []string {
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.DocID
	}
	return out
}

// where returns "partition/collection:id" for each of ms, in order.
func where(ms []vector.Match) []string {
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.Coord.Partition + "/" + m.Coord.Collection + ":" + m.DocID
	}
	return out
}

func upsertReplaces(t *testing.T, s vector.Store) {
	ctx := context.Background()
	upsert(t, s, coord, "", vector.Doc{ID: "doc", Hash: "h1", Chunks: []vector.Chunk{
		{Seq: 0, Text: "old-0", Vector: axis(0)},
		{Seq: 1, Text: "old-1", Vector: axis(1)},
		{Seq: 2, Text: "old-2", Vector: axis(2)},
	}})
	upsert(t, s, coord, "", vector.Doc{ID: "doc", Hash: "h2", Chunks: []vector.Chunk{{Seq: 0, Text: "new-0", Vector: axis(3)}}})
	got := search(t, s, coord, vector.Query{Vector: axis(0), Limit: 10})
	if len(got) != 1 || got[0].Text != "new-0" {
		t.Fatalf("search after replace = %+v, want only the new chunk", got)
	}
	chunks, err := s.Chunks(ctx, coord, "doc")
	if err != nil || len(chunks) != 1 || chunks[0].Text != "new-0" {
		t.Fatalf("Chunks after replace = %+v, %v", chunks, err)
	}
	if h := hashes(t, s, coord, "doc"); h["doc"] != "h2" {
		t.Fatalf("hash after replace = %q, want h2", h["doc"])
	}
	// no chunks: the hash is kept and nothing is found
	upsert(t, s, coord, "", vector.Doc{ID: "doc", Hash: "h3"})
	if got := search(t, s, coord, vector.Query{Vector: axis(3), Limit: 10}); len(got) != 0 {
		t.Fatalf("a document without chunks was found: %+v", got)
	}
	if h := hashes(t, s, coord, "doc"); h["doc"] != "h3" {
		t.Fatalf("hash of a document without chunks = %q, want h3", h["doc"])
	}
}

func batchUpsertIsAtomic(t *testing.T, s vector.Store) {
	ctx := context.Background()
	upsert(t, s, coord, "", doc("a", axis(0)), doc("b", axis(1)), doc("c", axis(2)))
	if got := hashes(t, s, coord, "a", "b", "c"); len(got) != 3 {
		t.Fatalf("batch stored %v, want all three", got)
	}
	bad := doc("e", []float32{float32(math.NaN()), 0, 0, 1})
	if err := s.Upsert(ctx, coord, "", doc("d", axis(0)), bad); !errors.Is(err, vector.ErrInvalidVector) {
		t.Fatalf("batch with a NaN vector: %v, want ErrInvalidVector", err)
	}
	if err := s.Upsert(ctx, coord, "", doc("a", axis(3)), doc("a", axis(3))); err == nil {
		t.Fatal("batch with a repeated id was accepted")
	}
	if err := s.Upsert(ctx, coord, "", doc("", axis(3))); err == nil {
		t.Fatal("empty document id was accepted")
	}
	if got := hashes(t, s, coord, "a", "d", "e"); len(got) != 1 || got["a"] != "h-a" {
		t.Fatalf("a failed batch left %v, want only the earlier a", got)
	}
	if got := search(t, s, coord, vector.Query{Vector: axis(3), Limit: 10}); slices.Contains(ids(got), "d") {
		t.Fatalf("a failed batch left searchable chunks: %v", ids(got))
	}
}

func deleteRemovesIDs(t *testing.T, s vector.Store) {
	ctx := context.Background()
	upsert(t, s, coord, "", doc("a", axis(0)), doc("b", axis(0)), doc("c", axis(0)))
	upsert(t, s, other, "", doc("a", axis(0)))
	if err := s.Delete(ctx, coord, "a", "c", "missing"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if got := ids(search(t, s, coord, vector.Query{Vector: axis(0), Limit: 10})); !slices.Equal(got, []string{"b"}) {
		t.Fatalf("search after delete = %v, want [b]", got)
	}
	if got := hashes(t, s, coord, "a", "b", "c"); len(got) != 1 {
		t.Fatalf("hashes after delete = %v, want only b", got)
	}
	if got := hashes(t, s, other, "a"); len(got) != 1 {
		t.Fatalf("delete reached another collection: %v", got)
	}
	if err := s.Delete(ctx, coord); err != nil {
		t.Fatalf("Delete of no ids: %v", err)
	}
}

func hashesOfKnownIDs(t *testing.T, s vector.Store) {
	upsert(t, s, coord, "", doc("a", axis(0)), vector.Doc{ID: "empty", Hash: "h-empty"})
	got := hashes(t, s, coord, "a", "empty", "missing")
	want := map[string]string{"a": "h-a", "empty": "h-empty"}
	if len(got) != len(want) || got["a"] != want["a"] || got["empty"] != want["empty"] {
		t.Fatalf("Hashes = %v, want %v", got, want)
	}
	if got := hashes(t, s, coord); len(got) != 0 {
		t.Fatalf("Hashes of no ids = %v", got)
	}
	if got := hashes(t, s, vector.Coord{Index: coord.Index, Partition: "x", Collection: coord.Collection}, "a"); len(got) != 0 {
		t.Fatalf("Hashes of another partition = %v", got)
	}
}

func docsListsByID(t *testing.T, s vector.Store) {
	ctx := context.Background()
	// byte order: upper case before lower case before multi-byte runes
	all := []string{"B", "a", "b", "b2", "é"}
	for _, id := range []string{"é", "b", "a", "b2", "B"} {
		upsert(t, s, coord, "", doc(id, axis(0)))
	}
	upsert(t, s, other, "", doc("zzz", axis(0)))
	list, err := s.Docs(ctx, coord, "", 0)
	if err != nil {
		t.Fatalf("Docs: %v", err)
	}
	var got []string
	for _, d := range list {
		got = append(got, d.ID)
		if d.Hash != "h-"+d.ID {
			t.Errorf("Docs hash of %s = %q", d.ID, d.Hash)
		}
	}
	if !slices.Equal(got, all) {
		t.Fatalf("Docs = %v, want %v", got, all)
	}
	// paging: two at a time, each page after the last id of the one before
	var paged []string
	for after := ""; ; {
		page, err := s.Docs(ctx, coord, after, 2)
		if err != nil {
			t.Fatalf("Docs page: %v", err)
		}
		if len(page) == 0 {
			break
		}
		if len(page) > 2 {
			t.Fatalf("page of %d, want at most 2", len(page))
		}
		for _, d := range page {
			paged = append(paged, d.ID)
		}
		after = page[len(page)-1].ID
	}
	if !slices.Equal(paged, all) {
		t.Fatalf("paged Docs = %v, want %v", paged, all)
	}
}

func chunksRoundTrip(t *testing.T, s vector.Store) {
	ctx := context.Background()
	want := []vector.Chunk{
		{Seq: 0, Text: "a", Vector: []float32{1, 0, 0, 0}},
		{Seq: 4000, Text: "b", Vector: []float32{0, 0.5, -1.25, 1.5e19}},
	}
	upsert(t, s, coord, "", vector.Doc{ID: "doc", Hash: "h", Chunks: []vector.Chunk{want[1], want[0]}})
	upsert(t, s, coord, "", doc("other", axis(1)))
	got, err := s.Chunks(ctx, coord, "doc")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("Chunks = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i].Seq != want[i].Seq || got[i].Text != want[i].Text || !slices.Equal(got[i].Vector, want[i].Vector) {
			t.Errorf("chunk %d = %+v, want %+v", i, got[i], want[i])
		}
	}
	for _, c := range []vector.Coord{{Index: "else", Partition: "p", Collection: "c"}, {Index: "tenant", Partition: "x", Collection: "c"}, other} {
		if got, err := s.Chunks(ctx, c, "doc"); err != nil || len(got) != 0 {
			t.Errorf("Chunks at %v = %v, %v; want nothing", c, got, err)
		}
	}
	if got, err := s.Chunks(ctx, coord, "missing"); err != nil || len(got) != 0 {
		t.Errorf("Chunks of an unknown doc = %v, %v", got, err)
	}
}

func slicesAreCopies(t *testing.T, s vector.Store) {
	ctx := context.Background()
	v := axis(0)
	upsert(t, s, coord, "", doc("doc", v))
	v[0], v[1] = 0, 1 // the caller reuses its slice
	got, err := s.Chunks(ctx, coord, "doc")
	if err != nil || len(got) != 1 || !slices.Equal(got[0].Vector, axis(0)) {
		t.Fatalf("the store kept the caller's slice: %+v, %v", got, err)
	}
	got[0].Vector[0] = 9
	again, err := s.Chunks(ctx, coord, "doc")
	if err != nil || again[0].Vector[0] != 1 {
		t.Fatalf("Chunks handed out the stored slice: %+v, %v", again, err)
	}
}

func indexesAreIsolated(t *testing.T, s vector.Store) {
	// same coordinates and id, indexes that differ only in case
	upper := vector.Coord{Index: "A", Partition: "p", Collection: "c"}
	lower := vector.Coord{Index: "a", Partition: "p", Collection: "c"}
	upsert(t, s, upper, "", vector.Doc{ID: "1", Hash: "upper", Chunks: []vector.Chunk{{Text: "upper", Vector: axis(0)}}})
	upsert(t, s, lower, "", vector.Doc{ID: "1", Hash: "lower", Chunks: []vector.Chunk{{Text: "lower", Vector: axis(0)}}})
	for c, want := range map[vector.Coord]string{upper: "upper", lower: "lower"} {
		got := search(t, s, c, vector.Query{Vector: axis(0), Limit: 10, AllPartitions: true, AllCollections: true})
		if len(got) != 1 || got[0].Text != want || got[0].Coord != c {
			t.Errorf("search of index %q = %+v, want only its own", c.Index, got)
		}
		if h := hashes(t, s, c, "1"); h["1"] != want {
			t.Errorf("hash in index %q = %q, want %q", c.Index, h["1"], want)
		}
	}
}

func coordsMatchExactly(t *testing.T, s vector.Store) {
	at := func(p, c string) vector.Coord { return vector.Coord{Index: "tenant", Partition: p, Collection: c} }
	upsert(t, s, at("10", "tags"), "", doc("a", axis(0)))
	upsert(t, s, at("10", "notes"), "", doc("b", axis(0)))
	upsert(t, s, at("20", "tags"), "", doc("c", axis(0)))
	upsert(t, s, at("", "tags"), "", doc("d", axis(0)))
	upsert(t, s, at("10", ""), "", doc("e", axis(0)))
	upsert(t, s, at("", ""), "", doc("f", axis(0)))
	for _, tc := range []struct {
		at   vector.Coord
		want string
	}{
		{at("10", "tags"), "a"},
		{at("", "tags"), "d"},
		{at("10", ""), "e"},
		{at("", ""), "f"},
	} {
		got := ids(search(t, s, tc.at, vector.Query{Vector: axis(0), Limit: 10}))
		if !slices.Equal(got, []string{tc.want}) {
			t.Errorf("search at %v = %v, want [%s]: an empty coordinate is a value, not a wildcard", tc.at, got, tc.want)
		}
	}
	if got := ids(search(t, s, at("30", "tags"), vector.Query{Vector: axis(0), Limit: 10})); len(got) != 0 {
		t.Errorf("search of an empty partition = %v", got)
	}
}

func scopeOpensCoords(t *testing.T, s vector.Store) {
	at := func(p, c string) vector.Coord { return vector.Coord{Index: "tenant", Partition: p, Collection: c} }
	upsert(t, s, at("10", "tags"), "", doc("a", near(0, 1, 0.1)))
	upsert(t, s, at("10", "notes"), "", doc("b", near(0, 1, 0.2)))
	upsert(t, s, at("20", "tags"), "", doc("c", near(0, 1, 0.3)))
	upsert(t, s, at("20", "tags"), "", doc("a", near(0, 1, 0.4))) // same id, other partition
	upsert(t, s, vector.Coord{Index: "else", Partition: "10", Collection: "tags"}, "", doc("x", axis(0)))
	q := vector.Query{Vector: axis(0), Limit: 10}
	for _, tc := range []struct {
		name string
		at   vector.Coord
		q    func(vector.Query) vector.Query
		want []string
	}{
		{"partition only", at("10", "tags"), func(q vector.Query) vector.Query { q.AllCollections = true; return q }, []string{"10/tags:a", "10/notes:b"}},
		{"collection only", at("10", "tags"), func(q vector.Query) vector.Query { q.AllPartitions = true; return q }, []string{"10/tags:a", "20/tags:c", "20/tags:a"}},
		{"whole index", at("", ""), func(q vector.Query) vector.Query { q.AllPartitions, q.AllCollections = true, true; return q }, []string{"10/tags:a", "10/notes:b", "20/tags:c", "20/tags:a"}},
	} {
		got := where(search(t, s, tc.at, tc.q(q)))
		if !slices.Equal(got, tc.want) {
			t.Errorf("%s: %v, want %v", tc.name, got, tc.want)
		}
	}
}

func onlyAndExclude(t *testing.T, s vector.Store) {
	upsert(t, s, coord, "", doc("a", near(0, 1, 0.1)), doc("b", near(0, 1, 0.2)), doc("c", near(0, 1, 0.3)), doc("d", near(0, 1, 0.4)))
	for _, tc := range []struct {
		name string
		q    vector.Query
		want []string
	}{
		{"only", vector.Query{Only: []string{"d", "b", "missing"}}, []string{"b", "d"}},
		{"exclude", vector.Query{Exclude: []string{"a", "c"}}, []string{"b", "d"}},
		{"both", vector.Query{Only: []string{"a", "b", "c"}, Exclude: []string{"a"}}, []string{"b", "c"}},
		{"only within limit", vector.Query{Only: []string{"a", "b", "c"}, Limit: 2}, []string{"a", "b"}},
	} {
		q := tc.q
		q.Vector = axis(0)
		if q.Limit == 0 {
			q.Limit = 10
		}
		if got := ids(search(t, s, coord, q)); !slices.Equal(got, tc.want) {
			t.Errorf("%s: %v, want %v", tc.name, got, tc.want)
		}
	}
}

// oneMatchPerDoc: a document with many chunks next to the query, a
// near-static video's frames, must not crowd the other documents out.
func oneMatchPerDoc(t *testing.T, s vector.Store) {
	video := vector.Doc{ID: "video", Hash: "h"}
	for i := range 64 {
		video.Chunks = append(video.Chunks, vector.Chunk{Seq: i * 1000, Vector: near(0, 1, float32(i+1)/1000)})
	}
	upsert(t, s, coord, "", video)
	var want []string
	for i := range 12 {
		id := fmt.Sprintf("doc%02d", i)
		want = append(want, id)
		upsert(t, s, coord, "", doc(id, near(0, 2, 0.5+float32(i)/10)))
	}
	got := search(t, s, coord, vector.Query{Vector: axis(0), Limit: 10})
	if !slices.Equal(ids(got), append([]string{"video"}, want[:9]...)) {
		t.Fatalf("search = %v, want video then doc00..doc08", ids(got))
	}
	if got[0].Seq != 0 {
		t.Errorf("video matched by chunk %d, want its best, 0", got[0].Seq)
	}
}

// largeLimit asks for more documents than SQLite's KNN query may return at
// once (4096).
func largeLimit(t *testing.T, s vector.Store) {
	const n = 4200
	docs := make([]vector.Doc, n)
	for i := range docs {
		docs[i] = doc(fmt.Sprintf("d%05d", i), near(0, 1+i%3, float32(i)/n))
	}
	upsert(t, s, coord, "", docs...)
	got := search(t, s, coord, vector.Query{Vector: axis(0), Limit: n + 10})
	if len(got) != n {
		t.Fatalf("search with limit %d returned %d of %d documents", n+10, len(got), n)
	}
	if got[0].DocID != "d00000" {
		t.Errorf("best match = %s, want d00000", got[0].DocID)
	}
	list, err := s.Docs(context.Background(), coord, "", 0)
	if err != nil || len(list) != n {
		t.Fatalf("Docs = %d documents, %v; want %d", len(list), err, n)
	}
}

func scoresAreCosine(t *testing.T, s vector.Store) {
	upsert(t, s, coord, "", doc("same", []float32{2, 0, 0, 0}), doc("diag", []float32{1, 1, 0, 0}), doc("orth", axis(1)), doc("opp", []float32{-1, 0, 0, 0}))
	got := search(t, s, coord, vector.Query{Vector: axis(0), Limit: 10})
	want := map[string]float64{"same": 1, "diag": math.Sqrt2 / 2, "orth": 0, "opp": -1}
	if !slices.Equal(ids(got), []string{"same", "diag", "orth", "opp"}) {
		t.Fatalf("order = %v", ids(got))
	}
	for _, m := range got {
		if math.Abs(m.Score-want[m.DocID]) > 1e-5 {
			t.Errorf("score of %s = %v, want %v", m.DocID, m.Score, want[m.DocID])
		}
	}
	if got := search(t, s, coord, vector.Query{Vector: axis(0), Limit: 0}); len(got) != 0 {
		t.Errorf("limit 0 returned %v", got)
	}
}

func invalidVectors(t *testing.T, s vector.Store) {
	ctx := context.Background()
	upsert(t, s, coord, "", doc("ok", axis(0)))
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	for name, v := range map[string][]float32{
		"zero": make([]float32, Dims),
		"NaN":  {nan, 0, 0, 1},
		"Inf":  {inf, 0, 0, 1},
		// the backends compute the norm in float32: its square underflows to
		// 0 (SQLite would score it +Inf) or overflows to infinity
		"tiny": {1e-25, 1e-25, 0, 0},
		"huge": {1e20, 1e20, 0, 0},
	} {
		if err := s.Upsert(ctx, coord, "", doc("bad", v)); !errors.Is(err, vector.ErrInvalidVector) {
			t.Errorf("Upsert of a %s vector: %v, want ErrInvalidVector", name, err)
		}
		if _, err := s.Search(ctx, coord, vector.Query{Vector: v, Limit: 10}); !errors.Is(err, vector.ErrInvalidVector) {
			t.Errorf("Search with a %s vector: %v, want ErrInvalidVector", name, err)
		}
	}
	// the refused vectors left the scope searchable
	if got := ids(search(t, s, coord, vector.Query{Vector: axis(0), Limit: 10})); !slices.Equal(got, []string{"ok"}) {
		t.Fatalf("search after refused writes = %v, want [ok]", got)
	}
}

func dimensionMismatch(t *testing.T, s vector.Store) {
	ctx := context.Background()
	upsert(t, s, coord, "", doc("ok", axis(0)))
	short := []float32{1, 0, 0}
	if err := s.Upsert(ctx, coord, "", doc("short", short)); !errors.Is(err, vector.ErrDimensions) {
		t.Errorf("Upsert of a %d-d vector: %v, want ErrDimensions", len(short), err)
	}
	if err := s.Upsert(ctx, other, "", doc("long", append(axis(0), 1))); !errors.Is(err, vector.ErrDimensions) {
		t.Errorf("Upsert of a %d-d vector in another collection: %v, want ErrDimensions", Dims+1, err)
	}
	if _, err := s.Search(ctx, coord, vector.Query{Vector: short, Limit: 10}); !errors.Is(err, vector.ErrDimensions) {
		t.Errorf("Search with a %d-d vector: %v, want ErrDimensions", len(short), err)
	}
}

func modelMismatch(t *testing.T, s vector.Store) {
	ctx := context.Background()
	upsert(t, s, coord, "", doc("unknown", axis(0))) // no model: nothing recorded
	upsert(t, s, coord, "m1", doc("a", axis(0)))
	upsert(t, s, coord, "", doc("b", axis(0)))   // no model: no check
	upsert(t, s, other, "m2", doc("c", axis(0))) // another collection, another model
	otherPartition := vector.Coord{Index: coord.Index, Partition: "q", Collection: coord.Collection}
	if err := s.Upsert(ctx, otherPartition, "m2", doc("d", axis(0))); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("Upsert of m2 into m1's collection (another partition): %v, want ErrModelMismatch", err)
	}
	if err := s.Upsert(ctx, coord, "m2", doc("a", axis(1))); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("Upsert of m2 into m1's collection: %v, want ErrModelMismatch", err)
	}
	if got, err := s.Chunks(ctx, coord, "a"); err != nil || len(got) != 1 || !slices.Equal(got[0].Vector, axis(0)) {
		t.Errorf("a refused write changed the document: %+v, %v", got, err)
	}
	if _, err := s.Search(ctx, coord, vector.Query{Vector: axis(0), Limit: 10, Model: "m2"}); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("Search with m2 in m1's collection: %v, want ErrModelMismatch", err)
	}
	for _, q := range []vector.Query{{Model: "m1"}, {Model: ""}, {Model: "m1", AllPartitions: true}} {
		q.Vector, q.Limit = axis(0), 10
		if got := search(t, s, coord, q); len(got) != 3 {
			t.Errorf("Search %+v = %v, want unknown, a and b", q, ids(got))
		}
	}
	if _, err := s.Search(ctx, coord, vector.Query{Vector: axis(0), Limit: 10, Model: "m1", AllCollections: true}); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("Search with m1 across collections, one of them m2's: %v, want ErrModelMismatch", err)
	}
}

func deleteIndex(t *testing.T, s vector.Store) {
	ctx := context.Background()
	upsert(t, s, coord, "m1", doc("a", axis(0)))
	upsert(t, s, other, "m1", doc("b", axis(0)))
	upsert(t, s, vector.Coord{Index: coord.Index, Partition: "q", Collection: "c"}, "m1", doc("c", axis(0)))
	keep := vector.Coord{Index: "kept", Partition: "p", Collection: "c"}
	upsert(t, s, keep, "m1", doc("k", axis(0)))
	if err := s.DeleteIndex(ctx, coord.Index); err != nil {
		t.Fatalf("DeleteIndex: %v", err)
	}
	all := vector.Query{Vector: axis(0), Limit: 10, AllPartitions: true, AllCollections: true}
	if got := search(t, s, coord, all); len(got) != 0 {
		t.Fatalf("search after DeleteIndex = %v", where(got))
	}
	if got := search(t, s, keep, all); len(got) != 1 {
		t.Fatalf("DeleteIndex reached another index: %v", where(got))
	}
	// the index is new again: another model may write to it
	upsert(t, s, coord, "m2", doc("a", axis(0)))
	if err := s.DeleteIndex(ctx, "never-written"); err != nil {
		t.Fatalf("DeleteIndex of an unknown index: %v", err)
	}
}

func tiesAreOrdered(t *testing.T, s vector.Store) {
	// equal scores: by document id, then by Seq within a document
	upsert(t, s, coord, "", doc("c", axis(0)), doc("a", axis(0)), doc("b", axis(0)))
	upsert(t, s, coord, "", vector.Doc{ID: "aa", Hash: "h", Chunks: []vector.Chunk{
		{Seq: 7, Text: "seven", Vector: axis(0)},
		{Seq: 3, Text: "three", Vector: axis(0)},
	}})
	want := []string{"a", "aa", "b", "c"}
	for range 3 {
		got := search(t, s, coord, vector.Query{Vector: axis(0), Limit: 10})
		if !slices.Equal(ids(got), want) {
			t.Fatalf("tied search = %v, want %v", ids(got), want)
		}
		if got[1].Seq != 3 {
			t.Fatalf("aa matched by chunk %d, want its lowest tied Seq, 3", got[1].Seq)
		}
	}
	// the same id in two partitions, scoped: by partition
	upsert(t, s, vector.Coord{Index: coord.Index, Partition: "a", Collection: coord.Collection}, "", doc("a", axis(0)))
	got := where(search(t, s, coord, vector.Query{Vector: axis(0), Limit: 2, AllPartitions: true}))
	if !slices.Equal(got, []string{"a/c:a", "p/c:a"}) {
		t.Fatalf("tied search across partitions = %v", got)
	}
}

// writers is how many goroutines the concurrency cases run.
const writers = 8

// concurrentSameDoc: writers replacing one document at once each leave a
// whole chunk set, never two mixed.
func concurrentSameDoc(t *testing.T, s vector.Store) {
	ctx := context.Background()
	upsert(t, s, coord, "", doc("seed", axis(0)))
	var wg sync.WaitGroup
	errs := make(chan error, writers*5)
	for w := range writers {
		wg.Go(func() {
			for range 5 {
				d := vector.Doc{ID: "doc", Hash: strconv.Itoa(w)}
				for i := range w + 1 {
					d.Chunks = append(d.Chunks, vector.Chunk{Seq: i, Text: strconv.Itoa(w), Vector: near(0, 1, float32(i+1)/10)})
				}
				errs <- s.Upsert(ctx, coord, "", d)
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Upsert: %v", err)
		}
	}
	chunks, err := s.Chunks(ctx, coord, "doc")
	if err != nil {
		t.Fatal(err)
	}
	hash := hashes(t, s, coord, "doc")["doc"]
	w, _ := strconv.Atoi(hash)
	if len(chunks) != w+1 {
		t.Fatalf("doc has %d chunks after concurrent writes, want the %d of writer %s", len(chunks), w+1, hash)
	}
	for _, ch := range chunks {
		if ch.Text != hash {
			t.Fatalf("doc mixes the chunks of writers %s and %s", ch.Text, hash)
		}
	}
}

// concurrentDocs: writers of different documents at once all succeed.
func concurrentDocs(t *testing.T, s vector.Store) {
	ctx := context.Background()
	upsert(t, s, coord, "m", doc("seed", axis(0)))
	var wg sync.WaitGroup
	errs := make(chan error, writers*10)
	for w := range writers {
		wg.Go(func() {
			for i := range 10 {
				errs <- s.Upsert(ctx, coord, "m", doc(fmt.Sprintf("w%d-%d", w, i), near(0, 1, float32(i+1)/10)))
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Upsert: %v", err)
		}
	}
	list, err := s.Docs(ctx, coord, "", 0)
	if err != nil || len(list) != writers*10+1 {
		t.Fatalf("Docs = %d documents, %v; want %d", len(list), err, writers*10+1)
	}
}

// unknownIndexReads: reads and deletes of an index never written find
// nothing and fail with nothing. (Backends check on their own that they
// created no tables.)
func unknownIndexReads(t *testing.T, s vector.Store) {
	ctx := context.Background()
	c := vector.Coord{Index: "never-written", Partition: "p", Collection: "c"}
	if got := hashes(t, s, c, "a"); len(got) != 0 {
		t.Errorf("Hashes = %v", got)
	}
	if got, err := s.Docs(ctx, c, "", 0); err != nil || len(got) != 0 {
		t.Errorf("Docs = %v, %v", got, err)
	}
	if got, err := s.Chunks(ctx, c, "a"); err != nil || len(got) != 0 {
		t.Errorf("Chunks = %v, %v", got, err)
	}
	if got := search(t, s, c, vector.Query{Vector: axis(0), Limit: 10, Model: "m", AllPartitions: true, AllCollections: true}); len(got) != 0 {
		t.Errorf("Search = %v", got)
	}
	if err := s.Delete(ctx, c, "a"); err != nil {
		t.Errorf("Delete: %v", err)
	}
}

// emptyIndexIsAnError: every method refuses an empty index with
// ErrEmptyIndex, whether the store has written anything or not.
func emptyIndexIsAnError(t *testing.T, s vector.Store) {
	ctx := context.Background()
	c := vector.Coord{Partition: "p", Collection: "c"}
	for _, written := range []bool{false, true} {
		if written {
			upsert(t, s, coord, "", doc("a", axis(0)))
		}
		for name, call := range calls(ctx, s, c, "a", "") {
			if err := call(); !errors.Is(err, vector.ErrEmptyIndex) {
				t.Errorf("%s with an empty index (store written %v): %v, want ErrEmptyIndex", name, written, err)
			}
		}
		if err := s.DeleteIndex(ctx, ""); !errors.Is(err, vector.ErrEmptyIndex) {
			t.Errorf("DeleteIndex of an empty index: %v, want ErrEmptyIndex", err)
		}
	}
}

// calls is every method of s that takes a coordinate, called at c with id
// wherever it takes a document id and model wherever it takes a model.
func calls(ctx context.Context, s vector.Store, c vector.Coord, id, model string) map[string]func() error {
	return map[string]func() error{
		"Upsert": func() error { return s.Upsert(ctx, c, model, doc(id, axis(0))) },
		"Delete": func() error { return s.Delete(ctx, c, id) },
		"Hashes": func() error { _, err := s.Hashes(ctx, c, id); return err },
		"Docs":   func() error { _, err := s.Docs(ctx, c, id, 0); return err },
		"Chunks": func() error { _, err := s.Chunks(ctx, c, id); return err },
		"Search": func() error {
			_, err := s.Search(ctx, c, vector.Query{Vector: axis(0), Limit: 10, Model: model, Only: []string{id}})
			return err
		},
		"Search exclude": func() error {
			_, err := s.Search(ctx, c, vector.Query{Vector: axis(0), Limit: 10, Model: model, Exclude: []string{id}})
			return err
		},
	}
}

// oddIDs: a document id, a partition, a collection or a model that is not
// valid UTF-8 or holds a NUL byte is refused by every method with
// ErrInvalidID, on every backend alike, before anything is read or
// written; Postgres refuses such text and SQLite's JSON id lists turn it
// into U+FFFD, so neither could keep it. Every other string, however odd,
// round-trips.
func oddIDs(t *testing.T, s vector.Store) {
	upsert(t, s, coord, "m", doc("kept", axis(0)))
	badIDsAreRefused(t, s)
	oddIDsRoundTrip(t, s)
}

// badIDsAreRefused is oddIDs' half that refuses.
func badIDsAreRefused(t *testing.T, s vector.Store) {
	ctx := context.Background()
	for _, bad := range []string{"caf\xe9.txt", "a\x00b", "\xff"} {
		for name, call := range calls(ctx, s, coord, bad, "m") {
			if err := call(); !errors.Is(err, vector.ErrInvalidID) {
				t.Errorf("%s of id %q: %v, want ErrInvalidID", name, bad, err)
			}
		}
		for _, c := range []vector.Coord{
			{Index: coord.Index, Partition: bad, Collection: coord.Collection},
			{Index: coord.Index, Partition: coord.Partition, Collection: bad},
		} {
			for name, call := range calls(ctx, s, c, "kept", "m") {
				if err := call(); !errors.Is(err, vector.ErrInvalidID) {
					t.Errorf("%s at %v: %v, want ErrInvalidID", name, c, err)
				}
			}
		}
		if err := s.Upsert(ctx, coord, bad, doc("x", axis(0))); !errors.Is(err, vector.ErrInvalidID) {
			t.Errorf("Upsert with model %q: %v, want ErrInvalidID", bad, err)
		}
		if _, err := s.Search(ctx, coord, vector.Query{Vector: axis(0), Limit: 10, Model: bad}); !errors.Is(err, vector.ErrInvalidID) {
			t.Errorf("Search with model %q: %v, want ErrInvalidID", bad, err)
		}
		// one bad id fails the whole batch: nothing of it is stored
		if err := s.Upsert(ctx, coord, "m", doc("good", axis(1)), doc(bad, axis(1))); !errors.Is(err, vector.ErrInvalidID) {
			t.Errorf("batch with id %q: %v, want ErrInvalidID", bad, err)
		}
	}
	if got := hashes(t, s, coord, "good"); len(got) != 0 {
		t.Errorf("a refused batch stored %v", got)
	}
	if err := s.Upsert(ctx, coord, "m", doc("", axis(0))); !errors.Is(err, vector.ErrInvalidID) {
		t.Errorf("Upsert of an empty id: %v, want ErrInvalidID", err)
	}
}

// oddIDsRoundTrip is oddIDs' half that keeps: odd but valid strings, which
// every operation finds.
func oddIDsRoundTrip(t *testing.T, s vector.Store) {
	ctx := context.Background()
	odd := []string{"café.txt", "日本語", `quote"d`, `back\slash`, "tab\tid", "line\nbreak", " ", "😀", "%_like*", " "}
	for _, id := range odd {
		upsert(t, s, coord, "m", doc(id, axis(1)))
	}
	got := hashes(t, s, coord, odd...)
	for _, id := range odd {
		if got[id] != "h-"+id {
			t.Errorf("Hashes[%q] = %q, want %q", id, got[id], "h-"+id)
		}
	}
	q := vector.Query{Vector: axis(1), Limit: 100, Model: "m", Exclude: odd[1:]}
	if found := ids(search(t, s, coord, q)); !slices.Equal(found, []string{odd[0], "kept"}) {
		t.Errorf("search excluding all odd ids but %q = %q", odd[0], found)
	}
	q = vector.Query{Vector: axis(1), Limit: 100, Only: odd[:3]}
	if found := ids(search(t, s, coord, q)); len(found) != 3 {
		t.Errorf("search of only %q = %q", odd[:3], found)
	}
	for _, id := range odd {
		if chunks, err := s.Chunks(ctx, coord, id); err != nil || len(chunks) != 1 {
			t.Errorf("Chunks(%q) = %v, %v", id, chunks, err)
		}
	}
	if err := s.Delete(ctx, coord, odd...); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if got := hashes(t, s, coord, odd...); len(got) != 0 {
		t.Errorf("Delete left %v", got)
	}
	// an odd partition and collection are values like any other
	at := vector.Coord{Index: coord.Index, Partition: "pä\"rt", Collection: `co\ll`}
	upsert(t, s, at, "m", doc("x", axis(0)))
	if got := hashes(t, s, at, "x"); got["x"] != "h-x" {
		t.Errorf("Hashes at %v = %v", at, got)
	}
}

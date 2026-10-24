package semsearch

import (
	"context"
	"testing"
)

// unit vectors along axes make cosine scores trivially predictable.
func axis(dims, i int) []float32 {
	v := make([]float32, dims)
	v[i] = 1
	return v
}

func TestMemStore_IndexIsolation(t *testing.T) {
	ctx := context.Background()
	s := NewMemStore()
	// same docID, same coordinates, two different indexes
	if err := s.Upsert(ctx, "tenantA", "10", "tags", "1", "h", []Chunk{{Seq: 0, Text: "a", Vector: axis(4, 0)}}); err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(ctx, "tenantB", "10", "tags", "1", "h", []Chunk{{Seq: 0, Text: "b", Vector: axis(4, 0)}}); err != nil {
		t.Fatal(err)
	}
	got, err := s.Search(ctx, "tenantA", "10", "tags", axis(4, 0), 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Text != "a" {
		t.Fatalf("index A search leaked or missed: %#v", got)
	}
}

func TestMemStore_CoordinateEquality(t *testing.T) {
	ctx := context.Background()
	s := NewMemStore()
	seed := func(part, coll, id string, ax int) {
		if err := s.Upsert(ctx, "t", part, coll, id, "h", []Chunk{{Seq: 0, Text: id, Vector: axis(4, ax)}}); err != nil {
			t.Fatal(err)
		}
	}
	seed("10", "tags", "a", 0)
	seed("10", "bookmarks", "b", 0)
	seed("20", "tags", "c", 0)

	// exact: only partition 10, tags
	got, _ := s.Search(ctx, "t", "10", "tags", axis(4, 0), 10)
	if len(got) != 1 || got[0].DocID != "a" {
		t.Fatalf("exact search wrong: %#v", got)
	}
}

func TestMemStore_SearchScopeSubsets(t *testing.T) {
	ctx := context.Background()
	s := NewMemStore()
	seed := func(part, coll, id string) {
		if err := s.Upsert(ctx, "t", part, coll, id, "h", []Chunk{{Seq: 0, Text: id, Vector: axis(4, 0)}}); err != nil {
			t.Fatal(err)
		}
	}
	seed("10", "tags", "a")
	seed("10", "bookmarks", "b")
	seed("20", "tags", "c")

	// partition only -> every collection in partition 10
	got, _ := s.SearchScope(ctx, "t", ScopeFilter{Partition: "10"}, axis(4, 0), 10)
	if !idSet(got, "a", "b") {
		t.Fatalf("partition-only wrong: %v", ids(got))
	}
	// collection only -> tags across all partitions
	got, _ = s.SearchScope(ctx, "t", ScopeFilter{Collection: "tags"}, axis(4, 0), 10)
	if !idSet(got, "a", "c") {
		t.Fatalf("collection-only wrong: %v", ids(got))
	}
	// neither -> everything in the index
	got, _ = s.SearchScope(ctx, "t", ScopeFilter{}, axis(4, 0), 10)
	if !idSet(got, "a", "b", "c") {
		t.Fatalf("unconstrained wrong: %v", ids(got))
	}
}

func ids(ms []Match) []string {
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.DocID
	}
	return out
}
func idSet(ms []Match, want ...string) bool {
	if len(ms) != len(want) {
		return false
	}
	have := map[string]bool{}
	for _, m := range ms {
		have[m.DocID] = true
	}
	for _, w := range want {
		if !have[w] {
			return false
		}
	}
	return true
}

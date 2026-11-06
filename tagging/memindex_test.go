package tagging_test

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/go-bumbu/langai/tagging"
)

func tv(id string, dim int) tagging.TagVector {
	return tagging.TagVector{ID: id, Name: "name-" + id, Fingerprint: "f" + id, Vec: make([]float32, dim), Cutoff: 0.5}
}

func ids(tags []tagging.TagVector) []string {
	out := make([]string, 0, len(tags))
	for _, t := range tags {
		out = append(out, t.ID)
	}
	return out
}

func fps(t *testing.T, m *tagging.MemIndex, index, partition string) (string, map[string]string, bool) {
	t.Helper()
	state, f, ok, err := m.State(context.Background(), index, partition)
	if err != nil {
		t.Fatal(err)
	}
	return state, f, ok
}

func apply(t *testing.T, m *tagging.MemIndex, index, partition, state string, put []tagging.TagVector, drop []string) {
	t.Helper()
	if err := m.Apply(context.Background(), index, partition, state, put, drop); err != nil {
		t.Fatal(err)
	}
}

func eq(a, b []string) bool { return fmt.Sprint(a) == fmt.Sprint(b) }

func TestMemIndex_ApplyThenStateAndCandidates(t *testing.T) {
	m := tagging.NewMemIndex(0)
	apply(t, m, "i", "p", "s1", []tagging.TagVector{tv("b", 3), tv("a", 3)}, nil)
	state, f, ok := fps(t, m, "i", "p")
	if state != "s1" || !ok || len(f) != 2 || f["a"] != "fa" || f["b"] != "fb" {
		t.Fatalf("State = %q %v %v", state, f, ok)
	}
	got, err := m.Candidates(context.Background(), "i", "p", []float32{1, 2, 3})
	if err != nil || !eq(ids(got), []string{"a", "b"}) {
		t.Fatalf("Candidates = %v %v", ids(got), err)
	}
}

func TestMemIndex_ApplyEmptyStateKeepsTheRecordedOne(t *testing.T) {
	m := tagging.NewMemIndex(0)
	apply(t, m, "i", "p", "s1", []tagging.TagVector{tv("a", 3), tv("b", 3)}, nil)
	apply(t, m, "i", "p", "", []tagging.TagVector{tv("c", 3)}, []string{"a"})
	state, f, ok := fps(t, m, "i", "p")
	if state != "s1" || !ok || len(f) != 2 || f["b"] == "" || f["c"] == "" {
		t.Fatalf("State = %q %v %v", state, f, ok)
	}
}

func TestMemIndex_PutWinsOverDrop(t *testing.T) {
	m := tagging.NewMemIndex(0)
	apply(t, m, "i", "p", "s1", nil, nil)
	apply(t, m, "i", "p", "", []tagging.TagVector{tv("a", 3)}, []string{"a"})
	if _, f, _ := fps(t, m, "i", "p"); len(f) != 1 {
		t.Fatalf("fingerprints = %v", f)
	}
}

func TestMemIndex_EmptySpaceWithStateIsOK(t *testing.T) {
	m := tagging.NewMemIndex(0)
	apply(t, m, "i", "p", "s1", nil, nil)
	state, f, ok := fps(t, m, "i", "p")
	if state != "s1" || !ok || len(f) != 0 {
		t.Fatalf("State = %q %v %v", state, f, ok)
	}
	got, err := m.Candidates(context.Background(), "i", "p", nil)
	if err != nil || len(got) != 0 {
		t.Fatalf("Candidates = %v %v", got, err)
	}
}

func TestMemIndex_UnknownSpace(t *testing.T) {
	m := tagging.NewMemIndex(0)
	if _, _, ok := fps(t, m, "i", "p"); ok {
		t.Fatal("unknown space is ok")
	}
	got, err := m.Candidates(context.Background(), "i", "p", nil)
	if got != nil || err != nil {
		t.Fatalf("Candidates = %v %v", got, err)
	}
	// An Apply that leaves nothing and records nothing creates nothing.
	apply(t, m, "i", "p", "", nil, []string{"x"})
	if _, _, ok := fps(t, m, "i", "p"); ok {
		t.Fatal("empty Apply created a space")
	}
}

func TestMemIndex_EmptyIndexIsAnError(t *testing.T) {
	m := tagging.NewMemIndex(0)
	ctx := context.Background()
	if _, _, _, err := m.State(ctx, "", "p"); err == nil {
		t.Error("State accepted an empty index")
	}
	if err := m.Apply(ctx, "", "p", "s", nil, nil); err == nil {
		t.Error("Apply accepted an empty index")
	}
	if _, err := m.Candidates(ctx, "", "p", nil); err == nil {
		t.Error("Candidates accepted an empty index")
	}
}

func TestMemIndex_SpacesAreIsolated(t *testing.T) {
	m := tagging.NewMemIndex(0)
	apply(t, m, "i1", "p", "s1", []tagging.TagVector{tv("a", 3)}, nil)
	apply(t, m, "i2", "p", "s2", []tagging.TagVector{tv("b", 3)}, nil)
	apply(t, m, "i1", "q", "s3", []tagging.TagVector{tv("c", 3)}, nil)
	for _, c := range []struct{ index, part, state, id string }{
		{"i1", "p", "s1", "a"}, {"i2", "p", "s2", "b"}, {"i1", "q", "s3", "c"},
	} {
		state, f, ok := fps(t, m, c.index, c.part)
		if !ok || state != c.state || len(f) != 1 || f[c.id] == "" {
			t.Errorf("%s/%s: State = %q %v %v", c.index, c.part, state, f, ok)
		}
	}
}

func TestMemIndex_EvictsLeastRecentlyUsedOverBudget(t *testing.T) {
	one := int64(4*1000 + len("a") + len("name-a") + len("fa") + 64)
	m := tagging.NewMemIndex(2*one + one/2) // about two spaces
	ctx := context.Background()
	apply(t, m, "i", "A", "s", []tagging.TagVector{tv("a", 1000)}, nil)
	apply(t, m, "i", "B", "s", []tagging.TagVector{tv("a", 1000)}, nil)
	apply(t, m, "i", "C", "s", []tagging.TagVector{tv("a", 1000)}, nil)
	if _, _, ok := fps(t, m, "i", "A"); ok {
		t.Error("A should be evicted")
	}
	for _, p := range []string{"B", "C"} {
		if _, _, ok := fps(t, m, "i", p); !ok {
			t.Errorf("%s should be held", p)
		}
	}

	m = tagging.NewMemIndex(2*one + one/2)
	apply(t, m, "i", "A", "s", []tagging.TagVector{tv("a", 1000)}, nil)
	apply(t, m, "i", "B", "s", []tagging.TagVector{tv("a", 1000)}, nil)
	if _, err := m.Candidates(ctx, "i", "A", nil); err != nil {
		t.Fatal(err)
	}
	apply(t, m, "i", "C", "s", []tagging.TagVector{tv("a", 1000)}, nil)
	if _, _, ok := fps(t, m, "i", "B"); ok {
		t.Error("B should be evicted")
	}
	for _, p := range []string{"A", "C"} {
		if _, _, ok := fps(t, m, "i", p); !ok {
			t.Errorf("%s should be held", p)
		}
	}
}

func TestMemIndex_KeepsASpaceLargerThanTheBudget(t *testing.T) {
	m := tagging.NewMemIndex(100)
	apply(t, m, "i", "p", "s", []tagging.TagVector{tv("a", 1000)}, nil)
	if _, _, ok := fps(t, m, "i", "p"); !ok {
		t.Fatal("oversized space was evicted")
	}
}

func TestMemIndex_ConcurrentUse(t *testing.T) {
	m := tagging.NewMemIndex(10_000)
	ctx := context.Background()
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			part := fmt.Sprint("p", g%3)
			for n := range 200 {
				_ = m.Apply(ctx, "i", part, "s", []tagging.TagVector{tv(fmt.Sprint(n%5), 100)}, []string{fmt.Sprint((n + 1) % 5)})
				_, _, _, _ = m.State(ctx, "i", part)
				_, _ = m.Candidates(ctx, "i", part, nil)
			}
		}()
	}
	wg.Wait()
}

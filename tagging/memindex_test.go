package tagging_test

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"testing"

	"github.com/go-bumbu/sense/tagging"
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

// apply is an Apply by a caller that has just read the space, so its from is
// still the state.
func apply(t *testing.T, m *tagging.MemIndex, index, partition, state string, put []tagging.TagVector, drop []string) {
	t.Helper()
	from, _, _ := fps(t, m, index, partition)
	if err := m.Apply(context.Background(), index, partition, from, state, put, drop); err != nil {
		t.Fatal(err)
	}
}

// applyFrom is an Apply to the space ("i", "p") by a caller that read the state
// from, whatever the space's state is now.
func applyFrom(t *testing.T, m *tagging.MemIndex, from, state string, drop []string, put ...tagging.TagVector) {
	t.Helper()
	if err := m.Apply(context.Background(), "i", "p", from, state, put, drop); err != nil {
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

// TestMemIndex_ApplyRecordsOnlyOverTheStateTheCallerRead: an Apply records its
// state only when the space is still as its caller read it. A stale one takes
// its put and drop, records nothing, and leaves a state no caller holds, so the
// next caller diffs again and can then record its own.
func TestMemIndex_ApplyRecordsOnlyOverTheStateTheCallerRead(t *testing.T) {
	stale := func(t *testing.T, m *tagging.MemIndex, from string) {
		t.Helper()
		applyFrom(t, m, from, "s2", nil, tv("b", 3))
		state, f, ok := fps(t, m, "i", "p")
		if !ok || state == "s2" || state == "" || f["b"] == "" {
			t.Fatalf("after a stale Apply: State = %q %v %v, want the put held under a state of the index's own", state, f, ok)
		}
		// a caller that reads that state and diffs against it can record its own
		apply(t, m, "i", "p", "s3", []tagging.TagVector{tv("c", 3)}, nil)
		if state, f, ok := fps(t, m, "i", "p"); !ok || state != "s3" || f["b"] == "" || f["c"] == "" {
			t.Fatalf("after an up-to-date Apply: State = %q %v %v, want s3 with b and c", state, f, ok)
		}
	}

	t.Run("the state moved", func(t *testing.T) {
		m := tagging.NewMemIndex(0)
		apply(t, m, "i", "p", "s1", []tagging.TagVector{tv("a", 3)}, nil)
		stale(t, m, "s0")
	})
	t.Run("the caller saw no space and there is one now", func(t *testing.T) {
		m := tagging.NewMemIndex(0)
		apply(t, m, "i", "p", "s1", []tagging.TagVector{tv("a", 3)}, nil)
		stale(t, m, "")
	})
	t.Run("the caller saw a space and it is gone", func(t *testing.T) {
		m := tagging.NewMemIndex(0)
		stale(t, m, "s1")
	})
	t.Run("an evicted space comes back unrecorded", func(t *testing.T) {
		m := tagging.NewMemIndex(1) // only the space just used is kept
		apply(t, m, "i", "p", "s1", []tagging.TagVector{tv("a", 3)}, nil)
		apply(t, m, "i", "q", "s1", []tagging.TagVector{tv("x", 3)}, nil)
		if _, _, ok := fps(t, m, "i", "p"); ok {
			t.Fatal("p should be evicted")
		}
		stale(t, m, "s1")
	})
}

// TestMemIndex_StaleAppliesLeaveNoStateToMatch: each stale Apply leaves a state
// of its own, so a caller that read the one before it does not record.
func TestMemIndex_StaleAppliesLeaveNoStateToMatch(t *testing.T) {
	m := tagging.NewMemIndex(0)
	apply(t, m, "i", "p", "s1", []tagging.TagVector{tv("a", 3)}, nil)
	applyFrom(t, m, "s0", "s2", nil, tv("b", 3))
	first, _, _ := fps(t, m, "i", "p")
	applyFrom(t, m, "s0", "s2", nil, tv("c", 3))
	applyFrom(t, m, first, "s4", nil, tv("d", 3))
	if state, _, _ := fps(t, m, "i", "p"); state == "s4" {
		t.Fatal("an Apply over a state another stale Apply moved was recorded")
	}
}

// TestMemIndex_StaleOverwriteIsNotRecordedOver: three rebuilds overlap. The one
// at s3 finishes, the one at s4 diffs against it and skips tag a, then the one
// at s2, the oldest, overwrites a with its vector. s4 must not be recorded over
// that: its next diff has to see a.
func TestMemIndex_StaleOverwriteIsNotRecordedOver(t *testing.T) {
	m := tagging.NewMemIndex(0)
	apply(t, m, "i", "p", "s1", []tagging.TagVector{{ID: "a", Fingerprint: "a1"}}, nil)
	applyFrom(t, m, "s1", "s3", nil, tagging.TagVector{ID: "a", Fingerprint: "a3"})
	applyFrom(t, m, "s1", "s2", nil, tagging.TagVector{ID: "a", Fingerprint: "a2"})
	applyFrom(t, m, "s3", "s4", nil, tagging.TagVector{ID: "b", Fingerprint: "b4"})
	state, f, _ := fps(t, m, "i", "p")
	if state == "s4" {
		t.Fatalf("s4 recorded over the stale vector: fingerprints %v", f)
	}
	if f["a"] != "a2" || f["b"] != "b4" {
		t.Fatalf("fingerprints = %v, want the next diff to see a2 and b4", f)
	}
}

// TestMemIndex_StaleApplyThatLeavesNothingKeepsNothing: with no tags and no
// state to stand for, the space is simply unknown.
func TestMemIndex_StaleApplyThatLeavesNothingKeepsNothing(t *testing.T) {
	m := tagging.NewMemIndex(0)
	apply(t, m, "i", "p", "s1", []tagging.TagVector{tv("a", 3)}, nil)
	applyFrom(t, m, "s0", "s2", []string{"a"})
	if _, _, ok := fps(t, m, "i", "p"); ok {
		t.Fatal("a stale Apply that emptied the space left it known")
	}
}

// TestMemIndex_ReadsDoNotSeeLaterApplies: what State and Candidates hand out is
// shared, not copied, so an Apply must replace it rather than change it.
func TestMemIndex_ReadsDoNotSeeLaterApplies(t *testing.T) {
	m := tagging.NewMemIndex(0)
	apply(t, m, "i", "p", "s1", []tagging.TagVector{tv("a", 3), tv("b", 3)}, nil)
	_, before, _ := fps(t, m, "i", "p")
	tags, err := m.Candidates(context.Background(), "i", "p", nil)
	if err != nil {
		t.Fatal(err)
	}
	apply(t, m, "i", "p", "s2", []tagging.TagVector{tv("0", 3)}, []string{"b"})
	if len(before) != 2 || before["b"] == "" || before["0"] != "" {
		t.Errorf("fingerprints read before the Apply changed: %v", before)
	}
	if !eq(ids(tags), []string{"a", "b"}) {
		t.Errorf("candidates read before the Apply changed: %v", ids(tags))
	}
	got, err := m.Candidates(context.Background(), "i", "p", nil)
	if err != nil || !eq(ids(got), []string{"0", "a"}) {
		t.Errorf("Candidates after the Apply = %v, %v; want [0 a] by id", ids(got), err)
	}
}

// BenchmarkMemIndex_UnchangedSuggest is what a Suggest at an unchanged revision
// asks of the index, under its one mutex, at 10,000 tags: both are a lookup,
// with no sort and no copy of the fingerprints.
func BenchmarkMemIndex_UnchangedSuggest(b *testing.B) {
	ctx := context.Background()
	m := tagging.NewMemIndex(0)
	put := make([]tagging.TagVector, 10000)
	for i := range put {
		put[i] = tagging.TagVector{ID: "tag-" + strconv.Itoa(i), Name: "group / tag " + strconv.Itoa(i), Fingerprint: "0123456789abcdef0123456789abcdef", Vec: make([]float32, 1024), Cutoff: 0.5}
		put[i].Vec[i%1024] = 1
	}
	if err := m.Apply(ctx, "i", "p", "", "s", put, nil); err != nil {
		b.Fatal(err)
	}
	doc := make([]float32, 1024)
	doc[3] = 1
	b.Run("State", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			_, _, _, _ = m.State(ctx, "i", "p")
		}
	})
	b.Run("Candidates", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			_, _ = m.Candidates(ctx, "i", "p", doc)
		}
	})
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
	if err := m.Apply(ctx, "", "p", "", "s", nil, nil); err == nil {
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
				from, _, _, _ := m.State(ctx, "i", part)
				_ = m.Apply(ctx, "i", part, from, "s", []tagging.TagVector{tv(fmt.Sprint(n%5), 100)}, []string{fmt.Sprint((n + 1) % 5)})
				_, _, _, _ = m.State(ctx, "i", part)
				_, _ = m.Candidates(ctx, "i", part, nil)
			}
		}()
	}
	wg.Wait()
}

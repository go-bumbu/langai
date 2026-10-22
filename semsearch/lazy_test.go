package semsearch

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
)

// TestLazyEmbedder_LoadsOnceAndCachesError uses a stub loader so this test
// never downloads a real model: it asserts Dimensions never triggers a load,
// a failing load is attempted exactly once, and its error is returned
// (cached, not retried) by every later Embed call.
func TestLazyEmbedder_LoadsOnceAndCachesError(t *testing.T) {
	calls := 0
	l := &LazyEmbedder{dims: 4, load: func() (Embedder, error) {
		calls++
		return nil, errors.New("boom")
	}}
	if l.Dimensions() != 4 {
		t.Fatal("dims must not require load")
	}
	_, err1 := l.Embed(context.Background(), RoleDocument, []string{"x"})
	_, err2 := l.Embed(context.Background(), RoleDocument, []string{"y"})
	if err1 == nil || err2 == nil {
		t.Fatal("expected cached error")
	}
	if calls != 1 {
		t.Fatalf("load attempted %d times, want 1", calls)
	}
}

// TestLazyEmbedder_DimMismatchCached covers the other error path: a load
// that succeeds but returns an embedder whose actual dimensionality disagrees
// with the configured one. That must also become a cached error rather than
// something retried on every call.
func TestLazyEmbedder_DimMismatchCached(t *testing.T) {
	calls := 0
	l := &LazyEmbedder{dims: 4, load: func() (Embedder, error) {
		calls++
		return FakeEmbedder{Dims: 8}, nil
	}}
	_, err1 := l.Embed(context.Background(), RoleDocument, []string{"x"})
	if err1 == nil {
		t.Fatal("expected dim-mismatch error")
	}
	_, err2 := l.Embed(context.Background(), RoleDocument, []string{"y"})
	if err2 == nil {
		t.Fatal("expected cached dim-mismatch error")
	}
	if calls != 1 {
		t.Fatalf("load attempted %d times, want 1", calls)
	}
}

// TestNewLazyEmbedder_DoesNotLoadOnConstruction is the guard behind the
// "lazy" in the name: building one must not call the loader, so a server can
// construct it at startup without paying for (or failing on) the downloads.
func TestNewLazyEmbedder_DoesNotLoadOnConstruction(t *testing.T) {
	l := NewLazyEmbedder(func() (Embedder, error) {
		t.Fatal("load must not run until the first Embed")
		return nil, nil
	}, 384)
	if l.Dimensions() != 384 {
		t.Fatalf("Dimensions() = %d, want the configured 384 without loading", l.Dimensions())
	}
	if attempted, err := l.State(); attempted || err != nil {
		t.Fatalf("construction loaded the model: attempted=%v err=%v", attempted, err)
	}
}

// TestLazyEmbedder_LoadSharesOnceWithEmbed: Load is the preload entry point
// (a server loading its model at startup). It must run the very same
// once-guarded load Embed uses, so a preload followed by the first request
// -- or the two racing -- loads the model exactly once, and a repeated Load
// is a free no-op.
func TestLazyEmbedder_LoadSharesOnceWithEmbed(t *testing.T) {
	calls := 0
	l := NewLazyEmbedder(func() (Embedder, error) {
		calls++
		return FakeEmbedder{Dims: 4}, nil
	}, 4)
	if err := l.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if attempted, err := l.State(); !attempted || err != nil {
		t.Fatalf("State() after Load = (%v, %v), want (true, nil)", attempted, err)
	}
	vecs, err := l.Embed(context.Background(), RoleDocument, []string{"x"})
	if err != nil || len(vecs) != 1 || len(vecs[0]) != 4 {
		t.Fatalf("Embed after Load = (%v, %v), want one 4-dim vector", vecs, err)
	}
	if err := l.Load(); err != nil {
		t.Fatalf("second Load: %v", err)
	}
	if calls != 1 {
		t.Fatalf("load attempted %d times, want 1", calls)
	}
}

// TestLazyEmbedder_LoadReturnsAndCachesError: a preload that fails must say
// so (the server logs it) and leave the same cached error for Embed and
// State, never a retry.
func TestLazyEmbedder_LoadReturnsAndCachesError(t *testing.T) {
	calls := 0
	l := NewLazyEmbedder(func() (Embedder, error) {
		calls++
		return nil, errors.New("boom")
	}, 4)
	if err := l.Load(); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("Load() = %v, want the loader's error", err)
	}
	if _, err := l.Embed(context.Background(), RoleDocument, []string{"x"}); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("Embed after a failed Load = %v, want the cached error", err)
	}
	if attempted, err := l.State(); !attempted || err == nil {
		t.Fatalf("State() = (%v, %v), want (true, boom)", attempted, err)
	}
	if calls != 1 {
		t.Fatalf("load attempted %d times, want 1", calls)
	}
}

// roleRecorder wraps an Embedder and records each call's role.
type roleRecorder struct {
	inner Embedder
	roles []Role
}

func (r *roleRecorder) Dimensions() int { return r.inner.Dimensions() }

func (r *roleRecorder) Embed(ctx context.Context, role Role, texts []string) ([][]float32, error) {
	r.roles = append(r.roles, role)
	return r.inner.Embed(ctx, role, texts)
}

// TestLazyEmbedder_ForwardsRole: Embed hands the caller's role to the loaded
// embedder untouched. An asymmetric model prefixes queries and documents
// differently, so a LazyEmbedder that fixed the role (or dropped it) would
// silently skew every search made through it.
func TestLazyEmbedder_ForwardsRole(t *testing.T) {
	rec := &roleRecorder{inner: FakeEmbedder{Dims: 4}}
	l := NewLazyEmbedder(func() (Embedder, error) { return rec, nil }, 4)
	ctx := context.Background()
	if _, err := l.Embed(ctx, RoleQuery, []string{"x"}); err != nil {
		t.Fatalf("Embed as query: %v", err)
	}
	if _, err := l.Embed(ctx, RoleDocument, []string{"x"}); err != nil {
		t.Fatalf("Embed as document: %v", err)
	}
	want := []Role{RoleQuery, RoleDocument}
	if !slices.Equal(rec.roles, want) {
		t.Fatalf("the loaded embedder saw roles %v, want %v", rec.roles, want)
	}
}

// TestLazyEmbedder_StateExposesCachedError: a health endpoint reads State,
// so a failed load must be visible there, not only as failed requests.
func TestLazyEmbedder_StateExposesCachedError(t *testing.T) {
	l := NewLazyEmbedder(func() (Embedder, error) { return nil, errors.New("boom") }, 4)
	if attempted, _ := l.State(); attempted {
		t.Fatal("State reports an attempt before any Embed call")
	}
	_, _ = l.Embed(context.Background(), RoleDocument, []string{"x"})
	attempted, err := l.State()
	if !attempted || err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("State() = (%v, %v), want (true, boom)", attempted, err)
	}
}

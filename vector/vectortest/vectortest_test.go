package vectortest_test

import (
	"testing"

	"github.com/go-bumbu/sense/vector"
	"github.com/go-bumbu/sense/vector/vectortest"
)

// TestRun_MemStore runs the suite against the reference store, which is the
// package's own test: it is what lets `make coverage` measure vectortest.
func TestRun_MemStore(t *testing.T) {
	vectortest.Run(t, func(*testing.T) vector.Store { return vector.NewMemStore() })
	vectortest.RunShared(t, func(_ *testing.T, n int) []vector.Store {
		s := vector.NewMemStore()
		stores := make([]vector.Store, n)
		for i := range stores {
			stores[i] = s
		}
		return stores
	})
}

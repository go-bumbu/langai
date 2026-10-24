package storetest_test

import (
	"testing"

	"github.com/go-bumbu/langai/autotag"
	"github.com/go-bumbu/langai/autotag/storetest"
)

// TestRun_MemStore runs the suite against the reference store, which is the
// package's own test: it is what lets `make coverage` measure storetest.
func TestRun_MemStore(t *testing.T) {
	storetest.Run(t, func(*testing.T) autotag.Store { return autotag.NewMemStore() })
}

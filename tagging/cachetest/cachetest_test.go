package cachetest_test

import (
	"testing"

	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/tagging/cachetest"
)

// TestRun_MemCache runs the suite against the reference cache, which is the
// package's own test: it is what lets `make coverage` measure cachetest.
func TestRun_MemCache(t *testing.T) {
	cachetest.Run(t, func(*testing.T) tagging.EmbeddingCache { return tagging.NewMemCache() })
}

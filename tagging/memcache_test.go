package tagging_test

import (
	"testing"

	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/tagging/cachetest"
)

func TestMemCache(t *testing.T) {
	cachetest.Run(t, func(*testing.T) tagging.EmbeddingCache { return tagging.NewMemCache() })
}

package tagging_test

import (
	"testing"
	"time"

	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/tagging/cachetest"
)

func TestMemCache(t *testing.T) {
	cachetest.Run(t, func(_ *testing.T, now func() time.Time) tagging.EmbeddingCache {
		return tagging.NewMemCacheWithClock(now)
	})
}

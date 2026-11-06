package tagging

import (
	"context"
	"crypto/sha256"
	"time"

	"github.com/go-bumbu/langai"
)

// CacheKey identifies one embedding: the model and role it was made under and
// the SHA-256 of the text.
type CacheKey struct {
	Model string
	Role  langai.Role
	Hash  [32]byte
}

// KeyFor returns the key of text embedded under model in role. The model and
// role are part of the key because the same text embeds to a different vector
// under each.
func KeyFor(model string, role langai.Role, text string) CacheKey {
	return CacheKey{Model: model, Role: role, Hash: sha256.Sum256([]byte(text))}
}

// EmbeddingCache keeps text embeddings per index. It is derived data: losing
// it costs embedding time, never correctness.
type EmbeddingCache interface {
	// Get returns the vectors of the keys it holds; missing keys are absent
	// from the map. It refreshes the last-used time of an entry older than
	// CacheRefreshAfter.
	Get(ctx context.Context, index string, keys []CacheKey) (map[CacheKey][]float32, error)
	// Put stores or replaces the vectors, last used now. An empty vector is an error.
	Put(ctx context.Context, index string, vecs map[CacheKey][]float32) error
	// DeleteIndex removes every entry of index.
	DeleteIndex(ctx context.Context, index string) error
	// Prune removes the entries, of every index, last used before unusedSince,
	// and returns how many.
	Prune(ctx context.Context, unusedSince time.Time) (int64, error)
}

// CacheRefreshAfter is how stale a last-used time gets before a Get rewrites
// it, so reads do not turn into writes.
const CacheRefreshAfter = 24 * time.Hour

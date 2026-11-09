package tagging

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/go-bumbu/langai"
)

// errEmptyIndex is refused by every cache, index and Tagger call: an index is
// the hard tenant boundary, and an empty one names no tenant.
var errEmptyIndex = errors.New("tagging: empty index")

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

// checkIndex refuses an index no cache takes: the empty one, and one a
// database cannot store as text (see checkName).
func checkIndex(index string) error {
	if index == "" {
		return errEmptyIndex
	}
	return checkName("index", index)
}

// checkName refuses an index or a model with a NUL byte or invalid UTF-8.
// Postgres stores neither in a text column while SQLite and a Go map take
// both, so every backend refuses them and an application meets the error on
// the first one it tries, not on the day it moves to another.
func checkName(what, name string) error {
	if !utf8.ValidString(name) || strings.ContainsRune(name, 0) {
		return fmt.Errorf("tagging: %s %q holds a NUL byte or is not valid UTF-8", what, name)
	}
	return nil
}

// EmbeddingCache keeps text embeddings per index. It is derived data: losing
// it costs embedding time, never correctness. Every call refuses an empty
// index, and an index or a key's model that holds a NUL byte or is not valid
// UTF-8.
type EmbeddingCache interface {
	// Get returns the vectors of the keys it holds; missing keys are absent
	// from the map, and a key listed twice counts once. It sets the last-used
	// time of an entry to now when that entry was last used CacheRefreshAfter
	// ago or longer, and leaves a more recent one alone; a cache that keeps
	// its times in seconds compares them in seconds. A cache that read the
	// vectors but could not write that refresh may return them together with
	// an error wrapping ErrCacheWrite; with any other error it returns no
	// vectors.
	Get(ctx context.Context, index string, keys []CacheKey) (map[CacheKey][]float32, error)
	// Put stores or replaces the vectors, last used now. An empty vector is an error.
	Put(ctx context.Context, index string, vecs map[CacheKey][]float32) error
	// Touch sets the last-used time of the keys it holds to now, whatever it
	// was; keys it does not hold are ignored. It is how a Tagger marks the
	// embeddings its tags are built from as in use before a Prune (see
	// Tagger.Prune): a tag that does not change is never read again, so no Get
	// would refresh them.
	Touch(ctx context.Context, index string, keys []CacheKey) error
	// DeleteIndex removes every entry of index.
	DeleteIndex(ctx context.Context, index string) error
	// Prune removes the entries, of every index, last used before unusedSince,
	// and returns how many. An entry last used at unusedSince stays. It
	// removes what it is told to: an application prunes through Tagger.Prune,
	// which first marks what is still in use and refuses a cutoff too recent
	// for the last-used times to be trusted.
	Prune(ctx context.Context, unusedSince time.Time) (int64, error)
}

// CacheRefreshAfter is how stale a last-used time gets before a Get rewrites
// it, so reads do not turn into writes: a Get this long after the last use, or
// later, refreshes the entry, and an earlier one does not. An entry's recorded
// time may therefore be up to this much older than its last read, so a
// Tagger.Prune keeps for certain only what was read in the last (age of its
// cutoff - CacheRefreshAfter), and refuses a cutoff younger than MinPruneAge.
const CacheRefreshAfter = 24 * time.Hour

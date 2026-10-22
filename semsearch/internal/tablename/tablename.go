// Package tablename derives the physical table names of the SQL stores
// (sqlitevec, pgvec), which keep one table set per semsearch index (e.g. a
// tenant) and generation. SQL identifiers are case-insensitive, so
// interpolating index directly would collide "A" and "a" into the same tables
// (a leak across indexes) and would reject indexes with characters unsafe in a
// bare identifier (e.g. a hyphenated UUID). Hashing index into a fixed-width
// hex token avoids both problems: distinct index strings always map to
// distinct tables, and the token is injection-safe.
//
// A generation gives every index a further table set, so the vectors of a
// second embedding model can be built next to the first's before an
// application switches over. The empty generation is the table set indexes
// had before generations existed.
package tablename

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
)

// generationToken is what a non-empty generation must be. It reaches SQL
// unhashed, so it is a bare lowercase token (SQL folds identifier case, so
// "G2" would land on "g2"'s tables), without underscores (so one
// generation's name prefix never covers another's tables: "a" would match
// "a_b"), and short enough to keep the longest derived name, pgvec's legacy
// HNSW index, well within Postgres's 63-byte identifier limit.
var generationToken = regexp.MustCompile(`^[a-z0-9]{1,12}$`)

// CheckGeneration reports whether generation can name a table set: "" or 1-12
// lowercase letters and digits. The stores call it when they are built, so a
// bad generation fails there rather than on the first write.
func CheckGeneration(generation string) error {
	if generation != "" && !generationToken.MatchString(generation) {
		return fmt.Errorf("invalid generation %q: use 1-12 lowercase letters and digits", generation)
	}
	return nil
}

// For returns the name of index's table with the given suffix in generation.
// The empty generation keeps the names existing databases hold their tables
// under, so its derivation must not change; any other generation puts its
// token ahead of the index hash: ss_<generation>_<hash>_<suffix>.
func For(generation, index, suffix string) (string, error) {
	if err := CheckGeneration(generation); err != nil {
		return "", err
	}
	if index == "" {
		return "", errors.New("empty index")
	}
	sum := sha256.Sum256([]byte(index))
	token := hex.EncodeToString(sum[:])[:16]
	if generation == "" {
		return "ss_" + token + "_" + suffix, nil
	}
	return "ss_" + generation + "_" + token + "_" + suffix, nil
}

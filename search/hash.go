package search

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/go-bumbu/langai"
)

// Fingerprinter is the optional interface of a Chunker, a media.Sampler or
// a langai.VideoEmbedder whose output depends on its settings: Fingerprint
// describes them, the same string for the same output and another one
// whenever a change of setting (or of algorithm) would change the chunks,
// frames or segments. The indexes fold it into the hash they store with
// every item, so that after such a change each item is indexed again
// instead of being found unchanged. One without it is described by its Go
// type name: swapping it for another type re-indexes, changing its settings
// does not.
type Fingerprinter interface {
	Fingerprint() string
}

// fingerprintOf is v's Fingerprint, or its Go type name.
func fingerprintOf(v any) string {
	if f, ok := v.(Fingerprinter); ok {
		return f.Fingerprint()
	}
	return fmt.Sprintf("%T", v)
}

// modelOf returns the model e names, "" when it names none.
func modelOf(e any) string {
	if id, ok := e.(langai.Identified); ok {
		return id.Model()
	}
	return ""
}

// settings joins what shapes an index's vectors into the string stamp
// folds into every stored hash.
func settings(parts ...any) string {
	s := make([]string, len(parts))
	for i, p := range parts {
		s[i] = fmt.Sprint(p)
	}
	return strings.Join(s, "\x00")
}

// sanitize makes text storable alike in every backend: invalid UTF-8 becomes
// U+FFFD and NUL bytes are dropped, both of which a Postgres TEXT column
// refuses while SQLite takes them.
func sanitize(text string) string {
	return strings.ReplaceAll(strings.ToValidUTF8(text, "�"), "\x00", "")
}

// stamp is the hash an index stores for an item: the SHA-256, hex, of the
// settings that shaped its vectors and of its content.
func stamp(settings string, content []byte) string {
	h := sha256.New()
	h.Write([]byte(settings))
	h.Write([]byte{0})
	h.Write(content)
	return hex.EncodeToString(h.Sum(nil))
}

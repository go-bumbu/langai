// Package tagging suggests tags for texts in two ways: a Tagger chooses among
// the tags an application already has, and a Generator writes new ones.
//
// A Tagger suggests tags for items, such as bookmarks or notes, from their
// embeddings, and learns from the items a person tags by hand. The
// application stays the owner of its tags, items and links: it asks for
// suggestions for a text, and the Tagger reads what it learns from when it
// needs it.
//
// Each tag's vector blends an embedding of its name (a prior) with the mean
// of the items tagged with it by hand, its members (see blend), so a
// brand-new tag with no members yet still tags by its name, and a tag that
// accumulates members gradually tags by example instead. An item gets every
// tag whose score reaches that tag's own cutoff (see tagCutoff), with no limit
// on how many, except that tag names are paths of a tree ("dev / go" under
// "dev") and of the tags on one path only the one the item clears the cutoff
// of by most is picked (see TaggerConfig.AllPathLevels).
//
// A space is an (index, partition) pair: index is the hard tenant boundary,
// and no data of one index is ever read on behalf of another; partition is
// the space within it. The application implements a Catalog with two
// read-only lookups, the tags of a space (each one's id, name and revision)
// and the member texts of some of its tags, in one batched call, and passes
// the space's revision on every Suggest: an opaque string, compared for
// equality, that changes whenever a tag, a link or a tagged item of the space
// does. The application's writes never call the Tagger.
//
// The Tagger keeps only derived data it can always rebuild. An
// EmbeddingCache (MemCache, or sqlitecache and pgcache for a database) holds
// every embedding, by index, model, role and the hash of the text, so an
// edited text or another model simply misses and is embedded again, and
// unused rows go with Prune. A TagIndex (MemIndex by default, an LRU of the
// spaces under a memory budget) holds each space's blended tag vectors and
// cutoffs, with the revision they were built at. On a Suggest at the revision
// the index holds, the catalog is not called: the text's vector comes from
// the cache and is scored against the index's candidates. At another
// revision, the Tagger reads the catalog's tags, rebuilds from the cache only
// the tags whose revision, name, model or settings changed, drops the ones
// gone, and records the revision, once for every caller of that revision. A
// rebuild that fails returns an error wrapping ErrNotRebuilt and leaves the
// index as it was, so the next call tries again. Losing the cache or the
// index costs time, never correctness.
//
// A Generator starts from nothing: a language model (a langai.Completer)
// proposes tags for the text, an embedding model (a langai.Embedder) scores
// each one by how close it sits to the text, and Generate returns the ones
// that reach the caller's cutoff, best first.
//
// The model supplies the words and the embedder the score. A model's own
// confidence is poorly calibrated and moves from run to run, while the cosine
// between a text and a tag is on a fixed scale: one cutoff means the same on
// every text, and a tag the model invented, or one that has little to do with
// the text, falls below it. The text goes to the embedder as
// langai.RoleQuery and the tags as langai.RoleDocument, the roles a Tagger
// gives an item and a tag name, so the scores sit on the scale of
// TaggerConfig.NameMinScore.
//
// The production Completer is the one in provider/yzma, a GGUF chat model run
// locally through llama.cpp. This package imports neither it nor llama.cpp, so
// a binary that uses a Generator with a Completer of its own links neither. A
// Generator keeps no state and is safe for concurrent use when its Completer
// and Embedder are.
package tagging

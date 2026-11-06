// Package tagging suggests tags for texts in two ways: a Tagger chooses among
// the tags an application already has, and a Generator writes new ones.
//
// A Tagger suggests tags for items, such as bookmarks or notes, from their
// embeddings, and learns from the items a person tags by hand. It owns what it
// learns from: an app reports its tags and its items, by id and text, and the
// tags a person put on items by hand, and asks for suggestions.
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
// A Store keeps the tags, the items with their embeddings and the hand-made
// links of every scope. Every change rebuilds the vectors of the tags it
// affects right away. A tag's Version goes up with every change to what its
// vectors depend on, and a rebuild stores its result only while the tag is at
// the Version it read, so a rebuild never overwrites a change that landed
// meanwhile. A tag left stale that way, by a rebuild that failed, or by a
// switch to another embedding model or to other scoring settings (see
// TaggerConfig) is rebuilt by the next Suggest, or by Rebuild.
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

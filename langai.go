// Package langai holds what the language-AI packages of this module share:
// the ports a model runtime implements, Embedder for embedding models and
// Completer for chat models, and the Role an Embedder is asked to embed a text
// as. It imports no runtime and no store.
//
// The capabilities live in subpackages: search answers free-text queries over
// documents, and tagging suggests tags for a text, from the tags an application
// already has (Tagger) or written from nothing (Generator). The implementations
// sit in groups by the port they implement: provider/yzma runs GGUF models
// locally through llama.cpp, and vector/sqlitevec and vector/pgvec store
// search vectors. LazyEmbedder defers loading a model to its first use,
// langaitest has the fakes for tests, and eval measures an Embedder's search
// and tagging quality over a labelled dataset.
package langai

import (
	"context"
	"fmt"
)

// Role tells an Embedder what a text is for. Asymmetric models are trained
// with different prefixes for the text being searched with and the texts it
// is searched against; symmetric models ignore the role.
type Role int

const (
	// RoleDocument marks text that is being indexed.
	RoleDocument Role = iota
	// RoleQuery marks text that is being searched with.
	RoleQuery
)

// String implements fmt.Stringer for log and error messages.
func (r Role) String() string {
	switch r {
	case RoleDocument:
		return "document"
	case RoleQuery:
		return "query"
	}
	return fmt.Sprintf("Role(%d)", int(r))
}

// Embedder turns texts into embedding vectors. All vectors returned must have
// exactly Dimensions() entries; the result slice matches the input order. The
// role lets an asymmetric model apply its query or document prefix.
type Embedder interface {
	Embed(ctx context.Context, role Role, texts []string) ([][]float32, error)
	Dimensions() int
}

// Completer is a chat model: it answers one prompt, a system message that sets
// the task and a user message that carries the text, with the text of its
// reply. provider/yzma's Completer is the local implementation. The user
// message comes from outside, so an implementation must treat it as data, not
// as markup for its own prompt format.
type Completer interface {
	Complete(ctx context.Context, system, user string) (string, error)
}

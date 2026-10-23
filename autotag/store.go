package autotag

import (
	"context"
	"errors"
)

// Errors for an item or a tag the scope does not hold. Test for them with
// errors.Is, because a Store may wrap them.
var (
	ErrUnknownTag  = errors.New("autotag: unknown tag")
	ErrUnknownItem = errors.New("autotag: unknown item")
)

// TagRecord is a tag as a Store holds it.
type TagRecord struct {
	ID      string
	Name    string
	NameVec []float32 // Name embedded under Model; nil until built, and cleared by a rename
	Vec     []float32 // the blended vector; nil until built
	Cutoff  float64
	Model   string // the model NameVec and Vec come from
	Version int64  // 1 when created; +1 on every change to what Vec depends on
	Built   int64  // the Version Vec was built from; 0 until built
}

// TagVectors are the vectors SetTagVectors stores for a tag, with the cutoff
// and the model they were built for.
type TagVectors struct {
	NameVec, Vec []float32
	Cutoff       float64
	Model        string
}

// ItemRecord is an item as a Store holds it: its text and the embedding of
// that text under Model.
type ItemRecord struct {
	ID, Text string
	Vec      []float32
	Model    string
}

// Member is an item linked to a tag, as Members reports it: the item's id,
// vector and model.
type Member struct {
	ItemID string
	Vec    []float32
	Model  string
}

// Store persists the tags, items and hand-made links of every scope. index is
// the hard isolation boundary (a tenant), partition a coordinate within it (a
// space); no scope ever sees another's data. Every method fails on an empty
// index. Vectors round-trip exactly; nil stays nil. Slices a Store is given or
// returns are never shared with what it keeps.
type Store interface {
	Tag(ctx context.Context, index, partition, tagID string) (TagRecord, bool, error)
	Tags(ctx context.Context, index, partition string) ([]TagRecord, error) // ordered by ID
	// PutTag creates the tag (Version 1, Built 0) or renames it (NameVec = nil,
	// Version+1); the current name changes nothing and reports false.
	PutTag(ctx context.Context, index, partition, tagID, name string) (bool, error)
	// SetTagVectors stores v and sets Built = version, only if the tag still
	// has that Version; otherwise (or when the tag is gone) it stores nothing
	// and reports false.
	SetTagVectors(ctx context.Context, index, partition, tagID string, version int64, v TagVectors) (bool, error)
	DeleteTag(ctx context.Context, index, partition, tagID string) error // and its links; unknown: no-op
	Item(ctx context.Context, index, partition, itemID string) (ItemRecord, bool, error)
	// PutItem creates or replaces the item, bumps the Version of every tag
	// linked to it and returns those tags' ids, ordered.
	PutItem(ctx context.Context, index, partition string, item ItemRecord) ([]string, error)
	// DeleteItem deletes the item and its links, bumps the Version of every tag
	// it was linked to and returns those ids, ordered; unknown: no-op.
	DeleteItem(ctx context.Context, index, partition, itemID string) ([]string, error)
	// Link links item and tag and bumps the tag's Version; an existing link
	// changes nothing and reports false. Unknown item: ErrUnknownItem (checked
	// first); unknown tag: ErrUnknownTag.
	Link(ctx context.Context, index, partition, itemID, tagID string) (bool, error)
	// Unlink removes the link and bumps the tag's Version; no link: false.
	Unlink(ctx context.Context, index, partition, itemID, tagID string) (bool, error)
	Members(ctx context.Context, index, partition, tagID string) ([]Member, error) // ordered by ItemID
}

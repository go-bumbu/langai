// Package autotag assigns tags to documents from their embeddings. Each
// tag's vector blends an embedding of its name (a prior) with the centroid of
// the documents tagged with it by hand, its members (see blend), so a
// brand-new tag with no members yet still tags by its name, and a tag that
// accumulates members gradually tags by example instead. A document gets
// every tag whose score reaches that tag's own cutoff (see tagCutoff), with
// no limit on how many.
//
// The tag vectors live in a semsearch.Store, one per tag, under the
// coordinates a Scope names and Config.Collection. What a Tagger caches and
// learns from beyond that, the tags' name embeddings and their members,
// belongs to the caller, who hands it over through a TagStore.
package autotag

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"slices"
	"strconv"

	"github.com/go-bumbu/langai/semsearch"
)

// Defaults for the zero values of Config. The two minimums are the default
// yzma model's values on the semsearch golden set (see TestGoldenTagging);
// that model's scores sit in a narrow band, so another model needs them
// measured again.
const (
	DefaultNameMinScore   = 0.46
	DefaultMemberMinScore = 0.70
	DefaultSmoothingK     = 3
	DefaultCollection     = "tags"
)

// Config tunes a Tagger. Model is the identity of the embedding model: it is
// part of every tag vector's fingerprint, so switching models recomputes them
// (and a cache of document embeddings the caller keeps must be keyed by it
// too). NameMinScore and MemberMinScore are the minimum cosine scores against
// a tag's name and against the centroid of its members. The two score on
// different scales, so each has its own minimum, mixed per tag by the blend
// weight (see tagCutoff); 0 defaults them to DefaultNameMinScore and
// DefaultMemberMinScore. SmoothingK is the Bayesian smoothing constant in
// blend: a tag's members weigh as much as its name at SmoothingK members, and
// outweigh it beyond that (<=0 defaults to DefaultSmoothingK). Collection is
// the semsearch collection the tag vectors are stored under ("" defaults to
// DefaultCollection). There is no limit on how many tags a document gets.
type Config struct {
	Model          string
	NameMinScore   float64
	MemberMinScore float64
	SmoothingK     float64
	Collection     string
}

// Scope locates one set of tags in the vector store: Index is the store's
// hard isolation boundary (a tenant, say) and Partition a further coordinate
// within it (a space of that tenant). Scopes never see each other's tags.
type Scope struct {
	Index     string
	Partition string
}

// Tag is one tag of a scope as the caller knows it. ID is unique within the
// scope and names the tag's vector in the store. Name is the text embedded as
// the tag's name prior: a full path such as "colors / warm / red" carries
// more than the leaf alone. NameHash changes whenever Name does, so a cached
// name embedding is reused only while it is current.
type Tag struct {
	ID       string
	Name     string
	NameHash string
}

// TagStore is the caller's storage behind one scope's tags: the cache of
// their name embeddings, and their members.
type TagStore interface {
	// NameVector returns the tag's cached name embedding and the NameHash it
	// was computed for; a nil vector means none is cached.
	NameVector(ctx context.Context, tagID string) ([]float32, string, error)
	// SetNameVector caches the tag's name embedding, computed for nameHash.
	SetNameVector(ctx context.Context, tagID string, vec []float32, nameHash string) error
	// Members returns the embeddings of the documents tagged with the tag by
	// hand, each embedded as Tagger.Embed embeds a document, and a hash of
	// each one's content, the two slices parallel: a changed set of hashes is
	// how a Tagger tells that the tag's members changed.
	Members(ctx context.Context, tagID string) ([][]float32, []string, error)
}

// Tagger computes hybrid tag vectors and assigns tags to documents. It keeps
// no state beyond its configuration, so one Tagger serves every scope.
type Tagger struct {
	emb   semsearch.Embedder
	store semsearch.Store
	cfg   Config
}

// New returns a Tagger using emb to embed text and store to hold the tag
// vectors. The zero values of cfg's fields take the defaults Config
// documents.
func New(emb semsearch.Embedder, store semsearch.Store, cfg Config) *Tagger {
	if cfg.NameMinScore == 0 {
		cfg.NameMinScore = DefaultNameMinScore
	}
	if cfg.MemberMinScore == 0 {
		cfg.MemberMinScore = DefaultMemberMinScore
	}
	if cfg.SmoothingK <= 0 {
		cfg.SmoothingK = DefaultSmoothingK
	}
	if cfg.Collection == "" {
		cfg.Collection = DefaultCollection
	}
	return &Tagger{emb: emb, store: store, cfg: cfg}
}

// Embed embeds a document's text the way Assign scores it, and the way
// TagStore.Members hands members over: as a query, since tag vectors are
// what a document is searched against.
func (t *Tagger) Embed(ctx context.Context, text string) ([]float32, error) {
	return t.embedOne(ctx, semsearch.RoleQuery, text)
}

// Refresh recomputes and re-stores one tag's hybrid vector if anything it
// depends on changed: the tag's name (a NameHash mismatch), or its set of
// members (their hashes, folded into blendHash below). It is cheap to call
// eagerly since both the name embedding and the final blended vector are
// skipped when their respective inputs are unchanged. The name is embedded as
// a document: tag vectors are what a document's embedding (a query) is
// searched against. Refresh also returns the tag's cutoff (see tagCutoff),
// computed even when nothing changed.
func (t *Tagger) Refresh(ctx context.Context, scope Scope, src TagStore, tag Tag) (float64, error) {
	// name embedding (cached by name hash)
	nameVec, cachedHash, err := src.NameVector(ctx, tag.ID)
	if err != nil {
		return 0, err
	}
	if nameVec == nil || cachedHash != tag.NameHash {
		nameVec, err = t.embedOne(ctx, semsearch.RoleDocument, tag.Name)
		if err != nil {
			return 0, err
		}
		if err := src.SetNameVector(ctx, tag.ID, nameVec, tag.NameHash); err != nil {
			return 0, err
		}
	}
	memberVecs, memberHashes, err := src.Members(ctx, tag.ID)
	if err != nil {
		return 0, err
	}
	// The cutoff is computed before the unchanged check below so a skipped
	// refresh reports it too.
	cutoff := tagCutoff(nameVec, memberVecs, t.cfg.SmoothingK, t.cfg.NameMinScore, t.cfg.MemberMinScore)
	blendHash := hashInputs(tag.Name, memberHashes, t.cfg.Model, t.cfg.SmoothingK)
	if h, err := t.store.ContentHash(ctx, scope.Index, scope.Partition, t.cfg.Collection, tag.ID); err == nil && h == blendHash {
		return cutoff, nil // unchanged
	}
	vec := blend(nameVec, memberVecs, t.cfg.SmoothingK)
	if err := t.store.Upsert(ctx, scope.Index, scope.Partition, t.cfg.Collection, tag.ID, blendHash,
		[]semsearch.Chunk{{Seq: 0, Text: tag.Name, Vector: vec}}); err != nil {
		return 0, err
	}
	return cutoff, nil
}

// Assign returns the ids of the tags whose score for vec, a document's
// embedding (see Embed), reaches that tag's own cutoff (see tagCutoff), in
// the order the search ranked them and with no limit on how many. It first
// refreshes every tag in tags (few per scope, and cheap thanks to the
// caches), so tags must be the scope's current tags, all of them: a vector in
// the scope whose tag is missing from tags is taken for a deleted tag's
// leftover and deleted (see searchTags). With no tags nothing is searched and
// no id is returned.
func (t *Tagger) Assign(ctx context.Context, scope Scope, src TagStore, tags []Tag, vec []float32) ([]string, error) {
	if len(tags) == 0 {
		return nil, nil
	}
	// each tag's cutoff, under the id its vector is stored with, so a search
	// hit finds its cutoff directly
	cuts := make(map[string]float64, len(tags))
	for _, tg := range tags {
		cut, err := t.Refresh(ctx, scope, src, tg)
		if err != nil {
			return nil, err
		}
		cuts[tg.ID] = cut
	}
	hits, err := t.searchTags(ctx, scope, vec, cuts)
	if err != nil {
		return nil, err
	}
	return pickTags(hits, cuts), nil
}

// embedOne embeds a single text, refusing an embedder that breaks its contract
// by answering with no vector, or several, rather than panicking on the
// missing one.
func (t *Tagger) embedOne(ctx context.Context, role semsearch.Role, text string) ([]float32, error) {
	vs, err := t.emb.Embed(ctx, role, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vs) != 1 {
		return nil, fmt.Errorf("autotag: embedder returned %d vectors for one %s", len(vs), role)
	}
	return vs[0], nil
}

// searchTags returns the hits for the scope's current tags (the keys of
// cuts). The search has one slot per tag, and a vector whose tag has since
// been deleted (a ghost) can take a slot and push a current tag out of the
// hits. So every ghost a search shows is deleted and the search is repeated,
// until a search shows no ghost that has not been deleted already; those are
// the hits returned. Each round deletes at least one new id and the store is
// finite, so the loop ends. The hits can still hold a ghost whose Delete had
// no effect, which pickTags skips like any hit that is not a current tag.
func (t *Tagger) searchTags(ctx context.Context, scope Scope, vec []float32, cuts map[string]float64) ([]semsearch.Match, error) {
	deleted := make(map[string]bool)
	for {
		hits, err := t.store.Search(ctx, scope.Index, scope.Partition, t.cfg.Collection, vec, len(cuts))
		if err != nil {
			return nil, err
		}
		fresh := 0
		for _, h := range hits {
			if _, ok := cuts[h.DocID]; ok || deleted[h.DocID] {
				continue
			}
			if err := t.store.Delete(ctx, scope.Index, scope.Partition, t.cfg.Collection, h.DocID); err != nil {
				return nil, err
			}
			deleted[h.DocID] = true
			fresh++
		}
		if fresh == 0 {
			return hits, nil
		}
	}
}

// pickTags returns the ids of the current tags whose hit reaches the tag's
// own cutoff. Each tag has its own cutoff, so a hit below one says nothing
// about the next: every hit is checked, and every tag that clears its own is
// picked, with no limit on how many. A hit that is not a current tag, such as
// a ghost whose Delete had no effect (see searchTags), is skipped. It fails
// closed: a NaN score or cutoff picks nothing.
func pickTags(hits []semsearch.Match, cuts map[string]float64) []string {
	var picks []string
	for _, h := range hits {
		cut, ok := cuts[h.DocID]
		// "not at or above" and not "below": every comparison with a NaN is false,
		// and with no cap a NaN reaching here would pick every tag
		if !ok || !(h.Score >= cut) {
			continue
		}
		picks = append(picks, h.DocID)
	}
	return picks
}

// hashInputs fingerprints everything a tag's blended vector depends on (its
// name, its members' content hashes, the embedding model, and the smoothing
// constant) so Refresh can skip recomputing the blend when none of them
// changed. The member hashes are sorted, on a copy that leaves the caller's
// slice as it was, so the fingerprint does not depend on the order the
// TagStore returns them in.
func hashInputs(name string, memberHashes []string, model string, k float64) string {
	sorted := slices.Clone(memberHashes)
	slices.Sort(sorted)
	h := sha256.New()
	h.Write([]byte(name))
	h.Write([]byte{0})
	for _, m := range sorted {
		h.Write([]byte(m))
		h.Write([]byte{0})
	}
	h.Write([]byte(model))
	h.Write([]byte(strconv.FormatFloat(k, 'f', -1, 64)))
	return hex.EncodeToString(h.Sum(nil))
}

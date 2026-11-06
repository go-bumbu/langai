package tagging

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/go-bumbu/langai"
)

var (
	errEmptyRevision = errors.New("tagging: empty revision")
	errBlankText     = errors.New("tagging: blank text")
)

// engine scores texts against the tags of a space without keeping a copy of
// the application's data. It reads the tags and hand-made links from the
// Catalog when the space's revision changes, keeps every embedding it makes
// in the EmbeddingCache, and keeps each space's blended tag vectors in the
// TagIndex, recorded under a state that names the revision and the settings
// they were built at. Everything it holds is derived, so losing the index or
// the cache costs time, never correctness. It is safe for concurrent use when
// its parts are.
type engine struct {
	emb   langai.Embedder
	cat   Catalog
	cache EmbeddingCache
	ix    TagIndex
	cfg   TaggerConfig
	// params fingerprints cfg's scoring settings (see checkConfig)
	params string
	// itemModel is what item and member vectors are cached under (see itemModel)
	itemModel string
	flight    flight
}

// newEngine returns an engine over its four parts. It fails on a missing
// part, and on a configuration NewTagger refuses (see checkConfig).
func newEngine(emb langai.Embedder, cat Catalog, cache EmbeddingCache, ix TagIndex, cfg TaggerConfig) (*engine, error) {
	switch {
	case emb == nil:
		return nil, errors.New("tagging: no embedder")
	case cat == nil:
		return nil, errors.New("tagging: no catalog")
	case cache == nil:
		return nil, errors.New("tagging: no embedding cache")
	case ix == nil:
		return nil, errors.New("tagging: no tag index")
	}
	cfg, params, err := checkConfig(cfg)
	if err != nil {
		return nil, err
	}
	return &engine{emb: emb, cat: cat, cache: cache, ix: ix, cfg: cfg, params: params, itemModel: itemModel(cfg)}, nil
}

// scores returns text's score against every tag of the space at revision,
// best first, as Scope.Scores does: it gets text's vector (from the cache, or
// embedded and cached), brings the space up to revision (see refresh), and
// scores the index's candidates. Any error returns no scores; one of a
// rebuild wraps ErrNotRebuilt.
func (e *engine) scores(ctx context.Context, index, partition, revision, text string) ([]TagScore, error) {
	if err := checkSpace(index, revision); err != nil {
		return nil, err
	}
	if strings.TrimSpace(text) == "" {
		return nil, errBlankText
	}
	vecs, err := e.itemVectors(ctx, index, []string{text})
	if err != nil {
		return nil, err
	}
	vec := vecs[text]
	if err := e.refresh(ctx, index, partition, revision); err != nil {
		return nil, err
	}
	cands, err := e.ix.Candidates(ctx, index, partition, vec)
	if err != nil {
		return nil, err
	}
	return scoreVectors(vec, cands, !e.cfg.AllPathLevels), nil
}

// warm brings the space up to revision (see refresh) and caches the vectors
// of texts, the blank ones skipped, so no later scores pays for them.
func (e *engine) warm(ctx context.Context, index, partition, revision string, texts []string) error {
	if err := checkSpace(index, revision); err != nil {
		return err
	}
	if err := e.refresh(ctx, index, partition, revision); err != nil {
		return err
	}
	_, err := e.itemVectors(ctx, index, nonBlank(texts))
	return err
}

// checkSpace refuses an empty index, which a TagIndex and an EmbeddingCache
// refuse too, and an empty revision, which could never change.
func checkSpace(index, revision string) error {
	switch {
	case index == "":
		return errEmptyIndex
	case revision == "":
		return errEmptyRevision
	}
	return nil
}

// digest is the first 16 bytes, in hex, of the SHA-256 of parts joined by
// NULs: short enough to keep per tag, long enough never to collide by chance.
func digest(parts ...string) string {
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x00")))
	return hex.EncodeToString(sum[:16])
}

// state is what the index records for a space built at revision: the model
// and the scoring settings are part of it, so an engine with other ones takes
// the space for stale even at the same revision.
func (e *engine) state(revision string) string { return digest(revision, e.cfg.Model, e.params) }

// fingerprint is what a tag's vector was built from: its revision (its
// members), its name, the model and the scoring settings. A rename changes it
// without a new revision.
func (e *engine) fingerprint(t CatalogTag) string {
	return digest(t.Revision, t.Name, e.cfg.Model, e.params)
}

// refresh brings the space up to revision unless the index records that
// state already. The rebuild runs once for every caller of one space and
// state at a time (see flight), and re-reads the state first, since another
// caller may just have applied it.
func (e *engine) refresh(ctx context.Context, index, partition, revision string) error {
	state := e.state(revision)
	cur, _, ok, err := e.ix.State(ctx, index, partition)
	switch {
	case err != nil:
		return err
	case ok && cur == state:
		return nil
	}
	return e.flight.do(ctx, index+"\x00"+partition+"\x00"+state, func(ctx context.Context) error {
		cur, held, ok, err := e.ix.State(ctx, index, partition)
		switch {
		case err != nil:
			return err
		case ok && cur == state:
			return nil
		}
		put, drop, err := e.diff(ctx, index, partition, held)
		if err != nil {
			// the index is left as it was and the state unrecorded, so the
			// next call diffs again; what was embedded stays cached
			return fmt.Errorf("%w: %w", ErrNotRebuilt, err)
		}
		return e.ix.Apply(ctx, index, partition, state, put, drop)
	})
}

// diff reads the catalog's tags and returns the vectors of the ones whose
// fingerprint is not the one held, and the ids held that the catalog no
// longer lists. It refuses a tag with an empty id, an id listed twice, a
// blank name or an empty revision, naming it.
func (e *engine) diff(ctx context.Context, index, partition string, held map[string]string) (put []TagVector, drop []string, err error) {
	tags, err := e.cat.Tags(ctx, index, partition)
	if err != nil {
		return nil, nil, err
	}
	listed := make(map[string]bool, len(tags))
	for _, t := range tags {
		switch {
		case t.ID == "":
			return nil, nil, fmt.Errorf("tagging: catalog tag %q has an empty id", t.Name)
		case listed[t.ID]:
			return nil, nil, fmt.Errorf("tagging: catalog lists tag %q twice", t.ID)
		case strings.TrimSpace(t.Name) == "":
			return nil, nil, fmt.Errorf("tagging: catalog tag %q has a blank name", t.ID)
		case t.Revision == "":
			return nil, nil, fmt.Errorf("tagging: catalog tag %q has an empty revision", t.ID)
		}
		listed[t.ID] = true
		if fp := e.fingerprint(t); held[t.ID] != fp {
			put = append(put, TagVector{ID: t.ID, Name: t.Name, Fingerprint: fp})
		}
	}
	for id := range held {
		if !listed[id] {
			drop = append(drop, id)
		}
	}
	slices.Sort(drop)
	if len(put) == 0 {
		return nil, drop, nil
	}
	if err := e.build(ctx, index, partition, put); err != nil {
		return nil, nil, err
	}
	return put, drop, nil
}

// build fills in the Vec and Cutoff of tags from their names and members, in
// one Members call. Members of ids not asked for are ignored, and blank texts
// skipped. A member whose vector is not as long as the tag's name vector is
// an error naming the tag: blend and tagCutoff index one by the other.
func (e *engine) build(ctx context.Context, index, partition string, tags []TagVector) error {
	ids, names := make([]string, len(tags)), make([]string, len(tags))
	for i, t := range tags {
		ids[i], names[i] = t.ID, t.Name
	}
	got, err := e.cat.Members(ctx, index, partition, ids)
	if err != nil {
		return err
	}
	members := make(map[string][]string, len(ids))
	var texts []string
	for _, id := range ids {
		members[id] = nonBlank(got[id])
		texts = append(texts, members[id]...)
	}
	items, err := e.itemVectors(ctx, index, texts)
	if err != nil {
		return err
	}
	nameVecs, err := e.nameVectors(ctx, index, names)
	if err != nil {
		return err
	}
	k := e.cfg.SmoothingK
	for i := range tags {
		t := &tags[i]
		nameVec := nameVecs[t.Name]
		vecs := make([][]float32, 0, len(members[t.ID]))
		for _, m := range members[t.ID] {
			v := items[m]
			if len(v) != len(nameVec) {
				return fmt.Errorf("tagging: tag %q has a member with a vector of %d dimensions and a name vector of %d", t.ID, len(v), len(nameVec))
			}
			vecs = append(vecs, v)
		}
		t.Vec = blend(nameVec, vecs, k)
		t.Cutoff = storedCutoff(tagCutoff(nameVec, vecs, k, e.cfg.NameMinScore, e.cfg.MemberMinScore))
	}
	return nil
}

// nonBlank returns the texts that are not blank, in order.
func nonBlank(texts []string) []string {
	var out []string
	for _, t := range texts {
		if strings.TrimSpace(t) != "" {
			out = append(out, t)
		}
	}
	return out
}

// itemVectors returns the query vector of each of texts, by text, as items
// and members are scored (see embedItems).
func (e *engine) itemVectors(ctx context.Context, index string, texts []string) (map[string][]float32, error) {
	key := func(text string) CacheKey { return KeyFor(e.itemModel, langai.RoleQuery, text) }
	return e.vectors(ctx, index, texts, key, e.embedItems)
}

// nameVectors returns the document vector of each of names, by name.
func (e *engine) nameVectors(ctx context.Context, index string, names []string) (map[string][]float32, error) {
	key := func(name string) CacheKey { return KeyFor(e.cfg.Model, langai.RoleDocument, name) }
	embed := func(ctx context.Context, names []string) ([][]float32, error) {
		return embedBatch(ctx, e.emb, langai.RoleDocument, names)
	}
	return e.vectors(ctx, index, names, key, embed)
}

// vectors returns the vector of each of texts, by text: the cache's, in one
// Get, where it holds one under key(text), and embed's for the rest, which
// are then cached in one Put. A text listed twice is looked up and embedded
// once.
func (e *engine) vectors(ctx context.Context, index string, texts []string, key func(string) CacheKey,
	embed func(context.Context, []string) ([][]float32, error)) (map[string][]float32, error) {
	var uniq []string
	keys := map[string]CacheKey{}
	for _, t := range texts {
		if _, ok := keys[t]; !ok {
			keys[t] = key(t)
			uniq = append(uniq, t)
		}
	}
	out := make(map[string][]float32, len(uniq))
	if len(uniq) == 0 {
		return out, nil
	}
	lookup := make([]CacheKey, len(uniq))
	for i, t := range uniq {
		lookup[i] = keys[t]
	}
	got, err := e.cache.Get(ctx, index, lookup)
	if err != nil {
		return nil, err
	}
	var misses []string
	for _, t := range uniq {
		if v, ok := got[keys[t]]; ok {
			out[t] = v
		} else {
			misses = append(misses, t)
		}
	}
	if len(misses) == 0 {
		return out, nil
	}
	vs, err := embed(ctx, misses)
	if err != nil {
		return nil, err
	}
	put := make(map[CacheKey][]float32, len(misses))
	for i, t := range misses {
		out[t], put[keys[t]] = vs[i], vs[i]
	}
	if err := e.cache.Put(ctx, index, put); err != nil {
		return nil, err
	}
	return out, nil
}

// embedItems embeds item texts as queries: the ones that make one chunk at
// most whole, as given, in one call, and longer ones one at a time in pieces
// (see embedItem).
func (e *engine) embedItems(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	var whole []string
	var wholeAt, chunkedAt []int
	for i, t := range texts {
		if len(Chunk(t, e.cfg.ChunkBytes)) <= 1 {
			whole, wholeAt = append(whole, t), append(wholeAt, i)
		} else {
			chunkedAt = append(chunkedAt, i)
		}
	}
	if len(whole) > 0 {
		vs, err := embedBatch(ctx, e.emb, langai.RoleQuery, whole)
		if err != nil {
			return nil, err
		}
		for j, i := range wholeAt {
			out[i] = vs[j]
		}
	}
	for _, i := range chunkedAt {
		v, err := embedItem(ctx, e.emb, e.cfg.ChunkBytes, texts[i])
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

// embedBatch embeds texts in one call, checking the answer as embedOne does:
// one vector per text, none empty, and each of the embedder's Dimensions when
// it says them.
func embedBatch(ctx context.Context, emb langai.Embedder, role langai.Role, texts []string) ([][]float32, error) {
	vs, err := emb.Embed(ctx, role, texts)
	if err != nil {
		return nil, err
	}
	if len(vs) != len(texts) {
		return nil, fmt.Errorf("tagging: embedder returned %d vectors for %d %s texts", len(vs), len(texts), role)
	}
	dims := emb.Dimensions()
	for _, v := range vs {
		switch {
		case dims > 0 && len(v) != dims:
			return nil, fmt.Errorf("tagging: embedder returned a vector of %d dimensions, want %d", len(v), dims)
		case len(v) == 0:
			return nil, errors.New("tagging: embedder returned an empty vector")
		}
	}
	return vs, nil
}

package tagging

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/go-bumbu/langai"
)

var (
	errEmptyRevision = errors.New("tagging: empty revision")
	errBlankText     = errors.New("tagging: blank text")
)

// skippedMark ends the fingerprint of a tag built without one of its members,
// because the member's text could not be embedded. No fingerprint the engine
// computes ends this way (they are hex digests), so the tag is never taken
// for up to date: it is built again whenever the space's revision changes,
// and the member gets another try.
const skippedMark = "+skipped"

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
	flight flight
	log    *slog.Logger     // cfg.Logger, or one that discards
	now    func() time.Time // time.Now, replaced by tests

	// closing ends, by stop, when the engine is closed (see enter and close);
	// mu guards closed, and calls counts the calls in flight
	closing context.Context
	stop    context.CancelFunc
	mu      sync.Mutex
	closed  bool
	calls   sync.WaitGroup
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
	logger := cfg.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	e := &engine{emb: emb, cat: cat, cache: cache, ix: ix, cfg: cfg, params: params, log: logger, now: time.Now}
	e.closing, e.stop = context.WithCancel(context.Background())
	e.flight.grace = cfg.RebuildGrace
	e.flight.onPanic = func(value any, stack []byte) {
		e.log.Error("tagging: panic in a rebuild", logComponent, slog.Any("panic", value), slog.String("stack", string(stack)))
	}
	return e, nil
}

// scores returns text's score against every tag of the space at revision,
// best first, as Scope.Scores does: it gets the vectors of text's pieces (from
// the cache, or embedded and cached), brings the space up to revision (see
// refresh), and scores the index's candidates (see itemScore). Any error
// returns no scores; one of a rebuild wraps ErrNotRebuilt.
func (e *engine) scores(ctx context.Context, index, partition, revision, text string) ([]TagScore, error) {
	if err := checkSpace(index, revision); err != nil {
		return nil, err
	}
	if strings.TrimSpace(text) == "" {
		return nil, errBlankText
	}
	vecs, failed, err := e.itemVectors(ctx, index, []string{text}, "")
	if err != nil {
		return nil, err
	}
	if err := failed[text]; err != nil {
		return nil, fmt.Errorf("tagging: the text could not be embedded: %w", err)
	}
	pieces := vecs[text]
	query, err := queryVector(pieces)
	if err != nil {
		return nil, fmt.Errorf("tagging: the pieces of the text cannot be scored: %w", err)
	}
	cands, err := e.candidates(ctx, index, partition, revision, query)
	if err != nil {
		return nil, err
	}
	return scoreVectors(pieces, cands, !e.cfg.AllPathLevels)
}

// candidates brings the space up to revision (see refresh) and returns the
// index's candidates for vec. A space evicted between the two reads as holding
// no tags, which is not an answer: when the index knows no such space, it
// refreshes once more and reads the candidates again. Once, not in a loop: a
// budget too small to keep the space until the next line cannot be out-waited.
func (e *engine) candidates(ctx context.Context, index, partition, revision string, vec []float32) ([]TagVector, error) {
	if err := e.refresh(ctx, index, partition, revision); err != nil {
		return nil, err
	}
	cands, err := e.ix.Candidates(ctx, index, partition, vec)
	if err != nil || len(cands) > 0 {
		return cands, err
	}
	_, _, ok, err := e.ix.State(ctx, index, partition)
	if err != nil || ok {
		return cands, err
	}
	if err := e.refresh(ctx, index, partition, revision); err != nil {
		return nil, err
	}
	return e.ix.Candidates(ctx, index, partition, vec)
}

// warm brings the space up to revision (see refresh) and caches the vectors
// of texts, the blank ones skipped, so no later scores pays for them. A text
// the embedder fails on while it takes others does not stop them: they are
// embedded and cached, and the error names each one that was not by its place
// in texts. An embedder that is down ends the call with its error (see
// vectors).
func (e *engine) warm(ctx context.Context, index, partition, revision string, texts []string) error {
	if err := checkSpace(index, revision); err != nil {
		return err
	}
	if err := e.refresh(ctx, index, partition, revision); err != nil {
		return err
	}
	_, failed, err := e.itemVectors(ctx, index, nonBlank(texts), "")
	if err != nil || len(failed) == 0 {
		return err
	}
	var errs []error
	for i, t := range texts {
		if cause := failed[t]; cause != nil {
			errs = append(errs, fmt.Errorf("tagging: text %d of %d (%d bytes) could not be embedded: %w", i+1, len(texts), len(t), cause))
		}
	}
	return errors.Join(errs...)
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
// state at a time, and the rebuilds of one space run one after the other
// whatever state each is for (see flight): two instances of an application
// that are a revision apart would otherwise rebuild the space side by side,
// and the second Apply would find the space moved under it and leave it
// recorded as nobody's. A caller that waited for a rebuild to another state
// looks at the index again, and rebuilds only if that left the space at
// neither.
func (e *engine) refresh(ctx context.Context, index, partition, revision string) error {
	state := e.state(revision)
	for {
		cur, _, ok, err := e.ix.State(ctx, index, partition)
		switch {
		case err != nil:
			return err
		case ok && cur == state:
			return nil
		}
		again, err := e.flight.do(ctx, Space{index, partition}, state, func(ctx context.Context) error {
			return e.rebuild(ctx, index, partition, state)
		})
		if again {
			continue
		}
		var panicked *panicError
		if errors.As(err, &panicked) {
			return fmt.Errorf("%w: %w", ErrNotRebuilt, err)
		}
		return err
	}
}

// rebuild brings the space to state, as one run of the flight. It re-reads
// the index first, since another caller may just have applied that state, and
// hands Apply the state it diffed against, so an Apply over a space that
// changed in the meantime records nothing and the next call diffs again (see
// TagIndex.Apply).
func (e *engine) rebuild(ctx context.Context, index, partition, state string) error {
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
	return e.ix.Apply(ctx, index, partition, cur, state, put, drop)
}

// enter begins a call on the engine: it fails with errClosed once the engine
// is closed, and else returns a context that ends with ctx or with close, and
// the function that ends the call, which the caller must run.
func (e *engine) enter(ctx context.Context) (context.Context, func(), error) {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, nil, errClosed
	}
	e.calls.Add(1)
	e.mu.Unlock()
	ctx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(e.closing, cancel)
	return ctx, func() {
		stop()
		cancel()
		e.calls.Done()
	}, nil
}

// call runs fn as one call on the engine (see enter), and returns its error
// as closedErr gives it.
func (e *engine) call(ctx context.Context, fn func(context.Context) error) error {
	ctx, done, err := e.enter(ctx)
	if err != nil {
		return err
	}
	defer done()
	return e.closedErr(fn(ctx))
}

// closedErr is the error of a call that ended with err: err itself, or, when
// the engine was closed while the call ran and so cut it short, one wrapping
// langai.ErrClosed too.
func (e *engine) closedErr(err error) error {
	if err == nil || errors.Is(err, langai.ErrClosed) || e.closing.Err() == nil {
		return err
	}
	return fmt.Errorf("%w: %w", errClosed, err)
}

// close is Tagger.Close: no call starts after it, the contexts of the calls
// and rebuilds in flight are cancelled, and it returns when they have ended.
func (e *engine) close() {
	e.mu.Lock()
	e.closed = true
	e.mu.Unlock()
	e.stop()
	e.flight.close()
	e.calls.Wait()
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
	for start := 0; start < len(put); start += membersGroupSize {
		if err := e.build(ctx, index, partition, put[start:min(start+membersGroupSize, len(put))]); err != nil {
			return nil, nil, err
		}
	}
	return put, drop, nil
}

// build fills in the Vec and Cutoff of tags, at most membersGroupSize of
// them, from their names and members, in one Members call. Members of ids not
// asked for are ignored, and blank texts skipped. A member enters as one
// vector (see memberVector).
//
// A member whose text the embedder could not take is left out, reported to
// the Logger, and marks its tag to be built again at the next revision (see
// skippedMark): one text the embedder cannot take must not keep a whole space
// from being built. That is only so while the embedder works, which a later
// text of the call, or else the first tag's name as the canary, has to show
// (see embedRun): an embedder that is down fails the build, so that an outage
// is never recorded as a space built from its names alone. A name that could
// not be embedded leaves its tag with nothing to be built from, so it is an
// error naming the tag, and so is a member with a vector not as long as the
// tag's name vector, which blend and tagCutoff index one by the other.
// Everything else is embedded and cached before either error is returned.
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
	items, skipped, err := e.itemVectors(ctx, index, texts, names[0])
	if err != nil {
		return err
	}
	nameVecs, unnamed, err := e.vectors(ctx, index, langai.RoleDocument, names, "")
	if err != nil {
		return err
	}
	var errs []error
	for i := range tags {
		t := &tags[i]
		if cause := unnamed[t.Name]; cause != nil {
			errs = append(errs, fmt.Errorf("tagging: the name of tag %q could not be embedded: %w", t.ID, cause))
			continue
		}
		if err := e.buildTag(index, partition, t, nameVecs[t.Name], members[t.ID], items, skipped); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// buildTag fills in the Vec and Cutoff of t from its name's vector and the
// vectors, in items, of its members, leaving out and reporting the members in
// skipped (see build).
func (e *engine) buildTag(index, partition string, t *TagVector, nameVec []float32, members []string,
	items map[string][][]float32, skipped map[string]error) error {
	vecs := make([][]float32, 0, len(members))
	reported := map[string]bool{}
	for _, m := range members {
		if cause := skipped[m]; cause != nil {
			if !reported[m] {
				reported[m] = true
				sum := sha256.Sum256([]byte(m))
				e.log.Warn("tagging: member text skipped, it could not be embedded", logComponent,
					slog.String("index", index), slog.String("partition", partition), slog.String("tag", t.ID),
					slog.String("text_sha256", hex.EncodeToString(sum[:])), slog.Int("text_bytes", len(m)), slog.Any("err", cause))
			}
			continue
		}
		pieces := items[m]
		for _, v := range pieces {
			if len(v) != len(nameVec) {
				return fmt.Errorf("tagging: tag %q has a member with a vector of %d dimensions and a name vector of %d", t.ID, len(v), len(nameVec))
			}
		}
		vecs = append(vecs, memberVector(pieces, nameVec))
	}
	if len(reported) > 0 {
		t.Fingerprint += skippedMark
	}
	k := e.cfg.SmoothingK
	t.Vec = blend(nameVec, vecs, k)
	t.Cutoff = storedCutoff(tagCutoff(nameVec, vecs, k, e.cfg.NameMinScore, e.cfg.MemberMinScore))
	return nil
}

// incomplete is Scope.Incomplete: the ids, sorted, of the tags the index
// holds under a fingerprint that ends in skippedMark.
func (e *engine) incomplete(ctx context.Context, index, partition string) ([]string, error) {
	if index == "" {
		return nil, errEmptyIndex
	}
	_, held, _, err := e.ix.State(ctx, index, partition)
	if err != nil {
		return nil, err
	}
	var ids []string
	for id, fp := range held {
		if strings.HasSuffix(fp, skippedMark) {
			ids = append(ids, id)
		}
	}
	slices.Sort(ids)
	return ids, nil
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

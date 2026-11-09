package tagging

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"

	"github.com/go-bumbu/langai"
)

// embedBatchSize is the most texts the engine hands the embedder in one call.
// Each batch is cached when it comes back, so a failure or a cancel loses at
// most one batch of work, and a batch that fails is small enough to try again
// one text at a time. An embedder gives a text the same vector in any batch,
// so the size changes no score.
const embedBatchSize = 32

// maxEmbedFailuresInARow is how many texts in a row the embedder may fail on,
// each embedded alone, before the engine stops and checks that it still
// answers at all (see embedRun.confirm): a dead embedder is called a handful
// of times, not once per text.
const maxEmbedFailuresInARow = 3

// ErrCacheWrite is wrapped by the error of an EmbeddingCache.Get that read
// its vectors and returns them, but could not rewrite a last-used time: the
// database is a read-only replica, say, or the file is locked. The vectors
// are good and the Tagger uses them; it reports the error to
// TaggerConfig.Logger instead of failing the call. A cache of your own that
// cannot fail this way never returns it.
var ErrCacheWrite = errors.New("tagging: cache not written")

// logComponent tags the Tagger's log lines the way the providers tag theirs.
var logComponent = slog.String("component", "tagging")

// itemVectors returns the query vectors of the pieces of each of texts (see
// itemPieces), by text, as items and members are scored. Each piece is cached
// on its own, under the Tagger's model, so a piece another text shares, or a
// text that makes one piece, is embedded once whatever ChunkBytes is. A text
// with a piece that could not be embedded has no vectors: it is in failed,
// with the cause, and the others are not held up by it. The error is the one
// of vectors, which canary is passed on to: nothing more can be embedded.
func (e *engine) itemVectors(ctx context.Context, index string, texts []string, canary string) (vecs map[string][][]float32, failed map[string]error, err error) {
	pieces := make(map[string][]string, len(texts))
	var all []string
	for _, t := range texts {
		if _, ok := pieces[t]; ok {
			continue
		}
		pieces[t] = itemPieces(t, e.cfg.ChunkBytes)
		all = append(all, pieces[t]...)
	}
	got, bad, err := e.vectors(ctx, index, langai.RoleQuery, all, canary)
	if err != nil {
		return nil, nil, err
	}
	vecs = make(map[string][][]float32, len(pieces))
	failed = map[string]error{}
texts:
	for t, ps := range pieces {
		vs := make([][]float32, len(ps))
		for i, p := range ps {
			if err := bad[p]; err != nil {
				if len(ps) > 1 {
					err = fmt.Errorf("piece %d of %d: %w", i+1, len(ps), err)
				}
				failed[t] = err
				continue texts
			}
			vs[i] = got[p]
		}
		vecs[t] = vs
	}
	return vecs, failed, nil
}

// vectors returns the vector of each of texts embedded as role, by text: the
// cache's, in one Get, where it holds one as long as the embedder's Dimensions
// (any, when it does not say them), and the embedder's for the rest,
// embedBatchSize texts at a time, each batch cached as soon as it is embedded.
// A text listed twice is looked up and embedded once.
//
// A text the embedder fails on while it takes others, or answers with a vector
// that is no use (see checkVector), is in failed with the cause and is not
// cached; the rest are embedded all the same (see embedRun). The error is for
// what stops the whole call: a cache that cannot be read, a ctx that ended, or
// an embedder that is down (it is closed, or it fails on every text it is
// given, canary included), after which what was embedded before stays cached.
// canary, when not empty, is a tag's name: a text the embedder is expected to
// take as a document, embedded past the cache to tell the two apart when no
// text of the call embedded after one that failed. A cache that cannot be
// written is no error: the vectors are returned and the failure reported (see
// ErrCacheWrite for a Get; a failed Put only means embedding again later).
func (e *engine) vectors(ctx context.Context, index string, role langai.Role, texts []string, canary string) (vecs map[string][]float32, failed map[string]error, err error) {
	keys := make(map[string]CacheKey, len(texts))
	var uniq []string
	var lookup []CacheKey
	for _, t := range texts {
		if _, ok := keys[t]; !ok {
			keys[t] = KeyFor(e.cfg.Model, role, t)
			uniq, lookup = append(uniq, t), append(lookup, keys[t])
		}
	}
	vecs, failed = make(map[string][]float32, len(uniq)), map[string]error{}
	if len(uniq) == 0 {
		return vecs, failed, nil
	}
	got, err := e.cache.Get(ctx, index, lookup)
	if err != nil {
		if !errors.Is(err, ErrCacheWrite) {
			return nil, nil, err
		}
		e.log.Warn("tagging: last-used times of cached embeddings not refreshed", logComponent,
			slog.String("index", index), slog.Any("err", err))
	}
	// a cached vector that is not as long as the embedder's is another
	// model's, kept under this Model name: a miss, embedded again and
	// overwritten, never scored against this model's vectors
	dims := e.emb.Dimensions()
	var misses []string
	for _, t := range uniq {
		if v, ok := got[keys[t]]; ok && (dims <= 0 || len(v) == dims) {
			vecs[t] = v
		} else {
			misses = append(misses, t)
		}
	}
	run := embedRun{emb: e.emb, role: role, known: canary, knownRole: langai.RoleDocument, hasKnown: canary != ""}
	for start := 0; start < len(misses); start += embedBatchSize {
		batch := misses[start:min(start+embedBatchSize, len(misses))]
		vs, errs, err := run.batch(ctx, batch)
		if err != nil {
			return nil, nil, err
		}
		put := make(map[CacheKey][]float32, len(batch))
		for i, t := range batch {
			if errs[i] != nil {
				failed[t] = errs[i]
				continue
			}
			vecs[t], put[keys[t]] = vs[i], vs[i]
		}
		if len(put) == 0 {
			continue
		}
		if err := e.cache.Put(ctx, index, put); err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, nil, ctxErr
			}
			e.log.Warn("tagging: embeddings not cached, they will be embedded again", logComponent,
				slog.String("index", index), slog.Int("embeddings", len(put)), slog.Any("err", err))
		}
	}
	// the last texts may have failed with no Embed after them to vouch for
	// the embedder
	if err := run.confirm(ctx); err != nil {
		return nil, nil, err
	}
	return vecs, failed, nil
}

// embedRun is the embedder as one vectors call uses it. It tells a text the
// embedder cannot take from an embedder that is down, which looks the same
// one Embed at a time: a text's failure is its own only once an Embed after
// it has worked, of a later text of the call or of one the embedder is known
// to take. Until then nothing vouches for the embedder, and when that Embed
// fails too the call ends with the error: taking every text of an outage for
// a bad one would leave tags built without their members (see engine.build).
type embedRun struct {
	emb  langai.Embedder
	role langai.Role
	// known is a text the embedder takes in knownRole, when hasKnown: the
	// caller's canary at first, then the last text of the call it embedded
	known     string
	knownRole langai.Role
	hasKnown  bool
	// row is how many texts in a row failed, each embedded alone, since the
	// last Embed that worked, and cause the error of the last of them
	row   int
	cause error
}

// batch embeds texts as the run's role in one call and returns a vector or an
// error for each. When the call fails, or answers with another number of
// vectors than texts, the texts are embedded again one at a time, so one text
// the embedder cannot take costs only itself. A vector that is no use (see
// checkVector) is that text's error. The error it returns for the whole batch
// ends the vectors call: the one of embed, or of confirm after
// maxEmbedFailuresInARow texts in a row have failed.
func (r *embedRun) batch(ctx context.Context, texts []string) ([][]float32, []error, error) {
	errs := make([]error, len(texts))
	vs, failure, err := r.embed(ctx, texts)
	switch {
	case err != nil:
		return nil, nil, err
	case failure != nil:
		vs = make([][]float32, len(texts))
		for i := range texts {
			if len(texts) > 1 {
				var one [][]float32
				if one, failure, err = r.embed(ctx, texts[i:i+1]); err != nil {
					return nil, nil, err
				}
				if failure == nil {
					vs[i] = one[0]
					continue
				}
			}
			errs[i] = failure
			r.row, r.cause = r.row+1, failure
			if r.row < maxEmbedFailuresInARow {
				continue
			}
			if err := r.confirm(ctx); err != nil {
				return nil, nil, err
			}
		}
	}
	dims := r.emb.Dimensions()
	for i, v := range vs {
		if errs[i] != nil {
			continue
		}
		if errs[i] = checkVector(v, dims); errs[i] != nil {
			vs[i] = nil
		}
	}
	return vs, errs, nil
}

// embed embeds texts in one call. failure is the embedder's error when texts
// may be to blame for it, to be tried again one at a time and confirmed. err
// is for what no text is to blame for and so ends the whole vectors call: ctx
// has ended, or the embedder says that it is closed (langai.ErrClosed) or
// that a context of its own was cancelled or timed out. Nothing is tried
// again after it, since every text would fail the same way.
func (r *embedRun) embed(ctx context.Context, texts []string) (vs [][]float32, failure, err error) {
	vs, failure = embedTexts(ctx, r.emb, r.role, texts)
	switch {
	case failure == nil:
		r.known, r.knownRole, r.hasKnown = texts[len(texts)-1], r.role, true
		r.row, r.cause = 0, nil
		return vs, nil, nil
	case ctx.Err() != nil:
		return nil, nil, ctx.Err()
	case errors.Is(failure, langai.ErrClosed), errors.Is(failure, context.Canceled), errors.Is(failure, context.DeadlineExceeded):
		return nil, nil, failure
	}
	return nil, failure, nil
}

// confirm settles the failures since the last Embed that worked, if there are
// any: it embeds the text the embedder is known to take once more. When that
// works the embedder is up, and the failures were their texts'. When it fails
// too, or the run knows no such text because nothing has embedded yet, the
// embedder is taken for down and the error, wrapping the cause, ends the
// vectors call.
func (r *embedRun) confirm(ctx context.Context) error {
	switch {
	case r.row == 0:
		return nil
	case !r.hasKnown:
		return fmt.Errorf("tagging: the embedder failed on every text it was given: %w", r.cause)
	}
	if _, err := embedTexts(ctx, r.emb, r.knownRole, []string{r.known}); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return fmt.Errorf("tagging: the embedder is down, it fails on a text it is known to take: %w", err)
	}
	r.row, r.cause = 0, nil
	return nil
}

// embedTexts embeds texts in one call and refuses an embedder that breaks its
// contract by answering with another number of vectors than texts, rather than
// panicking on the missing one.
func embedTexts(ctx context.Context, emb langai.Embedder, role langai.Role, texts []string) ([][]float32, error) {
	vs, err := emb.Embed(ctx, role, texts)
	if err != nil {
		return nil, err
	}
	if len(vs) != len(texts) {
		return nil, fmt.Errorf("tagging: embedder returned %d vectors for %d %s texts", len(vs), len(texts), role)
	}
	return vs, nil
}

// checkVector refuses a vector the engine must not cache: an empty one, one
// that is not as long as the embedder's Dimensions when it says them (dims
// above zero), one with a NaN or an infinite value, and one of all zeros. The
// last two compare with nothing: blended into a tag they would keep it from
// matching any text for as long as the cache holds them.
func checkVector(v []float32, dims int) error {
	switch {
	case len(v) == 0:
		return errors.New("tagging: embedder returned an empty vector")
	case dims > 0 && len(v) != dims:
		return fmt.Errorf("tagging: embedder returned a vector of %d dimensions, want %d", len(v), dims)
	}
	var norm2 float64
	for _, x := range v {
		f := float64(x)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return errors.New("tagging: embedder returned a vector with a NaN or infinite value")
		}
		norm2 += f * f
	}
	if norm2 == 0 {
		return errors.New("tagging: embedder returned a vector of all zeros")
	}
	return nil
}

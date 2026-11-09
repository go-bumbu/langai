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
// of vectors: nothing more can be embedded.
func (e *engine) itemVectors(ctx context.Context, index string, texts []string) (vecs map[string][][]float32, failed map[string]error, err error) {
	pieces := make(map[string][]string, len(texts))
	var all []string
	for _, t := range texts {
		if _, ok := pieces[t]; ok {
			continue
		}
		pieces[t] = itemPieces(t, e.cfg.ChunkBytes)
		all = append(all, pieces[t]...)
	}
	got, bad, err := e.vectors(ctx, index, langai.RoleQuery, all)
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
// A text the embedder fails on, or answers with a vector that is no use (see
// checkVector), is in failed with the cause and is not cached; the rest are
// embedded all the same (see embedBatch). The error is for what stops the
// whole call: a cache that cannot be read, or a ctx that ended, after which
// what was embedded before stays cached. A cache that cannot be written is
// neither: the vectors are returned and the failure reported (see
// ErrCacheWrite for a Get; a failed Put only means embedding again later).
func (e *engine) vectors(ctx context.Context, index string, role langai.Role, texts []string) (vecs map[string][]float32, failed map[string]error, err error) {
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
	for start := 0; start < len(misses); start += embedBatchSize {
		batch := misses[start:min(start+embedBatchSize, len(misses))]
		vs, errs, err := e.embedBatch(ctx, role, batch)
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
	return vecs, failed, nil
}

// embedBatch embeds texts as role in one call and returns a vector or an error
// for each. When the call fails, or answers with another number of vectors
// than texts, the texts are embedded again one at a time, so one text the
// embedder cannot take costs only itself. A vector that is no use (see
// checkVector) is that text's error. The error it returns for the whole batch
// is ctx's: once ctx has ended nothing is tried again, since every text would
// fail the same way.
func (e *engine) embedBatch(ctx context.Context, role langai.Role, texts []string) ([][]float32, []error, error) {
	errs := make([]error, len(texts))
	vs, err := embedTexts(ctx, e.emb, role, texts)
	switch {
	case err == nil:
	case ctx.Err() != nil:
		return nil, nil, ctx.Err()
	case len(texts) == 1:
		return make([][]float32, 1), []error{err}, nil
	default:
		vs = make([][]float32, len(texts))
		for i := range texts {
			one, oneErrs, err := e.embedBatch(ctx, role, texts[i:i+1])
			if err != nil {
				return nil, nil, err
			}
			vs[i], errs[i] = one[0], oneErrs[0]
		}
		return vs, errs, nil
	}
	dims := e.emb.Dimensions()
	for i, v := range vs {
		if errs[i] = checkVector(v, dims); errs[i] != nil {
			vs[i] = nil
		}
	}
	return vs, errs, nil
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

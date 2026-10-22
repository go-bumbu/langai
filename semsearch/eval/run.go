package eval

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/go-bumbu/langai/semsearch"
)

// Options tunes the tagging report. The tagging pass judges each tag by its
// name alone, autotag's rule for a tag with no hand-tagged examples, and the
// zero value is that rule's default, so the gated test measures production
// behaviour for such tags.
type Options struct {
	// Floor is the cutoff precision/recall are reported at; 0 means 0.46,
	// autotag.DefaultNameMinScore (keep in sync).
	Floor float64
}

const (
	defaultFloor   = 0.46
	retrievalLimit = 10
)

// RecallKs are the cutoffs recall is reported at.
var RecallKs = []int{1, 3, 5, 10}

// Result is one evaluation of one embedder over one dataset. Retrieval is
// split as Dataset.SplitQueries does: SameLanguage is the search the service
// must get right (a German query for a German document), CrossLingual the
// search across languages (a Spanish query for an English document), which a
// model may or may not offer. Tagging covers every document against the tag
// paths as the dataset wrote them, so it crosses languages wherever the
// documents and the paths differ.
type Result struct {
	Dims         int
	SameLanguage Retrieval
	CrossLingual Retrieval
	Tagging      Tagging
	Timing       Timing
}

// Retrieval is the free-text search quality over one set of queries. Queries
// is the set's size; every metric is zero when it was empty.
type Retrieval struct {
	Queries  int
	RecallAt map[int]float64 // keyed by RecallKs
	MRR      float64
	NDCG10   float64
}

// Tagging is the tag-assignment quality over the dataset's documents.
type Tagging struct {
	Documents int
	AtFloor   Cut     // at Options.Floor
	Best      Cut     // the sweep's best F1 and the floor that reached it
	Top1      float64 // share of documents whose best-scoring tag is expected
}

// Timing is the mean embedding time per text: short texts are tag paths and
// queries, long ones are essences.
type Timing struct {
	ShortPerText time.Duration
	LongPerText  time.Duration
}

// Run evaluates emb over ds: a retrieval pass (documents indexed through the
// production chunker, every query searched) and a tagging pass (tag paths
// indexed, every essence searched), both over the exact in-memory store.
func Run(ctx context.Context, emb semsearch.Embedder, ds *Dataset, opts Options) (*Result, error) {
	if opts.Floor == 0 {
		opts.Floor = defaultFloor
	}
	timed := &timedEmbedder{inner: emb}
	res := &Result{Dims: emb.Dimensions()}
	if err := runRetrieval(ctx, timed, ds, res); err != nil {
		return nil, err
	}
	if err := runTagging(ctx, timed, ds, opts, res); err != nil {
		return nil, err
	}
	res.Timing = timed.timing()
	return res, nil
}

func runRetrieval(ctx context.Context, emb *timedEmbedder, ds *Dataset, res *Result) error {
	ix, err := semsearch.New(emb, semsearch.NewMemStore(), "eval", "eval", "docs")
	if err != nil {
		return err
	}
	emb.long = true
	for _, doc := range ds.Documents {
		if err := ix.Index(ctx, doc.ID, doc.Essence()); err != nil {
			return fmt.Errorf("eval: index %q: %w", doc.ID, err)
		}
	}
	emb.long = false
	same, cross := ds.SplitQueries()
	if res.SameLanguage, err = searchAll(ctx, ix, same); err != nil {
		return err
	}
	res.CrossLingual, err = searchAll(ctx, ix, cross)
	return err
}

// searchAll searches every query of one set and averages the metrics over it.
func searchAll(ctx context.Context, ix *semsearch.Index, queries []Query) (Retrieval, error) {
	acc := retrievalAcc{recall: map[int]float64{}}
	for _, q := range queries {
		hits, err := ix.Search(ctx, q.Text, retrievalLimit)
		if err != nil {
			return Retrieval{}, fmt.Errorf("eval: search %q: %w", q.ID, err)
		}
		ranked := make([]string, len(hits))
		for i, h := range hits {
			ranked[i] = h.DocID
		}
		acc.add(ranked, toSet(q.Relevant))
	}
	return acc.result(), nil
}

// retrievalAcc sums per-query metrics over one set of queries.
type retrievalAcc struct {
	n         int
	recall    map[int]float64
	mrr, ndcg float64
}

func (a *retrievalAcc) add(ranked []string, relevant relevantSet) {
	a.n++
	for _, k := range RecallKs {
		a.recall[k] += recallAt(ranked, relevant, k)
	}
	a.mrr += reciprocalRank(ranked, relevant)
	a.ndcg += ndcgAt(ranked, relevant, retrievalLimit)
}

func (a *retrievalAcc) result() Retrieval {
	out := Retrieval{Queries: a.n, RecallAt: make(map[int]float64, len(RecallKs))}
	n := float64(a.n)
	for _, k := range RecallKs {
		out.RecallAt[k] = mean(a.recall[k], n)
	}
	out.MRR = mean(a.mrr, n)
	out.NDCG10 = mean(a.ndcg, n)
	return out
}

// mean is sum over n, and 0 for an empty set rather than NaN, which would
// break the JSON report and read as a failure instead of an absence.
func mean(sum, n float64) float64 {
	if n == 0 {
		return 0
	}
	return sum / n
}

func runTagging(ctx context.Context, emb *timedEmbedder, ds *Dataset, opts Options, res *Result) error {
	ix, err := semsearch.New(emb, semsearch.NewMemStore(), "eval", "eval", "tags")
	if err != nil {
		return err
	}
	for _, tag := range ds.Tags {
		if err := ix.Index(ctx, tag, tag); err != nil {
			return fmt.Errorf("eval: index tag %q: %w", tag, err)
		}
	}
	floors := sweepFloors()
	acc := taggingAcc{sweep: make([]counts, len(floors))}
	emb.long = true
	for _, doc := range ds.Documents {
		hits, err := ix.Search(ctx, doc.Essence(), len(ds.Tags))
		if err != nil {
			return fmt.Errorf("eval: tag %q: %w", doc.ID, err)
		}
		scoredHits := make([]scored, len(hits))
		for i, h := range hits {
			scoredHits[i] = scored{Tag: h.DocID, Score: h.Score}
		}
		acc.add(scoredHits, toSet(doc.Tags), opts, floors)
	}
	emb.long = false
	res.Tagging = acc.result(opts, floors)
	return nil
}

// taggingAcc sums per-document decisions, at the reporting floor and at every
// sweep floor.
type taggingAcc struct {
	n, top1 int
	atFloor counts
	sweep   []counts
}

func (a *taggingAcc) add(hits []scored, expected relevantSet, opts Options, floors []float64) {
	a.n++
	if top1(hits, expected) {
		a.top1++
	}
	a.atFloor.add(decisions(hits, expected, opts.Floor))
	for i, f := range floors {
		a.sweep[i].add(decisions(hits, expected, f))
	}
}

func (a *taggingAcc) result(opts Options, floors []float64) Tagging {
	cuts := make([]Cut, len(floors))
	for i, f := range floors {
		cuts[i] = a.sweep[i].cut(f)
	}
	return Tagging{
		Documents: a.n,
		AtFloor:   a.atFloor.cut(opts.Floor),
		Best:      bestCut(cuts),
		Top1:      mean(float64(a.top1), float64(a.n)),
	}
}

// timedEmbedder records how long the embedder spends per text, split into
// short texts (tag paths, queries) and long ones (essences); the runner flips
// long around each phase.
type timedEmbedder struct {
	inner         semsearch.Embedder
	long          bool
	shortN, longN int
	shortD, longD time.Duration
}

func (t *timedEmbedder) Dimensions() int { return t.inner.Dimensions() }

func (t *timedEmbedder) Embed(ctx context.Context, role semsearch.Role, texts []string) ([][]float32, error) {
	start := time.Now()
	out, err := t.inner.Embed(ctx, role, texts)
	d := time.Since(start)
	if t.long {
		t.longN += len(texts)
		t.longD += d
	} else {
		t.shortN += len(texts)
		t.shortD += d
	}
	return out, err
}

func (t *timedEmbedder) timing() Timing {
	var out Timing
	if t.shortN > 0 {
		out.ShortPerText = t.shortD / time.Duration(t.shortN)
	}
	if t.longN > 0 {
		out.LongPerText = t.longD / time.Duration(t.longN)
	}
	return out
}

// String renders the result for test logs: a line per query set, one for
// tagging and one for timing.
func (r *Result) String() string {
	var b strings.Builder
	for _, set := range []struct {
		name string
		r    Retrieval
	}{{"search", r.SameLanguage}, {"xling", r.CrossLingual}} {
		fmt.Fprintf(&b, "%-8s queries=%d r@1=%.3f r@3=%.3f r@5=%.3f r@10=%.3f mrr=%.3f ndcg@10=%.3f\n",
			set.name+":", set.r.Queries, set.r.RecallAt[1], set.r.RecallAt[3], set.r.RecallAt[5], set.r.RecallAt[10], set.r.MRR, set.r.NDCG10)
	}
	tg := r.Tagging
	fmt.Fprintf(&b, "%-8s documents=%d f1@%.2f=%.3f (p=%.3f r=%.3f) best f1=%.3f @%.2f top1=%.3f\n",
		"tagging:", tg.Documents, tg.AtFloor.Floor, tg.AtFloor.F1, tg.AtFloor.Precision, tg.AtFloor.Recall, tg.Best.F1, tg.Best.Floor, tg.Top1)
	fmt.Fprintf(&b, "timing: short=%s/text long=%s/text dims=%d\n", r.Timing.ShortPerText, r.Timing.LongPerText, r.Dims)
	return b.String()
}

package yzma

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/zarf/eval"
)

// TestLongTagStrategies measures how a tagging.Generator should tag the long
// documents of zarf/eval/testdata/long: three ways of generating candidate
// names (a prefix of the text, 2 KB windows, 6 KB windows) crossed with the
// eval.LongStrategies that choose among them. Generation is the slow part
// (about an hour on a CPU) and is cached per variant and document as JSON
// under $SEMSEARCH_YZMA_TEST_DIR/longtags/<variant>/, so an interrupted run
// resumes and a finished one is re-scored without loading the chat model.
//
//	SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=~/.cache/semsearch-yzma go test ./provider/yzma/ -run TestLongTagStrategies -v -timeout 0
//
// LONGTAGS_TAU (default 0.75) is the similarity at which a tag counts as a
// topic's alias. The report goes to longtags/report.md in the same directory.
//
//nolint:gocyclo // one linear gated harness: generate, score, report
func TestLongTagStrategies(t *testing.T) {
	if os.Getenv("SEMSEARCH_MODELS") == "" {
		t.Skip("set SEMSEARCH_MODELS=1 to download and test the real models")
	}
	dir := os.Getenv("SEMSEARCH_YZMA_TEST_DIR")
	if dir == "" {
		t.Skip("SEMSEARCH_YZMA_TEST_DIR not set; skipping real-model test")
	}
	tau := 0.75
	if s := os.Getenv("LONGTAGS_TAU"); s != "" {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			t.Fatalf("LONGTAGS_TAU=%q: %v", s, err)
		}
		tau = v
	}
	ctx := context.Background()
	docs, err := eval.LoadLong(filepath.Join("..", "..", "zarf", "eval", "testdata", "long"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		ModelsDir:     filepath.Join(dir, "models"),
		LibDir:        filepath.Join(dir, "llama-cpp", LibTag),
		ManagedLibDir: true,
		Logger:        slog.New(slog.NewTextHandler(os.Stderr, nil)),
	}
	emb, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	root := filepath.Join(dir, "longtags")

	// the chat model loads on the first cache miss only, once for all variants
	counter := &countingCompleter{}
	gens := map[string]*tagging.Generator{}
	generator := func(v longVariant) *tagging.Generator {
		if g, ok := gens[v.name]; ok {
			return g
		}
		if counter.inner == nil {
			spec := DefaultCompleterSpec(cfg.ModelsDir)
			spec.ContextTokens = 4096
			start := time.Now()
			llm, err := OpenCompleter(ctx, spec, Runtime{LibDir: cfg.LibDir, ManagedLibDir: cfg.ManagedLibDir, Threads: cfg.Threads, Logger: cfg.Logger})
			if err != nil {
				t.Fatalf("OpenCompleter: %v", err)
			}
			t.Cleanup(func() { _ = llm.Close() })
			counter.inner = llm
			t.Logf("chat model ready in %s", time.Since(start))
		}
		g, err := tagging.NewGenerator(counter, emb, tagging.GeneratorConfig{MaxBytes: v.maxBytes})
		if err != nil {
			t.Fatal(err)
		}
		gens[v.name] = g
		return g
	}

	// phase 1: generation, cached after each document
	type key struct{ variant, doc int }
	generated := map[key]genResult{}
	for vi, v := range longVariants {
		for di, d := range docs {
			path := filepath.Join(root, v.name, fmt.Sprintf("%s-%s.json", d.ID, textHash(d.Text)))
			if r, ok := loadGenResult(path, v.name, d); ok {
				generated[key{vi, di}] = r
				continue
			}
			g := generator(v) // may download and load the model: outside the clock
			counter.reset()
			start := time.Now()
			names, err := v.generate(ctx, g, d)
			if err != nil {
				t.Fatalf("%s %s: %v", v.name, d.ID, err)
			}
			r := genResult{
				Variant: v.name, ID: d.ID, Hash: textHash(d.Text), Names: names,
				Calls: counter.calls, Seconds: time.Since(start).Seconds(),
			}
			if err := saveGenResult(path, r); err != nil {
				t.Fatalf("cache %s: %v", path, err)
			}
			generated[key{vi, di}] = r
			t.Logf("generated %s %s: %d names, %d calls, %.1fs", v.name, d.ID, len(names), r.Calls, r.Seconds)
		}
	}
	type total struct {
		calls   int
		seconds float64
	}
	totals := make([]total, len(longVariants))
	for vi, v := range longVariants {
		for di := range docs {
			r := generated[key{vi, di}]
			totals[vi].calls += r.Calls
			totals[vi].seconds += r.Seconds
		}
		t.Logf("variant %s: %d LLM calls, %.0f s over %d documents", v.name, totals[vi].calls, totals[vi].seconds, len(docs))
	}

	// phase 2: scoring, with embeddings only
	strategies := eval.LongStrategies()
	type rowKey struct {
		variant, strategy string
		cutoff            int // index into cutoffs
	}
	type row struct {
		scores []eval.LongScore
		picks  [][]string // per document
	}
	rows := map[rowKey]*row{}
	vecs := map[string][]float32{} // RoleDocument vectors of tags and aliases
	simOf := func(a, b string) float64 {
		va, vb := vecs[a], vecs[b]
		if va == nil || vb == nil {
			return 0
		}
		return cosineF32(va, vb)
	}
	sim := func(tag, alias string) float64 { return simOf(tag, alias) }

	for di, d := range docs {
		chunks := tagging.Chunk(d.Text, 2048)
		cvecs, err := emb.Embed(ctx, langai.RoleQuery, chunks)
		if err != nil || len(cvecs) != len(chunks) {
			t.Fatalf("embed chunks of %s: %v", d.ID, err)
		}
		centroid := unitCentroid(cvecs)

		// every name of every variant and every alias, in one call
		var need []string
		seen := map[string]bool{}
		add := func(s string) {
			if _, ok := vecs[s]; !ok && !seen[s] {
				seen[s] = true
				need = append(need, s)
			}
		}
		for vi := range longVariants {
			for _, n := range generated[key{vi, di}].Names {
				add(n.Name)
			}
		}
		for _, tp := range d.Topics {
			for _, a := range tp.Aliases {
				add(a)
			}
		}
		if len(need) > 0 {
			out, err := emb.Embed(ctx, langai.RoleDocument, need)
			if err != nil || len(out) != len(need) {
				t.Fatalf("embed names of %s: %v", d.ID, err)
			}
			for i, s := range need {
				vecs[s] = out[i]
			}
		}

		for vi, v := range longVariants {
			var cands []eval.Candidate
			for _, n := range generated[key{vi, di}].Names {
				c := eval.Candidate{Name: n.Name, ProposedBy: n.ProposedBy}
				if v.single {
					c.Cos, c.Mean = []float64{n.Score}, n.Score
				} else {
					for _, cv := range cvecs {
						c.Cos = append(c.Cos, cosineF32(cv, vecs[n.Name]))
					}
					c.Mean = cosineF32(centroid, vecs[n.Name])
				}
				cands = append(cands, c)
			}
			for _, s := range strategies {
				for ci := range cutoffs {
					picked := eval.Pick(cands, s, cutoffs[ci], 10)
					tags := make([]string, len(picked))
					for i, c := range picked {
						tags[i] = c.Name
					}
					k := rowKey{v.name, s.Name, ci}
					r := rows[k]
					if r == nil {
						r = &row{picks: make([][]string, len(docs))}
						rows[k] = r
					}
					r.scores = append(r.scores, eval.MatchTopics(tags, d.Topics, sim, tau))
					r.picks[di] = tags
				}
			}
		}
	}

	// report
	type line struct {
		k     rowKey
		score eval.LongScore
		calls int
		secs  float64
	}
	var lines []line
	for k, r := range rows {
		vi := slices.IndexFunc(longVariants, func(v longVariant) bool { return v.name == k.variant })
		lines = append(lines, line{k, eval.MeanLongScores(r.scores), totals[vi].calls, totals[vi].seconds})
	}
	slices.SortFunc(lines, func(a, b line) int {
		return cmp.Or(cmp.Compare(a.k.variant, b.k.variant), cmp.Compare(a.k.strategy, b.k.strategy), cmp.Compare(a.k.cutoff, b.k.cutoff))
	})
	var md strings.Builder
	fmt.Fprintf(&md, "# Long-text tag strategies\n\n%d documents, tau %.2f, limit 10.\n\n", len(docs), tau)
	md.WriteString("| variant | strategy | cutoff | recall | main | side | late side | mention | precision | F1 | tags | LLM calls | seconds |\n")
	md.WriteString("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, l := range lines {
		s := l.score
		fmt.Fprintf(&md, "| %s | %s | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %d | %d | %.0f |\n",
			l.k.variant, l.k.strategy, cutoffs[l.k.cutoff], s.Recall, s.MainRecall, s.SideRecall, s.LateSideRecall, s.MentionRate, s.Precision, s.F1(), s.Tags, l.calls, l.secs)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "report.md"), []byte(md.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	ci45 := slices.IndexFunc(cutoffs, func(c float64) bool { return c == logCutoff })
	for _, l := range lines {
		if l.k.cutoff != ci45 {
			continue
		}
		s := l.score
		t.Logf("cutoff %.2f %-8s %-13s recall %.3f main %.3f side %.3f late %.3f mention %.3f prec %.3f F1 %.3f tags %d",
			logCutoff, l.k.variant, l.k.strategy, s.Recall, s.MainRecall, s.SideRecall, s.LateSideRecall, s.MentionRate, s.Precision, s.F1(), s.Tags)
	}

	// the decision rule: at the log cutoff, rows whose main recall is no worse
	// than prefix's, ranked by F1; the three best, and those within 0.02 of it
	var prefixMain float64
	for _, l := range lines {
		if l.k.cutoff == ci45 && l.k.variant == "prefix" {
			prefixMain = math.Max(prefixMain, l.score.MainRecall)
		}
	}
	var atMid []line
	for _, l := range lines {
		if l.k.cutoff == ci45 && l.score.MainRecall >= prefixMain {
			atMid = append(atMid, l)
		}
	}
	slices.SortStableFunc(atMid, func(a, b line) int { return cmp.Compare(b.score.F1(), a.score.F1()) })
	if len(atMid) > 0 {
		bestF1 := atMid[0].score.F1()
		t.Logf("decision rule at cutoff %.2f: prefix main recall %.3f, best eligible F1 %.3f; rows within 0.02:", logCutoff, prefixMain, bestF1)
		for _, l := range atMid {
			if bestF1-l.score.F1() <= 0.02 {
				t.Logf("  %s / %s F1 %.3f main %.3f, %d LLM calls, %.0f s", l.k.variant, l.k.strategy, l.score.F1(), l.score.MainRecall, l.calls, l.secs)
			}
		}
	}
	if len(atMid) > 3 {
		atMid = atMid[:3]
	}
	for _, l := range atMid {
		t.Logf("best row %s / %s at cutoff %.2f, F1 %.3f", l.k.variant, l.k.strategy, logCutoff, l.score.F1())
		for di, d := range docs {
			t.Logf("  %s: %s", d.ID, strings.Join(rows[l.k].picks[di], " | "))
		}
	}

	// tag x topic similarities for three documents, to calibrate tau
	if len(atMid) > 0 {
		best := atMid[0].k
		for _, di := range []int{0, len(docs) / 2, len(docs) - 1} {
			d := docs[di]
			t.Logf("tag x topic sim (best alias) for %s, tags of %s / %s:", d.ID, best.variant, best.strategy)
			for _, tag := range rows[best].picks[di] {
				var b strings.Builder
				for _, tp := range d.Topics {
					bestSim := math.Inf(-1)
					for _, a := range tp.Aliases {
						bestSim = math.Max(bestSim, sim(tag, a))
					}
					fmt.Fprintf(&b, " %s[%s]=%.2f", strings.ReplaceAll(tp.Label, " ", "_"), tp.Kind, bestSim)
				}
				t.Logf("  %q:%s", tag, b.String())
			}
		}
	}
}

// logCutoff is the cutoff whose rows are logged; it is off the 0.02 grid, so
// cutoffs holds it as well.
const logCutoff = 0.45

// cutoffs are the score cutoffs compared: 0.40 to 0.60 in steps of 0.02, plus
// logCutoff, ascending.
var cutoffs = func() []float64 {
	var cs []float64
	for i := 0; i <= 10; i++ {
		cs = append(cs, math.Round((0.40+0.02*float64(i))*100)/100)
	}
	cs = append(cs, logCutoff)
	slices.Sort(cs)
	return cs
}()

// longVariant is one way of generating candidate names for a long text.
type longVariant struct {
	name     string
	maxBytes int // the Generator's MaxBytes
	// single marks a variant scored by the one score Generate returned
	single   bool
	generate func(ctx context.Context, g *tagging.Generator, d eval.LongDocument) ([]genName, error)
}

var longVariants = []longVariant{
	{name: "prefix", maxBytes: 2048, single: true, generate: func(ctx context.Context, g *tagging.Generator, d eval.LongDocument) ([]genName, error) {
		tags, err := g.Generate(ctx, d.Title+"\n"+d.Text, -1)
		if err != nil {
			return nil, err
		}
		names := make([]genName, len(tags))
		for i, tg := range tags {
			names[i] = genName{Name: tg.Name, ProposedBy: 1, Score: tg.Score}
		}
		return names, nil
	}},
	{name: "win2k", maxBytes: 4096, generate: windowed(2048)},
	{name: "win6k", maxBytes: 8192, generate: windowed(6144)},
}

// windowed has the Generator tag each window of the text, prefixed with the
// title, and merges the names case-insensitively under their first spelling.
func windowed(size int) func(context.Context, *tagging.Generator, eval.LongDocument) ([]genName, error) {
	return func(ctx context.Context, g *tagging.Generator, d eval.LongDocument) ([]genName, error) {
		var names []genName
		index := map[string]int{}
		for _, w := range tagging.Chunk(d.Text, size) {
			tags, err := g.Generate(ctx, d.Title+"\n"+w, -1)
			if err != nil {
				return nil, err
			}
			inWindow := map[string]bool{}
			for _, tg := range tags {
				k := strings.ToLower(tg.Name)
				if inWindow[k] {
					continue
				}
				inWindow[k] = true
				if i, ok := index[k]; ok {
					names[i].ProposedBy++
					continue
				}
				index[k] = len(names)
				names = append(names, genName{Name: tg.Name, ProposedBy: 1})
			}
		}
		return names, nil
	}
}

// genName is a candidate name; Score is set for the prefix variant only.
type genName struct {
	Name       string  `json:"name"`
	ProposedBy int     `json:"proposed_by"`
	Score      float64 `json:"score,omitempty"`
}

// genResult is the cached generation of one variant for one document.
type genResult struct {
	Variant string    `json:"variant"`
	ID      string    `json:"id"`
	Hash    string    `json:"hash"`
	Calls   int       `json:"llm_calls"`
	Seconds float64   `json:"seconds"`
	Names   []genName `json:"names"`
}

func textHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])[:8]
}

// loadGenResult returns the cached result at path when it parses and belongs
// to this variant and text.
func loadGenResult(path, variant string, d eval.LongDocument) (genResult, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return genResult{}, false
	}
	var r genResult
	if err := json.Unmarshal(b, &r); err != nil {
		return genResult{}, false
	}
	if r.Variant != variant || r.ID != d.ID || r.Hash != textHash(d.Text) {
		return genResult{}, false
	}
	return r, true
}

// saveGenResult writes r atomically, so an interrupted run never leaves a
// half-written cache file.
func saveGenResult(path string, r genResult) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// countingCompleter counts the calls to the Completer behind it.
type countingCompleter struct {
	inner langai.Completer
	calls int
}

func (c *countingCompleter) reset() { c.calls = 0 }

func (c *countingCompleter) Complete(ctx context.Context, system, user string) (string, error) {
	c.calls++
	return c.inner.Complete(ctx, system, user)
}

// cosineF32 is the cosine of two vectors, 0 when either is empty or zero.
func cosineF32(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// unitCentroid is the normalized sum of the normalized vectors.
func unitCentroid(vs [][]float32) []float32 {
	if len(vs) == 0 {
		return nil
	}
	sum := make([]float64, len(vs[0]))
	for _, v := range vs {
		var n float64
		for _, x := range v {
			n += float64(x) * float64(x)
		}
		n = math.Sqrt(n)
		if n == 0 {
			continue
		}
		for i, x := range v {
			sum[i] += float64(x) / n
		}
	}
	var n float64
	for _, x := range sum {
		n += x * x
	}
	n = math.Sqrt(n)
	out := make([]float32, len(sum))
	if n == 0 {
		return out
	}
	for i, x := range sum {
		out[i] = float32(x / n)
	}
	return out
}

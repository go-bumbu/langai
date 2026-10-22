package eval

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/go-bumbu/langai/semsearch"
)

// vocabEmbedder maps each word of a fixed vocabulary to its own axis and a
// text to the unit-length sum of its words' axes (words outside the
// vocabulary are ignored), so every cosine in these tests is a number you
// can work out by hand: shared words over the geometric mean of word counts.
type vocabEmbedder struct {
	vocab []string
	fail  bool
}

func (v vocabEmbedder) Dimensions() int { return len(v.vocab) }

func (v vocabEmbedder) Embed(_ context.Context, _ semsearch.Role, texts []string) ([][]float32, error) {
	if v.fail {
		return nil, errors.New("embedder down")
	}
	out := make([][]float32, len(texts))
	for i, text := range texts {
		vec := make([]float32, len(v.vocab))
		for _, word := range strings.Fields(strings.ToLower(text)) {
			for j, known := range v.vocab {
				if word == known {
					vec[j]++
				}
			}
		}
		var sum float64
		for _, x := range vec {
			sum += float64(x) * float64(x)
		}
		if sum > 0 {
			inv := float32(1 / math.Sqrt(sum))
			for j := range vec {
				vec[j] *= inv
			}
		}
		out[i] = vec
	}
	return out, nil
}

// inlineDataset: two documents, two tags, four queries.
//
//	go-doc essence words:   go modules go modules        → (go 2, modules 2)
//	pasta-doc essence words: recetas de pasta pasta tomato recipes → (pasta 2, tomato 1, recipes 1)
//	tag "dev / go" words: dev go;  tag "cooking / recipes" words: cooking recipes
//
// Cosines: go-doc·"dev / go" = 2/(√8·√2) = 0.5; pasta-doc·"cooking / recipes"
// = 1/(√6·√2) ≈ 0.289; the other two pairs share nothing and score 0.
func inlineDataset() *Dataset {
	return &Dataset{
		Tags: []string{"dev / go", "cooking / recipes"},
		Documents: []Document{
			{ID: "go-doc", Lang: "en", Title: "Go modules", Text: "go modules", Tags: []string{"dev / go"}},
			{ID: "pasta-doc", Lang: "es", Title: "Recetas de pasta", Text: "pasta tomato recipes", Tags: []string{"cooking / recipes"}},
		},
		Queries: []Query{
			{ID: "q-go", Lang: "en", Text: "go modules", Relevant: []string{"go-doc"}},          // cos(go-doc) = 1
			{ID: "q-pasta", Lang: "en", Text: "pasta tomato", Relevant: []string{"pasta-doc"}},  // cross-lingual (en → es); cos(pasta-doc) ≈ 0.866
			{ID: "q-recipes", Lang: "en", Text: "recipes", Relevant: []string{"go-doc"}},        // pasta-doc ranks first (0.408), go-doc second (0)
			{ID: "q-tomate", Lang: "es", Text: "tomato pasta", Relevant: []string{"pasta-doc"}}, // same language; cos(pasta-doc) ≈ 0.866
		},
	}
}

var vocab = []string{"go", "modules", "pasta", "tomato", "recipes", "dev", "cooking"}

func runInline(t *testing.T) *Result {
	t.Helper()
	ds := inlineDataset()
	if err := ds.Validate(); err != nil {
		t.Fatal(err)
	}
	res, err := Run(context.Background(), vocabEmbedder{vocab: vocab}, ds, Options{})
	if err != nil {
		t.Fatal(err)
	}
	return res
}

// TestRun_SplitsRetrievalByLanguageCrossing: the same-language set holds
// q-go, q-recipes and q-tomate (Spanish, for the Spanish document), the
// cross-lingual set only q-pasta (English, for the Spanish document).
func TestRun_SplitsRetrievalByLanguageCrossing(t *testing.T) {
	res := runInline(t)
	if res.Dims != len(vocab) {
		t.Errorf("Dims = %d, want %d", res.Dims, len(vocab))
	}
	same := res.SameLanguage
	// q-go and q-tomate hit at rank 1; q-recipes at rank 2
	if same.Queries != 3 || !near(same.RecallAt[1], 2.0/3) || !near(same.RecallAt[3], 1) || !near(same.MRR, (1+0.5+1)/3) {
		t.Errorf("same-language = %+v", same)
	}
	wantNDCG := (1 + 1/math.Log2(3) + 1) / 3
	if !near(same.NDCG10, wantNDCG) {
		t.Errorf("same-language nDCG@10 = %f, want %f", same.NDCG10, wantNDCG)
	}
	cross := res.CrossLingual
	if cross.Queries != 1 || !near(cross.RecallAt[1], 1) || !near(cross.MRR, 1) {
		t.Errorf("cross-lingual = %+v, want only q-pasta, found at rank 1", cross)
	}
}

// TestRun_CrossLingualFollowsTheRelevantDocuments pins which set a query
// lands in: the cross-lingual one when a document it should find is in
// another language than the query itself. Both queries here are Spanish, and
// only the one that should find the English document crosses a language; a
// rule keyed on the query's language alone ("anything not English") would put
// both there. The titles use no vocabulary words, so each essence is exactly
// its text and each query matches its own document with cosine 1.
func TestRun_CrossLingualFollowsTheRelevantDocuments(t *testing.T) {
	ds := &Dataset{
		Tags: []string{"dev / go", "cooking / recipes"},
		Documents: []Document{
			{ID: "go-doc", Lang: "en", Title: "English note", Text: "go modules", Tags: []string{"dev / go"}},
			{ID: "pasta-doc", Lang: "es", Title: "Nota en español", Text: "pasta tomato", Tags: []string{"cooking / recipes"}},
		},
		Queries: []Query{
			// Spanish query, English document: cross-lingual
			{ID: "q-es-finds-en", Lang: "es", Text: "go modules", Relevant: []string{"go-doc"}},
			// Spanish query, Spanish document: same language
			{ID: "q-es-finds-es", Lang: "es", Text: "pasta tomato", Relevant: []string{"pasta-doc"}},
		},
	}
	if err := ds.Validate(); err != nil {
		t.Fatal(err)
	}
	res, err := Run(context.Background(), vocabEmbedder{vocab: vocab}, ds, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if cross := res.CrossLingual; cross.Queries != 1 || !near(cross.MRR, 1) {
		t.Errorf("cross-lingual = %+v, want just the Spanish query for the English document, found at rank 1", cross)
	}
	if same := res.SameLanguage; same.Queries != 1 || !near(same.MRR, 1) {
		t.Errorf("same-language = %+v, want just the Spanish query for the Spanish document, found at rank 1", same)
	}
}

// TestRun_EmptyQuerySetIsZero: a dataset without cross-lingual queries (a
// private export, say) reports zeros for that set, never NaN, which would
// read as a failure and cannot be encoded as JSON.
func TestRun_EmptyQuerySetIsZero(t *testing.T) {
	ds := inlineDataset()
	ds.Queries = ds.Queries[:1] // q-go alone: same-language
	res, err := Run(context.Background(), vocabEmbedder{vocab: vocab}, ds, Options{})
	if err != nil {
		t.Fatal(err)
	}
	cross := res.CrossLingual
	if cross.Queries != 0 || cross.RecallAt[5] != 0 || cross.MRR != 0 || cross.NDCG10 != 0 {
		t.Errorf("cross-lingual over no queries = %+v, want zeros", cross)
	}
	if len(cross.RecallAt) != len(RecallKs) {
		t.Errorf("RecallAt has %d entries, want one per cutoff even when empty", len(cross.RecallAt))
	}
	if same := res.SameLanguage; same.Queries != 1 || !near(same.MRR, 1) {
		t.Errorf("same-language = %+v, want the one query, found at rank 1", same)
	}
}

func TestRun_Tagging(t *testing.T) {
	tg := runInline(t).Tagging
	// at the default floor 0.46: go-doc gets "dev / go" (0.5), pasta-doc gets
	// nothing (0.289) → tp 1, fp 0, fn 1
	if tg.Documents != 2 || tg.AtFloor.Floor != 0.46 || !near(tg.AtFloor.Precision, 1) || !near(tg.AtFloor.Recall, 0.5) || !near(tg.AtFloor.F1, 2.0/3) {
		t.Errorf("at floor = %+v", tg.AtFloor)
	}
	// from 0.05 up to 0.28 both documents get exactly their tag: F1 1 at the
	// lowest such floor
	if !near(tg.Best.F1, 1) || !near(tg.Best.Floor, 0.05) {
		t.Errorf("best = %+v", tg.Best)
	}
	if !near(tg.Top1, 1) {
		t.Errorf("top1 = %f, want 1", tg.Top1)
	}
}

func TestRun_OptionsAreApplied(t *testing.T) {
	ds := inlineDataset()
	res, err := Run(context.Background(), vocabEmbedder{vocab: vocab}, ds, Options{Floor: 0.2})
	if err != nil {
		t.Fatal(err)
	}
	if at := res.Tagging.AtFloor; at.Floor != 0.2 || !near(at.F1, 1) {
		t.Errorf("at floor 0.2 = %+v, want both documents tagged", at)
	}

	// The check above cannot see whether anything but the floor limits what a
	// document gets: at floor 0.2 each document's second tag (score 0) is
	// already cut by the floor. A floor of -1 (Run only replaces a floor of 0,
	// and every cosine here is >= 0) admits every tag, which leaves a cap, were
	// there one, as the only limit. There is none, as in autotag:
	// go-doc gets "dev / go" (0.5) and its zero-scoring second tag, pasta-doc
	// gets "cooking / recipes" (~0.289) and its zero-scoring second tag. That
	// is tp 2, fp 2, so precision 0.5 and F1 2/3, where a runner that capped
	// each document at one tag would give tp 2, fp 0, so precision and F1 of 1.
	res, err = Run(context.Background(), vocabEmbedder{vocab: vocab}, ds, Options{Floor: -1})
	if err != nil {
		t.Fatal(err)
	}
	if at := res.Tagging.AtFloor; !near(at.F1, 2.0/3) || !near(at.Precision, 0.5) {
		t.Errorf("at floor -1 = %+v, want every tag assigned to every document (precision 0.5, F1 2/3)", at)
	}
}

func TestRun_ReportsEmbedderErrors(t *testing.T) {
	_, err := Run(context.Background(), vocabEmbedder{vocab: vocab, fail: true}, inlineDataset(), Options{})
	if err == nil || !strings.Contains(err.Error(), "embedder down") || !strings.Contains(err.Error(), "go-doc") {
		t.Fatalf("err = %v, want the cause and the document it failed on", err)
	}
}

func TestTimedEmbedder_SplitsShortAndLong(t *testing.T) {
	te := &timedEmbedder{inner: vocabEmbedder{vocab: vocab}}
	if _, err := te.Embed(context.Background(), semsearch.RoleQuery, []string{"a", "b"}); err != nil {
		t.Fatal(err)
	}
	te.long = true
	if _, err := te.Embed(context.Background(), semsearch.RoleDocument, []string{"c"}); err != nil {
		t.Fatal(err)
	}
	if te.shortN != 2 || te.longN != 1 {
		t.Fatalf("counted %d short and %d long texts, want 2 and 1", te.shortN, te.longN)
	}
	if te.Dimensions() != len(vocab) {
		t.Fatal("Dimensions must forward")
	}
}

func TestResult_String(t *testing.T) {
	s := runInline(t).String()
	// the labels are padded so the three metric lines line up
	for _, want := range []string{"search:  queries=3", "xling:   queries=1", "r@5=", "mrr=", "ndcg@10=", "tagging: documents=2", "f1@0.46=", "best f1=", "top1=", "timing:", "dims=7"} {
		if !strings.Contains(s, want) {
			t.Errorf("String() lacks %q:\n%s", want, s)
		}
	}
}

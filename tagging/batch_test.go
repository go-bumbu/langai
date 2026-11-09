package tagging

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/go-bumbu/langai"
)

// poisonText is a member the embedder of these tests refuses, alone or in a
// batch (see tallyEmbedder.failContaining).
const poisonText = "A recipe the embedder chokes on: boom"

// loggedCfg is engineCfg with a Logger the test reads back.
func loggedCfg() (TaggerConfig, *logRecorder) {
	rec := &logRecorder{}
	cfg := engineCfg
	cfg.Logger = slog.New(rec)
	return cfg, rec
}

// recipes returns n distinct member texts about croissants.
func recipes(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("Croissant dough with cold butter, recipe %d", i)
	}
	return out
}

// cached reports which of texts the cache holds as items of the fake model.
func cached(t *testing.T, cache EmbeddingCache, texts ...string) map[string]bool {
	t.Helper()
	keys := make([]CacheKey, len(texts))
	for i, text := range texts {
		keys[i] = KeyFor("fake", langai.RoleQuery, text)
	}
	got, err := cache.Get(context.Background(), "1", keys)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]bool{}
	for i, text := range texts {
		_, out[text] = got[keys[i]]
	}
	return out
}

// TestEngine_APoisonMemberIsSkippedAndReported: one of 201 member texts fails
// to embed, alone and in any batch. The rebuild used to embed all of them in
// one call, cache nothing and fail, at every Suggest, for ever. Now the texts
// go to the embedder embedBatchSize at a time and each batch is cached, the
// batch that fails is tried one text at a time, and the one text that still
// fails is left out of its tag and reported: the first Suggest answers, and
// the next ones embed nothing.
func TestEngine_APoisonMemberIsSkippedAndReported(t *testing.T) {
	cfg, rec := loggedCfg()
	good := recipes(200)
	members := slices.Insert(slices.Clone(good), 100, poisonText)
	cat := &fakeCatalog{tags: breadAndJapan("1"), members: map[string][]string{"bread": members}}
	r := newRig(t, cfg, cat, nil, nil)
	r.emb.failContaining("boom", errBoom)

	first := r.scores(t, "1", croissantText)
	if got := picked(first); !slices.Equal(got, []string{"bread"}) {
		t.Errorf("picked %v, want [bread]: the tag is built from its other members", got)
	}
	if n := r.emb.largestBatch(); n > embedBatchSize {
		t.Errorf("an Embed call held %d texts, want at most %d", n, embedBatchSize)
	}
	// the item, seven batches of members, and the failed batch once more, one
	// text at a time
	if n, want := r.emb.callsOf(langai.RoleQuery), 1+7+embedBatchSize; n != want {
		t.Errorf("%d Embed calls of items, want %d", n, want)
	}
	for text, ok := range cached(t, r.cache, members...) {
		if ok == (text == poisonText) {
			t.Errorf("cached(%q) = %v", text, ok)
		}
	}
	warned := rec.matching("WARN", "member", "tag=bread", "index=1", "partition=10", errBoom.Error())
	if len(warned) != 1 {
		t.Errorf("the skipped member was reported %d times, want once; log:\n%s", len(warned), strings.Join(rec.lines(), "\n"))
	}

	calls, embedded := r.emb.callsOf(langai.RoleQuery), r.emb.total()
	_, memberCalls := cat.counts()
	for i := range 2 {
		if again := r.scores(t, "1", croissantText); !reflect.DeepEqual(again, first) {
			t.Errorf("call %d scored %+v, the first %+v", i+2, again, first)
		}
	}
	if n := r.emb.callsOf(langai.RoleQuery) - calls; n != 0 || r.emb.total() != embedded {
		t.Errorf("two more calls made %d Embed calls, want none: the space is built", n)
	}
	if _, m := cat.counts(); m != memberCalls {
		t.Errorf("two more calls read the members %d times, want none", m-memberCalls)
	}
}

// TestEngine_ASkippedMemberIsTriedAgainAtTheNextRevision: a text may have
// failed for a reason that passes, so its tag is rebuilt at the next revision
// of the space although the tag's own revision did not change, and once the
// member embeds the tag is left alone again.
func TestEngine_ASkippedMemberIsTriedAgainAtTheNextRevision(t *testing.T) {
	cfg, rec := loggedCfg()
	members := append(slices.Clone(croissantRecipes), poisonText)
	cat := &fakeCatalog{tags: breadAndJapan("1"), members: map[string][]string{"bread": members}}
	r := newRig(t, cfg, cat, nil, nil)
	r.emb.failContaining("boom", errBoom)
	without := scoreOf(t, r.scores(t, "1", croissantText), "bread")

	r.scores(t, "2", croissantText)
	if n := len(rec.matching("WARN", "member", "tag=bread")); n != 2 {
		t.Fatalf("the member was reported %d times over two revisions, want twice", n)
	}
	if got := r.cat.lastMemberIDs(); !slices.Equal(got, []string{"bread"}) {
		t.Fatalf("the second revision read the members of %v, want [bread]: only the tag with a skipped member", got)
	}

	r.emb.failContaining("", nil)
	with := scoreOf(t, r.scores(t, "3", croissantText), "bread")
	if !cached(t, r.cache, poisonText)[poisonText] {
		t.Fatal("the member was not embedded once the embedder took it")
	}
	if with == without {
		t.Error("the tag scores as it did without the member")
	}
	_, memberCalls := cat.counts()
	r.scores(t, "4", croissantText)
	if _, m := cat.counts(); m != memberCalls {
		t.Errorf("the tag was rebuilt again at the next revision, with no member skipped")
	}
}

// TestEngine_ATagNameThatFailsFailsTheRebuildNamingTheTag: without its name a
// tag has no vector, so the rebuild fails, naming the tag; the other names are
// cached all the same, and the retry embeds only the one that failed.
func TestEngine_ATagNameThatFailsFailsTheRebuildNamingTheTag(t *testing.T) {
	r := newRig(t, engineCfg, &fakeCatalog{tags: breadAndJapan("1")}, nil, nil)
	r.emb.failContaining("cooking", errBoom)
	ss, err := r.e.scores(context.Background(), "1", "10", "1", diaryText)
	if !errors.Is(err, ErrNotRebuilt) || !errors.Is(err, errBoom) || !strings.Contains(err.Error(), `"bread"`) {
		t.Fatalf("err = %v, want one wrapping ErrNotRebuilt and the cause, naming the tag bread", err)
	}
	if strings.Contains(err.Error(), `"japan"`) || ss != nil {
		t.Fatalf("scores %+v, err %v: want no scores and only bread named", ss, err)
	}
	r.emb.failContaining("", nil)
	japan := r.emb.times(langai.RoleDocument, "travel / japan")
	if got := picked(r.scores(t, "1", diaryText)); !slices.Equal(got, []string{"japan"}) {
		t.Fatalf("retry: %v, want [japan]", got)
	}
	if n := r.emb.times(langai.RoleDocument, "travel / japan") - japan; n != 0 {
		t.Errorf("the retry embedded the name that had worked %d times, want 0", n)
	}
}

// TestEngine_WarmKeepsWhatItEmbeddedBeforeACancel: a Warm over a corpus that
// is cancelled keeps every batch it finished, so the next one embeds the rest,
// and a cancelled batch is not tried again one text at a time.
func TestEngine_WarmKeepsWhatItEmbeddedBeforeACancel(t *testing.T) {
	texts := recipes(200)
	r := newRig(t, engineCfg, nil, nil, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r.emb.atCall(func(call int) error {
		if call == 4 {
			cancel()
			return ctx.Err()
		}
		return nil
	})
	if err := r.e.warm(ctx, "1", "10", "1", texts); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if n := r.emb.callsOf(langai.RoleQuery); n != 4 {
		t.Fatalf("%d Embed calls, want 4: three batches and the cancelled one", n)
	}
	r.emb.atCall(nil)
	before := r.emb.count(langai.RoleQuery)
	if err := r.e.warm(context.Background(), "1", "10", "1", texts); err != nil {
		t.Fatal(err)
	}
	if n, want := r.emb.count(langai.RoleQuery)-before, len(texts)-3*embedBatchSize; n != want {
		t.Errorf("the second Warm embedded %d texts, want %d: the first kept three batches", n, want)
	}
}

// TestEngine_WarmReportsTheTextsItCouldNotEmbed: Warm embeds what it can and
// names the texts it could not, by their place among its arguments.
func TestEngine_WarmReportsTheTextsItCouldNotEmbed(t *testing.T) {
	r := newRig(t, engineCfg, nil, nil, nil)
	r.emb.failContaining("boom", errBoom)
	texts := []string{guideText, poisonText, "  ", diaryText, poisonText}
	err := r.e.warm(context.Background(), "1", "10", "1", texts)
	if !errors.Is(err, errBoom) || !strings.Contains(err.Error(), "text 2 of 5") || !strings.Contains(err.Error(), "text 5 of 5") {
		t.Fatalf("err = %v, want one wrapping the cause and naming texts 2 and 5 of 5", err)
	}
	if got := cached(t, r.cache, guideText, diaryText, poisonText); !got[guideText] || !got[diaryText] || got[poisonText] {
		t.Errorf("cached = %v, want the two good texts only", got)
	}
}

// TestEngine_VectorsThatAreNotNumbersAreNotCached: a vector with a NaN or an
// infinite value, or one of all zeros, compares with nothing. Cached, it would
// keep its tag from ever matching until the cache is wiped, so it is refused
// like a text that failed: a member is skipped and reported, and an item's
// own text fails the call.
func TestEngine_VectorsThatAreNotNumbersAreNotCached(t *testing.T) {
	bad := func(x float32) []float32 {
		v := make([]float32, 64)
		v[3] = x
		return v
	}
	for name, v := range map[string][]float32{
		"NaN":  bad(float32(math.NaN())),
		"+Inf": bad(float32(math.Inf(1))),
		"zero": bad(0),
	} {
		t.Run(name, func(t *testing.T) {
			cfg, rec := loggedCfg()
			r := newRig(t, cfg, breadWithRecipes("1"), nil, nil)
			r.emb.vecFor(croissantRecipes[1], v)
			ss := r.scores(t, "1", croissantText)
			pair := &fakeCatalog{tags: breadAndJapan("1"), members: map[string][]string{"bread": {croissantRecipes[0], croissantRecipes[2]}}}
			if want := newRig(t, cfg, pair, nil, nil).scores(t, "1", croissantText); !reflect.DeepEqual(ss, want) {
				t.Errorf("scores %+v, want those of the tag built from its two other members, %+v", ss, want)
			}
			if cached(t, r.cache, croissantRecipes[1])[croissantRecipes[1]] {
				t.Error("the member's vector was cached")
			}
			if n := len(rec.matching("WARN", "member", "tag=bread")); n != 1 {
				t.Errorf("the member was reported %d times, want once", n)
			}

			r.emb.vecFor(diaryText, v)
			if ss, err := r.e.scores(context.Background(), "1", "10", "1", diaryText); err == nil || ss != nil {
				t.Errorf("an item with such a vector scored %+v, %v; want an error", ss, err)
			}
			if cached(t, r.cache, diaryText)[diaryText] {
				t.Error("the item's vector was cached")
			}
		})
	}
}

// TestEngine_ACacheThatCannotWriteStillAnswers: the cache is derived data, and
// a write to it that fails (a read-only replica, a locked file) loses time,
// not the answer: the vectors just embedded, or just read, are used, and the
// failure is reported.
func TestEngine_ACacheThatCannotWriteStillAnswers(t *testing.T) {
	cfg, rec := loggedCfg()
	want := newRig(t, cfg, breadWithRecipes("1"), nil, nil).scores(t, "1", croissantText)

	cache := &faultyCache{EmbeddingCache: NewMemCache(), putErr: errors.New("read-only database")}
	r := newRig(t, cfg, breadWithRecipes("1"), cache, nil)
	if got := r.scores(t, "1", croissantText); !reflect.DeepEqual(got, want) {
		t.Errorf("with a Put that fails: %+v, want %+v", got, want)
	}
	if len(rec.matching("WARN", "not cached", "read-only database")) == 0 {
		t.Errorf("the failed Put was not reported; log:\n%s", strings.Join(rec.lines(), "\n"))
	}

	cache.putErr = nil
	newRig(t, cfg, breadWithRecipes("1"), cache, nil).scores(t, "1", croissantText) // fills the cache
	cache.getErr = fmt.Errorf("%w: locked", ErrCacheWrite)
	restarted := newRig(t, cfg, breadWithRecipes("1"), cache, nil)
	if got := restarted.scores(t, "1", croissantText); !reflect.DeepEqual(got, want) {
		t.Errorf("with a Get that could not refresh: %+v, want %+v", got, want)
	}
	if n := restarted.emb.total(); n != 0 {
		t.Errorf("embedded %d texts, want 0: the Get returned the vectors with its error", n)
	}
	if len(rec.matching("WARN", "last-used", "locked")) == 0 {
		t.Errorf("the failed refresh was not reported; log:\n%s", strings.Join(rec.lines(), "\n"))
	}

	cache.getErr = errors.New("connection refused")
	if _, err := restarted.e.scores(context.Background(), "1", "10", "1", "another text"); err == nil {
		t.Error("a Get that failed outright was taken for an answer")
	}
}

// TestEngine_MembersAreReadInGroups: a rebuild of many tags asks the catalog
// for the members of membersGroupSize tags at a time, each tag once.
func TestEngine_MembersAreReadInGroups(t *testing.T) {
	const n = 2*membersGroupSize + 22
	var tags []CatalogTag
	members := map[string][]string{}
	for i := range n {
		id := fmt.Sprintf("tag%03d", i)
		tags = append(tags, CatalogTag{id, fmt.Sprintf("topic / number %d", i), "1"})
		members[id] = []string{fmt.Sprintf("a member of tag number %d", i)}
	}
	cat := &fakeCatalog{tags: tags, members: members}
	r := newRig(t, engineCfg, cat, nil, nil)
	if ss := r.scores(t, "1", croissantText); len(ss) != n {
		t.Fatalf("%d scores, want %d", len(ss), n)
	}
	cat.mu.Lock()
	defer cat.mu.Unlock()
	if len(cat.memIDs) != 3 {
		t.Fatalf("%d Members calls, want 3", len(cat.memIDs))
	}
	seen := map[string]bool{}
	for _, ids := range cat.memIDs {
		if len(ids) > membersGroupSize {
			t.Errorf("a Members call asked for %d tags, want at most %d", len(ids), membersGroupSize)
		}
		for _, id := range ids {
			if seen[id] {
				t.Errorf("tag %s was asked for twice", id)
			}
			seen[id] = true
		}
	}
	if len(seen) != n {
		t.Errorf("the members of %d tags were read, want %d", len(seen), n)
	}
}

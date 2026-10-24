# Stateful autotag Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stateless `autotag.Tagger` with a library that owns tags, items and hand-made links, so an app only reports changes by id and text and asks for suggestions.

**Architecture:** One `Tagger` (embedder + `autotag.Store` + config) hands out `*Scope` handles for a tenant's space. Writes go through the `Store` and rebuild the affected tag vectors at once. Each tag carries a `Version` that every input change bumps, and a rebuild stores its result only if the version is unchanged, so a concurrent change is never overwritten. Tags left stale by a failed or overtaken rebuild, or by a model switch, are rebuilt by the next `Suggest`. Storage follows semsearch's layout: a `Store` interface, `MemStore` in the root package, and separate SQLite and Postgres packages over one shared internal implementation. They use plain tables, one set per index and generation, with no vector extension: a space has few tags, so they are scored in Go.

**Tech Stack:** Go 1.26, `database/sql`, `github.com/mattn/go-sqlite3` (tests), `github.com/go-bumbu/testdbs` (Postgres tests), `semsearch.Embedder`.

**Spec:** This conversation's pseudo code and decisions, summarized in **Design** below: autotag owns items and links, the stateless Tagger is removed, storage follows semsearch, and tenancy comes through a scope handle.

## Global Constraints

- Module `github.com/go-bumbu/langai`, `go 1.26.0`. Add no new module to `go.mod`.
- Defaults stay `DefaultNameMinScore = 0.46`, `DefaultMemberMinScore = 0.70` and `DefaultSmoothingK = 3`. `blend`, `centroid` and `tagCutoff` move over unchanged.
- Item text is embedded with `semsearch.RoleQuery` and tag names with `semsearch.RoleDocument`.
- Lint is `make lint` (golangci-lint v2). It enforces gocyclo ≤ 20, nestif ≤ 5, gosec, and dupl on non-test files. Every `//nolint` must name its linter and give a reason.
- Coverage must be ≥ 70 % per package. `make coverage` sets `TESTDBS_ALL=1`, so Postgres tests run and need Docker.
- SQL table names come only from `internal/tablename`: a hash of the index, and a generation matching `^[a-z0-9]{1,12}$`.
- The golden floors stay: names only 0.65, mixed examples 0.70, fits nothing untagged 0.70, two topics both tags 0.48. Mixed examples must beat names only.
- Git: one-line commit messages, with `git add` and `git commit` as separate steps. No Co-Authored-By line, and never push. Stage only the files a task names.
- The working tree already holds unrelated pending changes: `Makefile`, `README.md`, the deleted `cmd/semeval`, `semsearch/eval/doc.go`, `semsearch/yzma/models*.go` and the untracked `semsearch/eval/*_test.go` and `semsearch/eval/testdata/models.yaml`. Never stage any of them. Leave the README edits of Task 6 unstaged too, because that file already carries the user's pending edits.

## Review Focus

1. A hand tag lands while that tag is being rebuilt. The older rebuild must not overwrite it, and the next `Suggest` must see the new member. Pinned by `TestRebuild_KeepsAChangeThatLandsMidRebuild` (Task 4).
2. `Config.Model` changes on a filled store. Every vector is re-embedded from the stored text, and vectors of two models are never mixed. Pinned by `TestModelSwitch_ReembedsEverythingAndMixesNothing` (Task 4).
3. A write succeeds but its rebuild fails. The write sticks, the error is returned, and the next `Suggest` repairs the tag. Pinned by `TestRebuildFailure_KeepsTheChangeAndHealsOnSuggest` (Task 4).
4. A tag is deleted and its id reused. No old link comes back. Pinned by the suite's `DeleteTagDropsItsLinks` (Task 2) and `TestDeleteTag_DropsItAndForgetsItsLinks` (Task 4).
5. The same ids are used in two partitions or two indexes. Data never crosses between them, on any backend. Pinned by the suite's `ScopesAreIsolated` (Task 2) and `TestScopes_AreIsolated` (Task 4).

## Design

```go
tagger, err := autotag.New(emb, store, autotag.Config{Model: "embeddinggemma-300m"})
space := tagger.Scope("tenant-1", "space-7")
space.SetTag(ctx, "bread", "cooking / bread")   // create or rename
space.SetItem(ctx, "item-42", essence)          // create or edit; embeds only when text or model changed
space.TagByHand(ctx, "item-42", "bread")        // a person tagged it: teaches the tag
space.UntagByHand(ctx, "item-42", "bread")
ids, err := space.Suggest(ctx, "item-42")       // auto tags, best first; never stored as links
space.DeleteItem(ctx, "item-42"); space.DeleteTag(ctx, "bread")
space.Rebuild(ctx)                              // optional warm-up, e.g. after switching models
```

- A tag's vector is `blend(nameVec, current-model member vectors, k)`. Its cutoff is `storedCutoff(tagCutoff(...))`, which is any value `> 1`. A NaN or `+Inf` becomes `never = 2`, so a stored cutoff is always finite and fails closed.
- A tag is stale when `Built != Version || Model != cfg.Model`. `Suggest` and `Rebuild` make up to `rebuildPasses = 3` passes over the stale tags. A model switch needs two passes, because re-embedding a shared item makes already-rebuilt tags stale again.
- `Suggest` scores only tags with `Vec != nil && Model == cfg.Model`. It keeps those whose `score >= Cutoff`, with the NaN-safe test `!(score >= cut)` → skip. It orders them by score descending, with ties broken by tag id.

---

### Task 1: Share `tablename` between semsearch and autotag

**Files:**
- Move: `semsearch/internal/tablename/` → `internal/tablename/` (`git mv`)
- Modify: `internal/tablename/tablename.go` (package doc and `ForPrefix`), `internal/tablename/tablename_test.go`
- Modify imports: `semsearch/sqlitevec/sqlitevec.go:19`, `semsearch/pgvec/pgvec.go:17`, `semsearch/pgvec/pgvec_test.go:14`

**Interfaces:**
- Produces: `func ForPrefix(prefix, generation, index, suffix string) (string, error)`. It returns `<prefix>_<token>_<suffix>`, or `<prefix>_<generation>_<token>_<suffix>` with a generation. `For(g, i, s)` becomes `ForPrefix("ss", g, i, s)`, and its output must not change.

- [ ] **Step 1: Move the package and fix the three imports**

```bash
git mv semsearch/internal/tablename internal/tablename
```
Change the three import paths to `github.com/go-bumbu/langai/internal/tablename`.

- [ ] **Step 2: Write the failing test** in `internal/tablename/tablename_test.go`

```go
func TestForPrefix_NamesAnotherLibrarysTables(t *testing.T) {
	got, err := ForPrefix("autotag", "", "tenant-1", "tags")
	if err != nil || !strings.HasPrefix(got, "autotag_") || !strings.HasSuffix(got, "_tags") {
		t.Fatalf("ForPrefix = %q, %v", got, err)
	}
	ss, _ := For("", "tenant-1", "tags")
	if strings.TrimPrefix(got, "autotag_") != strings.TrimPrefix(ss, "ss_") {
		t.Fatalf("same index, same token: %q vs %q", got, ss)
	}
	if g, _ := ForPrefix("autotag", "g2", "tenant-1", "links"); !strings.HasPrefix(g, "autotag_g2_") {
		t.Fatalf("generation not after the prefix: %q", g)
	}
	if _, err := ForPrefix("autotag", "", "", "tags"); err == nil {
		t.Fatal("empty index must be an error")
	}
}
```

- [ ] **Step 3: Run it to see it fail**

Run: `go test ./internal/tablename/ -run TestForPrefix -v`
Expected: FAIL, `undefined: ForPrefix`.

- [ ] **Step 4: Implement `ForPrefix` and make `For` call it with `"ss"`.** Update the package doc so it names autotag's SQL stores as users too.

- [ ] **Step 5: Run the affected packages**

Run: `go test ./internal/tablename/ ./semsearch/...`
Expected: PASS. The existing `For` tests pin semsearch's names, which must not change.

- [ ] **Step 6: Commit**

```bash
git add internal/tablename semsearch/internal semsearch/sqlitevec/sqlitevec.go semsearch/pgvec/pgvec.go semsearch/pgvec/pgvec_test.go
git commit -m "refactor: share tablename between semsearch and autotag"
```

---

### Task 2: The `Store` contract, `MemStore` and a conformance suite

**Files:**
- Create: `autotag/store.go`, `autotag/memstore.go`, `autotag/storetest/storetest.go`, `autotag/memstore_test.go` (package `autotag_test`, to avoid an import cycle)

**Interfaces:**
- Produces: the following, in `autotag/store.go`. Copy the comments, because they are the contract every backend implements.

```go
var (
	ErrUnknownTag  = errors.New("autotag: unknown tag")
	ErrUnknownItem = errors.New("autotag: unknown item")
)

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

type TagVectors struct { NameVec, Vec []float32; Cutoff float64; Model string }

type ItemRecord struct { ID, Text string; Vec []float32; Model string }

type Member struct { ItemID string; Vec []float32; Model string }

// Store persists the tags, items and hand-made links of every scope. index is
// the hard isolation boundary (a tenant), partition a coordinate within it (a
// space); no scope ever sees another's data. Every method fails on an empty
// index. Vectors round-trip exactly; nil stays nil.
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
```

- Produces: `func NewMemStore() *MemStore`, which is safe for concurrent use and copies every slice it stores. Also `func storetest.Run(t *testing.T, newStore func(t *testing.T) autotag.Store)`.

- [ ] **Step 1: Write the suite** `storetest.Run`. It runs one `t.Run` per case, and each case calls `newStore(t)` itself. It uses `idx, part = "tenant", "space"` and the vectors `v1 = []float32{1.5, -2.25, 3.4028235e38}`, `v2 = []float32{1.1754944e-38, 0, -1}`.

| Subtest | Setup → assertions |
|---|---|
| `TagLifecycle` | `PutTag(t1,"a")` → `true`; `Tag` = `{ID:t1, Name:"a", Version:1, Built:0, NameVec:nil, Vec:nil}`; `PutTag(t1,"a")` → `false`, Version 1; after `SetTagVectors(…,1,{v1,v2,0.5,"m"})`, `PutTag(t1,"b")` → `true`, Version 2, `NameVec == nil`, Name "b"; `Tag(unknown)` → `ok == false`, nil error |
| `SetTagVectorsChecksTheVersion` | `SetTagVectors(t1, 1, {v1, v2, 0.5, "m"})` → `true`, record has `NameVec==v1, Vec==v2, Cutoff==0.5, Model=="m", Built==1`; `SetTagVectors(t1, 0, …)` → `false`, record unchanged; on unknown tag → `false`, nil error |
| `TagsAreOrderedByID` | `PutTag` "b","a","c" → `Tags` ids `[a b c]`; empty scope → empty, nil error |
| `DeleteTagDropsItsLinks` | tag t1, item i1, `Link(i1,t1)`; `DeleteTag(t1)` → `Tag` `!ok`; `PutTag(t1,"a")` again → `Members(t1)` empty; `PutItem(i1)` returns `[]` (no linked tag); `DeleteTag(unknown)` → nil |
| `ItemLifecycle` | `PutItem({i1,"x",v1,"m"})` → `[]`; `Item(i1)` equal; `PutItem({i1,"y",v2,"m"})` replaces; `Item(unknown)` → `!ok` |
| `LinkBumpsTheTagOnce` | `Link(i1,t1)` → `true`, Version 2; again → `false`, Version 2; `Members(t1)` = `[{i1, v1, "m"}]`; `Unlink` → `true`, Version 3; again → `false`, Version 3 |
| `LinkRejectsUnknownIDs` | `Link(nope,t1)` → `errors.Is(err, autotag.ErrUnknownItem)`; `Link(i1,nope)` → `ErrUnknownTag`; both unknown → `ErrUnknownItem`; `Unlink(nope,nope)` → `false`, nil |
| `PutItemBumpsItsTags` | i1 linked to t1,t2 (t3 not) → `PutItem(i1…)` returns `[t1 t2]`; t1,t2 Version +1; t3 unchanged |
| `DeleteItemBumpsItsTagsAndDropsLinks` | as above → `DeleteItem(i1)` returns `[t1 t2]`, both +1, `Members(t1)` empty, `Item(i1)` `!ok`; `DeleteItem(unknown)` → `nil, nil` |
| `MembersAreOrderedByItemID` | link i2 then i1 → `Members` ids `[i1 i2]` with their Vec and Model |
| `ScopesAreIsolated` | the same ids t1/i1 with different names/texts in `(tenant,space)`, `(tenant,other)`, `(other,space)` → each `Tag`/`Item`/`Tags` sees only its own; `Link(i1,t1)` in one scope leaves `Members(t1)` empty in the others; linking an item of one partition to a tag of another → `ErrUnknownItem` |
| `VectorsRoundTrip` | `v1`, `v2` and `[]float32{0.1}` come back `slices.Equal`; a never-built tag's `NameVec`/`Vec` are `nil` |
| `EmptyIndexIsAnError` | `Tag`, `Tags`, `PutTag`, `PutItem`, `Link`, `Members` with index `""` → non-nil error |
| `StoredSlicesAreCopies` | mutate `v` after `PutItem`/`SetTagVectors` → stored vectors unchanged |

`memstore_test.go`:

```go
func TestMemStore(t *testing.T) {
	storetest.Run(t, func(*testing.T) autotag.Store { return autotag.NewMemStore() })
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `go test ./autotag/ -run TestMemStore -v`
Expected: FAIL, `undefined: autotag.NewMemStore`.

- [ ] **Step 3: Implement `store.go` and `memstore.go`.** Use one `sync.Mutex` and `map[scopeKey]*memScope`, where `memScope{tags map[string]TagRecord; items map[string]ItemRecord; links map[string]map[string]bool}` with links keyed by tag id. Clone slices on write. An empty index returns `errors.New("autotag: empty index")`.

- [ ] **Step 4: Run it**

Run: `go test ./autotag/ -run TestMemStore -v`
Expected: PASS, all 14 subtests. The old Tagger still compiles, because none of these names clash with it.

- [ ] **Step 5: Commit**

```bash
git add autotag/store.go autotag/memstore.go autotag/memstore_test.go autotag/storetest/storetest.go
git commit -m "feat(autotag): add the Store contract, MemStore and a conformance suite"
```

---

### Task 3: Tag vector math in `autotag/vector.go`

**Files:**
- Create: `autotag/vector.go`, `autotag/vector_test.go`
- Modify: `autotag/autotag.go`. Move `blend`, `centroid` and `tagCutoff` out unchanged. `pickTags` stays until Task 4.
- Modify: `autotag/autotag_test.go`. Move `TestBlend_*`, the four `TestTagCutoff_*` and the helpers `length`, `unitVec`, `dot`, `randUnit` and `scoreVec` into `vector_test.go`, unchanged.

**Interfaces:**
- Consumes: `TagRecord` (Task 2).
- Produces: `func cosine(a, b []float32) float64` (0 for mismatched lengths or zero vectors), `const never = 2.0`, `func storedCutoff(c float64) float64`, `func rank(vec []float32, tags []TagRecord, model string) []string`, plus the moved `blend`, `centroid` and `tagCutoff`.

- [ ] **Step 1: Move the math and its tests.** Run `go test ./autotag/` and expect PASS before adding anything.

- [ ] **Step 2: Write the failing tests** in `vector_test.go`

```go
func TestStoredCutoff_KeepsScoresAndNeverStoresWhatNoScoreReaches(t *testing.T) {
	for in, want := range map[float64]float64{0.46: 0.46, 1: 1, 1.2: never, math.Inf(1): never, -0.1: -0.1} {
		if got := storedCutoff(in); got != want {
			t.Errorf("storedCutoff(%v) = %v, want %v", in, got, want)
		}
	}
	if got := storedCutoff(math.NaN()); got != never {
		t.Errorf("storedCutoff(NaN) = %v, want never", got)
	}
}

func TestCosine_IsZeroForMismatchedOrZeroVectors(t *testing.T) {
	if cosine([]float32{1, 0}, []float32{1, 0, 0}) != 0 || cosine([]float32{0, 0}, []float32{1, 0}) != 0 {
		t.Fatal("mismatched or zero vectors must score 0")
	}
	if got := cosine([]float32{2, 0}, []float32{1, 1}); math.Abs(got-math.Sqrt2/2) > 1e-6 {
		t.Fatalf("cosine = %v, want %v (not unit-length inputs)", got, math.Sqrt2/2)
	}
}
```

Write `TestRank_JudgesEachTagAgainstItsOwnCutoff` as a port of `TestPickTags_JudgesEachHitAgainstItsOwnTagsCutoff`, table-driven over `[]TagRecord`, with the document at `q := unitVec(1, 0, 0, 0)`. Each tag is `{ID, Vec: scoreVec(4, axis, score), Cutoff, Model: "m"}`, and `rank` is called with model `"m"`. The cases and expected ids are:
- No tags → `nil`.
- A tag below its cutoff does not hide the next ones → only the next.
- A score equal to the cutoff qualifies.
- A NaN cutoff picks nothing.
- A tag with `Vec == nil` is skipped, and so is one with `Model == "other"`.
- Seven tags scoring 0.9, 0.85 … 0.6 against a cutoff of 0.46 are all returned in that order.
- Two tags with equal scores come back ordered by id.

- [ ] **Step 3: Run them to see them fail**

Run: `go test ./autotag/ -run 'TestStoredCutoff|TestCosine|TestRank' -v`
Expected: FAIL, undefined `storedCutoff`, `cosine` and `rank`.

- [ ] **Step 4: Implement** `cosine`, `storedCutoff` (`!(c <= 1)` → `never`) and `rank` in `vector.go`. `rank` scores with `cosine`, applies the NaN-safe test, and sorts with `slices.SortStableFunc` by score descending and then by id.

- [ ] **Step 5: Run the package**

Run: `go test ./autotag/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add autotag/vector.go autotag/vector_test.go autotag/autotag.go autotag/autotag_test.go
git commit -m "refactor(autotag): move the tag vector math into vector.go and add rank"
```

---

### Task 4: The stateful Tagger

**Files:**
- Rewrite: `autotag/autotag.go`. It keeps the package doc, the defaults, `Config` (fields `Model`, `NameMinScore`, `MemberMinScore`, `SmoothingK`; `Collection` is gone), `Tagger`, `New`, `Scope` and its public methods, `embedOne` and `putItem`. Delete `Tag`, `TagStore`, the old `Scope` struct, `Embed`, `Refresh`, `Assign`, `searchTags`, `pickTags` and `hashInputs`.
- Create: `autotag/rebuild.go` with `stale`, `rebuild`, `refreshMembers`, `rebuildAll` and `rebuildStale`.
- Rewrite: `autotag/autotag_test.go`. Keep `countingEmbedder`, `stubEmbedder`, `vectorlessEmbedder`, `rustCooking` and `taughtAndPlain` (reshaped as below). Drop every test of the old API.
- Rewrite: `autotag/example_test.go`. Port: `autotag/golden_test.go`. Modify: `TODO.md`.

**Interfaces:**
- Consumes: Task 2 (`Store`, the records, `ErrUnknownTag`, `ErrUnknownItem`, `NewMemStore`) and Task 3 (`blend`, `tagCutoff`, `storedCutoff`, `rank`).
- Produces:

```go
func New(emb semsearch.Embedder, store Store, cfg Config) (*Tagger, error) // nil emb, nil store or empty Model: error
func (t *Tagger) Scope(index, partition string) *Scope
func (s *Scope) SetTag(ctx context.Context, tagID, name string) error       // empty id or blank name: error
func (s *Scope) DeleteTag(ctx context.Context, tagID string) error
func (s *Scope) SetItem(ctx context.Context, itemID, text string) error     // empty id or blank text: error
func (s *Scope) DeleteItem(ctx context.Context, itemID string) error
func (s *Scope) TagByHand(ctx context.Context, itemID, tagID string) error  // ErrUnknownItem / ErrUnknownTag
func (s *Scope) UntagByHand(ctx context.Context, itemID, tagID string) error
func (s *Scope) Suggest(ctx context.Context, itemID string) ([]string, error) // ErrUnknownItem
func (s *Scope) Rebuild(ctx context.Context) error
```

Algorithms the signatures leave open:

```text
SetTag:      changed := PutTag; if changed → rebuild(tag)
SetItem:     if stored Text == text && Model == cfg.Model → return; vec, linked := putItem; rebuildAll(linked)
TagByHand:   changed := Link; if changed → rebuild(tag)          (UntagByHand: same with Unlink)
DeleteItem:  linked := DeleteItem; rebuildAll(linked)            (DeleteTag: store only)
Suggest:     item (else ErrUnknownItem); if item.Model != cfg.Model → putItem(item.Text) for a fresh vec
             tags := rebuildStale(); return rank(vec, tags, cfg.Model)
rebuild(id): refreshMembers(id)            // re-embed members whose Model != cfg.Model (putItem; ignore linked)
             rec, ok := Tag(id); !ok → nil // deleted meanwhile
             members := Members(id)        // read AFTER rec: a change in between bumps Version, the CAS refuses
             nameVec := rec.NameVec, re-embedded (RoleDocument) if nil or rec.Model != cfg.Model
             vecs := member vectors with Model == cfg.Model
             SetTagVectors(id, rec.Version, {nameVec, blend(nameVec, vecs, k),
                           storedCutoff(tagCutoff(nameVec, vecs, k, nameMin, memberMin)), cfg.Model})
             // false = someone changed the tag meanwhile: they rebuild it, not an error
rebuildStale: up to rebuildPasses (3) × { tags := Tags(); no stale → return tags; rebuild each stale }; return Tags()
rebuildAll:  rebuild each id; stop at the first error
```

A rebuild error after a successful write is returned as `fmt.Errorf("autotag: rebuild tag %q: %w", id, err)`. The write itself stays, and the next `Suggest` repairs the tag.

- [ ] **Step 1: Record the golden baseline** on the current code, which needs the model cache in `~/.cache/semsearch-yzma`.

Run: `make test-models 2>&1 | grep -E 'names only|mixed|fits nothing|two topics'`
Expected: four values. The last measurement was 0.7071, 0.7515, 0.7500 and 0.5882. Note what this run prints, since that is the baseline for Step 8.

- [ ] **Step 2: Write the failing tests** in `autotag_test.go`. Unless a test says otherwise, use a MemStore and the scope `tg.Scope("1", "10")`. `mustNew` fails the test on error.

```go
func TestNew_RequiresEmbedderStoreAndModel(t *testing.T) {
	if _, err := New(nil, NewMemStore(), Config{Model: "m"}); err == nil {
		t.Error("nil embedder: New must fail")
	}
	if _, err := New(semsearch.FakeEmbedder{}, nil, Config{Model: "m"}); err == nil {
		t.Error("nil store: New must fail")
	}
	if _, err := New(semsearch.FakeEmbedder{}, NewMemStore(), Config{}); err == nil {
		t.Error("no model: New must fail")
	}
}
// TestNew_Defaults: Config{Model:"m"} → tg.cfg has 0.46, 0.70, 3.
```

Port the rest, each with its assertion:
- `TestSuggest_ColdStartTagsByName`: FakeEmbedder with `NameMinScore: 0.15`. `SetTag("1","rust")`, `SetTag("2","cooking")`, `SetItem("doc", rustCooking)` → `Suggest("doc")` returns `[1 2]`.
- `TestEmbedRoles_ItemsAreQueriesAndNamesAreDocuments`: the same flow through a `countingEmbedder` gives roles `[Document, Document, Query]`, and `Suggest` embeds nothing more (`calls == 3`).
- `TestSuggest_JudgesEachTagAgainstItsOwnCutoff`: `taughtAndPlain` now returns a stub whose texts are `"taught"`, `"plain"`, `"member"` → `m` and `"query"` → `q`. Items `m1`, `m2` and `m3` all have the text `"member"` and are hand-tagged `"1"`. The item `"q"` has the text `"query"`. → `[2]`.
- `TestSuggest_MemberMinScoreSetsTheCutoffOfATaughtTag`: the same fixture with `MemberMinScore: 0.5` → `[1 2]`.
- `TestSuggest_AssignsEveryTagThatClearsItsCutoff`: seven tags whose names each score 0.50 → all seven.
- `TestSuggest_NoTagsReturnsNothing`: `nil`, nil error.
- `TestSuggest_UnknownItem`: `errors.Is(err, ErrUnknownItem)`.
- `TestSuggest_NaNNameMinScoreAssignsNoTag`: port of the old NaN test → none.
- `TestTagByHand_UnknownItemOrTag`: `ErrUnknownItem` or `ErrUnknownTag`.
- `TestTagByHand_TeachesTheTagAndUntagForgets`: FakeEmbedder with 0.3/0.5, using the Example's croissant texts. `[]` → hand-tag three recipes → `[bread]` → untag all three → `[]`.
- `TestSetItem_UnchangedTextEmbedsNothing`: with a counting embedder, the same text twice gives `calls == 1`.
- `TestSetItem_EditRebuildsItsHandTags`: item linked to `"1"`, then its text changes → the tag record has `Built == Version` and a `Vec` different from before.
- `TestSetTag_RenameReembedsTheName`: `"rust"` → 1 call. `"rust"` again → no call. `"go"` → 1 call. An item with the text `"go"` does not get the tag before the rename and does after.
- `TestDeleteItem_RebuildsItsTags`: members a and b → `DeleteItem(a)` → `Members == [b]`, `Built == Version`, and `Vec` equals `blend(NameVec, [vec_b], 3)`.
- `TestDeleteTag_DropsItAndForgetsItsLinks`: after `DeleteTag("1")`, `Suggest` excludes it. `SetTag("1", sameName)` → `Members` is empty.
- `TestScopes_AreIsolated`: tag `"1"` is `"rust"` in scope (1,10) and `"cooking"` in scope (1,11). The same item `"doc"` with the text `"rust programming language"` → `[1]` in (1,10) and `[]` in (1,11).
- `TestEmbedderErrors_FailTheWrite`: `vectorlessEmbedder` → `SetItem` and `SetTag` both return an error containing `"returned 0 vectors"`.

The three Review Focus tests:

```go
// racingStore links item "b" to tag "1" the first time a rebuild of "1" tries
// to store its vectors once raced is false, so that rebuild was computed from
// an older Version.
type racingStore struct {
	*MemStore
	raced bool
}

func (r *racingStore) SetTagVectors(ctx context.Context, i, p, id string, v int64, tv TagVectors) (bool, error) {
	if !r.raced && id == "1" {
		r.raced = true
		if _, err := r.MemStore.Link(ctx, i, p, "b", "1"); err != nil {
			return false, err
		}
	}
	return r.MemStore.SetTagVectors(ctx, i, p, id, v, tv)
}

func TestRebuild_KeepsAChangeThatLandsMidRebuild(t *testing.T) {
	// setup with store := &racingStore{MemStore: NewMemStore(), raced: true}:
	// SetTag("1", …), SetItem("a", …), SetItem("b", …); then store.raced = false
	if err := space.TagByHand(ctx, "a", "1"); err != nil {
		t.Fatal(err)
	}
	rec, _, _ := store.Tag(ctx, "1", "10", "1")
	if rec.Built == rec.Version {
		t.Fatal("the rebuild that missed b must not have been stored")
	}
	if _, err := space.Suggest(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	rec, _, _ = store.Tag(ctx, "1", "10", "1")
	ms, _ := store.Members(ctx, "1", "10", "1")
	if rec.Built != rec.Version || len(ms) != 2 ||
		!slices.Equal(rec.Vec, blend(rec.NameVec, [][]float32{ms[0].Vec, ms[1].Vec}, 3)) {
		t.Fatalf("Suggest must rebuild the tag from both members: %+v", rec)
	}
}
```

- `TestModelSwitch_ReembedsEverythingAndMixesNothing`: fill a store under `Model: "a"` with FakeEmbedder: tags `t1` and `t2`, items `doc`, `x` and `y`; `x` hand-tagged with both tags and `y` with `t1`. Choose texts so `Suggest("doc")` returns a non-empty list, and note it. Then call `New(&countingEmbedder{inner: FakeEmbedder{}}, sameStore, Config{Model: "b"})`. Its `Suggest("doc")` must return the same ids. Afterwards every `Tags()` record has `Model == "b"` and `Built == Version`, and `doc`, `x` and `y` have `Model == "b"`. `calls == 5`: doc, x, y and the two names, each embedded once. Then `Rebuild(ctx)` makes no further calls.
- `TestRebuildFailure_KeepsTheChangeAndHealsOnSuggest`: a `failingStore{*MemStore; fail bool}` returns `errStoreDown` from `SetTagVectors` while `fail` is set. Use FakeEmbedder with `NameMinScore: 0.3, MemberMinScore: 0.5`, tag `"1"` named `"cooking / bread"`, and item `"a"` with the text `"Baking bread at home: a cooking guide"`, all set while `fail = false`. With `fail = true`, `TagByHand("a","1")` → `errors.Is(err, errStoreDown)`, `Members("1") == [a]` and `Built != Version`. With `fail = false`, `Suggest("a")` → the tag has `Built == Version` and the result contains `"1"`.

- [ ] **Step 3: Run them to see them fail**

Run: `go test ./autotag/`
Expected: compile errors for the new API.

- [ ] **Step 4: Implement** `autotag.go` and `rebuild.go` as described above. `embedOne` keeps its guard (`"autotag: embedder returned %d vectors for one %s"`). `putItem(ctx, itemID, text) (vec []float32, linked []string, err error)` embeds with `RoleQuery` and calls `PutItem` under `cfg.Model`.

- [ ] **Step 5: Rewrite `example_test.go`** with the new API. Use the same FakeEmbedder config (`Model: "fake", NameMinScore: 0.3, MemberMinScore: 0.5`), tags and texts. `suggest(id, doc)` calls `SetItem` and then `Suggest`. The croissant recipes become items that are tagged by hand. The `// Output:` block stays exactly as it is now (five lines).

- [ ] **Step 6: Port `golden_test.go`.** `goldenWorld` keeps `tagIDs`, `nums`, `byTag` and `allTags`, and adds:
  - `space *Scope`
  - `names map[string]string`: id → path
  - `present map[string]bool`: tags currently in the scope
  - `links map[string]map[string]bool`: a mirror of the hand links

  `newGoldenWorld` runs `SetTag` for every tag and `SetItem(doc.ID, doc.Essence())` for every document. `setMembers(skip, capOf)` computes the same member sets as today, but only for present tags, and applies the difference with `TagByHand`/`UntagByHand`. `assign(id)` becomes `Suggest(id)`.
  - `untagged` deletes the document's own tags (`DeleteTag`, then clears their `links` and `present`), suggests, and restores them with `SetTag`.
  - `bothTags` uses an item `"page"` set to each stitched essence, and runs `DeleteItem("page")` at the end.

  The floors and the "mixed beats names only" check stay as they are.

- [ ] **Step 7: Run the unit tests**

Run: `go test ./autotag/... -cover`
Expected: PASS, coverage ≥ 70 %.

- [ ] **Step 8: Run the real-model golden test**

Run: `make test-models 2>&1 | grep -E 'names only|mixed|fits nothing|two topics|PASS|FAIL'`
Expected: PASS, with the four values equal to the Step 1 baseline. The math is unchanged; only the order members are summed in differs. Any difference beyond the 4th decimal is a bug to investigate, not a new baseline.

- [ ] **Step 9: Update `TODO.md`.** Delete the `semsearch` item "let a Store list the doc ids it holds". It was about ghost tag vectors, which no longer exist.

- [ ] **Step 10: Commit**

```bash
git add autotag/autotag.go autotag/rebuild.go autotag/autotag_test.go autotag/example_test.go autotag/golden_test.go TODO.md
git commit -m "feat(autotag)!: make the Tagger own tags, items and hand-made links"
```

---

### Task 5: SQL store core and the SQLite backend

**Files:**
- Create: `autotag/internal/sqlstore/sqlstore.go` (Dialect, Store, New, schema, codec, rebind), `tags.go` (tag methods) and `items.go` (item, link and member methods), plus `autotag/internal/sqlstore/sqlstore_test.go`
- Create: `autotag/sqlitestore/sqlitestore.go`, `autotag/sqlitestore/sqlitestore_test.go`

**Interfaces:**
- Consumes: `autotag.Store` and its records (Task 2), `storetest.Run` (Task 2), and `tablename.ForPrefix` (Task 1).
- Produces:

```go
package sqlstore
type Dialect struct {
	Blob, Float, Int string // vector, cutoff and version column types
	Numbered         bool   // $1, $2… placeholders instead of ?
	ForUpdate        string // row lock for reads inside a write transaction
}
var SQLite   = Dialect{Blob: "BLOB", Float: "REAL", Int: "INTEGER"}
var Postgres = Dialect{Blob: "BYTEA", Float: "DOUBLE PRECISION", Int: "BIGINT", Numbered: true, ForUpdate: " FOR UPDATE"}
func New(db *sql.DB, d Dialect, generation string) (*Store, error) // nil db or bad generation: error
// *Store implements autotag.Store

package sqlitestore
func New(ctx context.Context, db *sql.DB, opts ...Option) (autotag.Store, error) // checks SELECT sqlite_version()
type Option func(*options)
func WithGeneration(generation string) Option
```

Schema per index, created lazily by `ensure`, which uses a mutex and a `created` map as `sqlitevec` does. Table names are `tablename.ForPrefix("autotag", generation, index, "tags"|"items"|"links")`.

```sql
CREATE TABLE IF NOT EXISTS {tags} (partition TEXT NOT NULL, tag_id TEXT NOT NULL, name TEXT NOT NULL,
  name_vec {Blob}, vec {Blob}, cutoff {Float} NOT NULL DEFAULT 0, model TEXT NOT NULL DEFAULT '',
  version {Int} NOT NULL, built {Int} NOT NULL DEFAULT 0, PRIMARY KEY (partition, tag_id))
CREATE TABLE IF NOT EXISTS {items} (partition TEXT NOT NULL, item_id TEXT NOT NULL, text TEXT NOT NULL,
  vec {Blob} NOT NULL, model TEXT NOT NULL, PRIMARY KEY (partition, item_id))
CREATE TABLE IF NOT EXISTS {links} (partition TEXT NOT NULL, item_id TEXT NOT NULL, tag_id TEXT NOT NULL,
  PRIMARY KEY (partition, item_id, tag_id))
CREATE INDEX IF NOT EXISTS {links}_tag ON {links}(partition, tag_id)
```

Statements the contract leaves open:
- `PutTag` needs no transaction. It runs `INSERT … ON CONFLICT (partition, tag_id) DO NOTHING`; one row inserted means `true`. Otherwise it runs `UPDATE … SET name = ?, name_vec = NULL, version = version + 1 WHERE partition = ? AND tag_id = ? AND name <> ?`, and one row updated means `true`.
- `SetTagVectors` is one `UPDATE … SET name_vec, vec, cutoff, model, built = ? WHERE partition = ? AND tag_id = ? AND version = ?`. It reports `RowsAffected() == 1`.
- `Link` is a transaction. It reads the item and then the tag `{ForUpdate}`; a missing one gives `ErrUnknownItem` or `ErrUnknownTag`. Then `INSERT … ON CONFLICT DO NOTHING`. If a row was inserted, it bumps the tag.
- `Unlink`, `PutItem`, `DeleteItem` and `DeleteTag` each run in one transaction. The linked-tag bump is `UPDATE {tags} SET version = version + 1 WHERE partition = ? AND tag_id IN (SELECT tag_id FROM {links} WHERE partition = ? AND item_id = ?)`. Every id list is read with `ORDER BY`.
- Vectors are stored as little-endian float32 bytes: `encode(nil) == nil`, and `decode` maps nil or empty to nil and fails when `len%4 != 0`.
- Queries are written with `?`, and `rebind` numbers them when `Numbered` is set.
- Table names go into SQL with `fmt.Sprintf`, carrying the same `//nolint:gosec` explanation as `sqlitevec` and `pgvec`: the table name is a hex hash of the index, and every value is a bound parameter.

- [ ] **Step 1: Write the failing tests**

`sqlstore_test.go`:

```go
func TestSQLiteConformance(t *testing.T) {
	storetest.Run(t, func(t *testing.T) autotag.Store {
		s, err := New(openSQLite(t), SQLite, "")
		if err != nil {
			t.Fatal(err)
		}
		return s
	})
}
// openSQLite: sql.Open("sqlite3", filepath.Join(t.TempDir(), "autotag.db")), closed on cleanup;
// the test file blank-imports github.com/mattn/go-sqlite3.
```

Add the following tests too:
- `TestCodec_RoundTripsAndRejectsTornBlobs`: `decode(encode(v1))` equals `v1`; `decode(nil)` is `nil`; `decode([]byte{1,2,3})` returns an error.
- `TestRebind`: `rebind("a = ? AND b = ?")` gives `"a = $1 AND b = $2"`.
- `TestNew_RejectsNilDBAndBadGeneration`.
- `TestGenerationsAreIsolated`: the same db with generations `""` and `"g2"`; a tag put in one is not seen by the other.

`sqlitestore_test.go`:
- `TestConformance`: `storetest.Run` through `sqlitestore.New(ctx, db)`.
- `TestNew_RejectsANilHandle`: `New(ctx, nil)` returns an error.
- `TestWithGeneration_RejectsABadToken`: `WithGeneration("G-2")` returns an error.

- [ ] **Step 2: Run them to see them fail**

Run: `go test ./autotag/internal/sqlstore/ ./autotag/sqlitestore/`
Expected: compile errors, the packages don't exist yet.

- [ ] **Step 3: Implement** the three `sqlstore` files and `sqlitestore.go`. The package doc tells users to open the database with `_txlock=immediate&_busy_timeout=5000` when several goroutines write, because SQLite allows one writer at a time.

- [ ] **Step 4: Run them**

Run: `go test ./autotag/... -cover`
Expected: PASS, `sqlstore` and `sqlitestore` each ≥ 70 %.

- [ ] **Step 5: Commit**

```bash
git add autotag/internal/sqlstore autotag/sqlitestore
git commit -m "feat(autotag): add the SQL store core and the SQLite backend"
```

---

### Task 6: The Postgres backend, docs and full verification

**Files:**
- Create: `autotag/pgstore/pgstore.go`, `autotag/pgstore/pgstore_test.go`
- Modify: `README.md`, the package table row for `autotag` (line 16) and the "Automatic tagging" section (lines 87–125). Leave it unstaged, as Global Constraints says.

**Interfaces:**
- Consumes: `sqlstore.New(db, sqlstore.Postgres, generation)` (Task 5) and `storetest.Run` (Task 2).
- Produces: `func New(ctx context.Context, db *sql.DB, opts ...Option) (autotag.Store, error)` in `pgstore`. It checks the database with `SELECT current_setting('server_version_num')` and offers `WithGeneration(generation string) Option`.

- [ ] **Step 1: Write the failing test.** `pgstore_test.go` mirrors `semsearch/pgvec/pgvec_test.go:17-51`: a `TestMain` with `testdbs.InitCustomDbs(nil, []testdbs.TargetDb{testdbs.NewPostgres("pgvector/pgvector:pg18")})`, and `newStore(t) autotag.Store` skipping when `len(testdbs.DBs()) == 0` and using `dbs[0].ConnDbName(t.Name()).DB()`. Then:

```go
func TestConformance(t *testing.T) { storetest.Run(t, newStore) }
// TestNew_RejectsANilHandle: New(ctx, nil) → error.
// TestConcurrentLinksAllLand: tag t1 and items i1…i8; 8 goroutines each Link(i_n, "t1") →
// no error, Members(t1) has 8 entries and t1's Version == 9.
```

- [ ] **Step 2: Run it to see it fail**

Run: `TESTDBS_ALL=1 go test ./autotag/pgstore/`
Expected: compile error, `New` is undefined.

- [ ] **Step 3: Implement `pgstore.go`.** Its package doc says it needs plain Postgres, with no extension.

- [ ] **Step 4: Run it**

Run: `TESTDBS_ALL=1 go test ./autotag/pgstore/ -cover -v`
Expected: PASS, coverage ≥ 70 %.

- [ ] **Step 5: Update the README.** Replace the usage snippet with the Design snippet. Add a "Storage" paragraph covering `MemStore`, `sqlitestore`, `pgstore` and `WithGeneration`. Keep the "Its vector" and "Its cutoff" bullets. Replace the "Ghosts" bullet with "changes rebuild the affected tags right away; a failed or overtaken rebuild leaves the tag stale and the next `Suggest` rebuilds it". Add a "Switching models" bullet: stored text is re-embedded lazily, and `Scope.Rebuild` warms a space up. Update the table row on line 16.

- [ ] **Step 6: Run the full verification**

Run: `make lint && make coverage && make test-models`
Expected: no lint findings, every package ✅ ≥ 70 %, and the golden test PASS with the Task 4 baseline values.

- [ ] **Step 7: Commit**

```bash
git add autotag/pgstore
git commit -m "feat(autotag): add the Postgres backend"
```

Then report to the user that `README.md` is updated but unstaged.

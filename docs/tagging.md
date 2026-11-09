# Tagging

Suggesting tags from the ones an application already has (`tagging.Tagger`) and writing new ones with a chat model (`tagging.Generator`). Changes between versions are in [upgrading.md](upgrading.md).

## Tagging flows

1. **Suggest** (existing tags)
   1. The app calls `Suggest` with the text and the space's revision.
   2. The library embeds the text, using the cache with the text's hash as the key.
   3. The library keeps an in-memory index of tag vectors. When the revision changes, it
      rebuilds the changed tags from the `Catalog` and the cached embeddings. A member
      text the embedder cannot take is left out, logged and listed by `Incomplete`; an
      embedder that is down fails the rebuild instead.
   4. A tag vector is its name blended with the average of its members' vectors, and
      each tag has its own cutoff.
   5. The library compares the text's vector with every tag vector by cosine.
   6. Tags that reach their cutoff are picked, and on one path only the better fit is
      kept.
   7. The library returns the picked tag ids.
2. **Scores** (the reasons behind Suggest)
   1. The app calls `Scores` with the text and the space's revision.
   2. It runs the same steps as `Suggest`.
   3. It returns every tag with its score, its cutoff and the tag that shadowed it, if
      any. The ones not picked are included.
3. **Warm** (pre-compute)
   1. The app calls `Warm` with the revision and some texts, for example on item save or
      after a model switch.
   2. The library rebuilds the space if the revision changed.
   3. It embeds and caches any texts the cache doesn't have yet, a batch at a time, so a
      cancelled Warm keeps its progress.
4. **Generate** (new tags)
   1. The app calls `Generate` with the text and a minimum score.
   2. The library cuts the text to `MaxBytes` (64 KB by default).
   3. A text of up to `ChunkBytes` (2 KB by default) goes to an LLM in one call. A longer
      one is read in windows of about 6 KB, one LLM call each, with a short first line (the
      title) repeated at the head of every window after the first, and the names of all
      the windows are merged.
   4. The LLM writes up to `Candidates` tags (10 by default) per call, either as short
      words or as tree paths, optionally extending the app's existing paths. A call that
      fails or replies with nothing is made once more; a window that fails twice is left
      out and reported with `ErrPartial`. A closed model, or one that fails on the first
      two windows, is not asked about the rest.
   5. The library embeds the text (in 2 KB chunks when it is long) and each candidate
      tag. Nothing is cached.
   6. Each candidate is scored by the cosine between its vector and the text's vector;
      for a long text, by its best cosine over the chunks.
   7. Candidates below the minimum score are dropped, and a long text's are capped at
      `Candidates`.
   8. The library returns the remaining tags with their scores, best first.
5. **Cache maintenance**
   1. The app calls `Tagger.Prune` on a schedule, with a cutoff and the spaces still in
      use.
   2. The library reads each space's `Catalog` and marks the embeddings of its tag names
      and members as used now. Nothing is embedded.
   3. It deletes the embeddings not used since the cutoff, across all tenants.
   4. The app calls `DeleteIndex` on the cache to delete all embeddings of one tenant,
      for example when the tenant is removed.

There's no "learn" call. The Tagger learns only when the app changes the revision.

## Automatic tagging

```go
// catalog is your tagging.Catalog: Tags lists a space's tags (id, name, revision),
// Members the texts of the items tagged with some of them by hand
cache, err := sqlitecache.New(ctx, db)
tagger, err := tagging.NewTagger(emb, catalog, cache, tagging.TaggerConfig{Model: emb.Model(), Logger: logger})
defer tagger.Close() // cancels and waits for what is in flight; later calls wrap sense.ErrClosed
space := tagger.Scope(tagging.Space{Index: "tenant-1", Partition: "space-7"})
ids, err := space.Suggest(ctx, rev, essence)    // auto tags, best first; never stored as links
scores, err := space.Scores(ctx, rev, essence)  // every tag's score and cutoff: why a tag was or was not picked
err = space.Warm(ctx, rev, essences...)         // optional warm-up, e.g. after switching models
```

Your application owns what the Tagger learns from: its tags, its items (bookmarks, notes,
pages) and the tags people put on items by hand. It never tells the Tagger about a
change; the Tagger reads the `Catalog` when the space's revision `rev` changes. `rev` is an
opaque string the application changes whenever a tag, a hand-made link or a tagged item
of the space does (a counter on the space's row will do), and each tag's own revision
changes whenever its members do: a link added or removed, a member's text edited, a
member deleted. At the revision it was last built at, a Suggest never calls the catalog.
A space is named by one `tagging.Space{Index, Partition}` wherever the Tagger takes one,
`Scope` and `Prune` alike, and `space.Space()` gives a scope's back. As in search, `Index`
is the hard isolation boundary (a tenant) and `Partition` a space within it; scopes never
see each other's tags, and no embedding is read on behalf of another index. Use a tag's
full path as its name (`colors / warm / red` says
more than `red`). `TaggerConfig.Model` is the identity of the embedding model, its
`Model()` (or its spec's `ModelID()`), and every cached vector is kept under the model
it came from.

**Long items.** An embedder reads a window of tokens and silently drops the rest: the
default yzma model reads 512 tokens, some 2 KB of prose, so only the start of a longer item
counts. `TaggerConfig.ChunkBytes` (off by default) embeds a longer item in pieces of about
equal size cut at whitespace (`tagging.Chunk`), each cached on its own; set it a little
under the window, 2000 for the default model. An item of three pieces or more scores
against a tag by the mean of its two best pieces, so a topic one section is about counts
in full and one paragraph alone does not carry a tag; an item of two pieces by the best of
its two halves and their mean, since a short page's header and body land in different
halves. A long member teaches its tag with its piece closest to the tag's name, not with
the mean of all of them. On the 24 long golden documents (7 to 31 KB, 93 side topics past
the first 2 KB), against a catalog of 158 name-only tags made of the topic labels of all
of them, every main topic and 0.90 of the side topics were found, against 0.65 of the side
topics without `ChunkBytes`, at a precision of 0.80 (0.86 without) and with one passing
mention of 54 tagged in both (`TestTaggerLongTags`). In the experiment that chose the rule,
scored by exact label, the mean of the pieces found 0.75 of the side topics where this rule
found 0.89, and the best piece alone 0.96, but that let single paragraphs pull in the tags
of other documents (5.0 a document, against 3.5). The two cutoffs
hold unchanged: pieces score on the scale of short items. Changing `ChunkBytes` embeds only
the pieces the cache does not hold.

**Storage.** The Tagger keeps only derived data, and losing it costs time, never
correctness. An `EmbeddingCache` holds every embedding, keyed by index, model, role and the
hash of the text, so an edited text or another model simply misses and is embedded again.
`tagging.NewMemCache()` keeps it in memory, for tests and tiny corpora;
`tagging/sqlitecache` and `tagging/pgcache` keep it in one plain table of a SQLite or
Postgres database that you open and pass to `New(ctx, db)`; your application imports the
driver. An index or a model name with a NUL byte or invalid UTF-8 is refused by all three,
since Postgres cannot store one. Processes that start together may all call `New`: the table
is created under a lock. SQLite allows one writer at a time: when several goroutines write,
open it with `?_busy_timeout=5000&_txlock=immediate`, so a writer waits for its turn instead
of failing with "database is locked". `tagging/cachetest` is the conformance suite for a
cache of your own; it runs on a clock it moves by hand, so your cache must be able to take
its last-used times from the `now` function the suite passes. Each space's tag vectors and
cutoffs live in a `TagIndex`, by default a `MemIndex`
(an LRU of spaces under a 256 MB budget; `WithTagIndex` sets another), rebuilt from the
cache, never from the model, after a restart or an eviction. A space has few tags, so the
Tagger scores them all in Go.

**Pruning.** `tagger.Prune(ctx, unusedSince, spaces...)` removes the embeddings last used
before `unusedSince`, across every tenant: the texts of items edited or deleted since, and
the rows of a model you no longer use. Run it on a schedule, and call it instead of the
cache's own `Prune`:

```go
spaces := []tagging.Space{{Index: "tenant-1", Partition: "space-7"}} // every space still in use
removed, err := tagger.Prune(ctx, time.Now().AddDate(0, 0, -30), spaces...)
```

A tag that does not change is built once and never read from the cache again, so the
embeddings of its name and members look unused however often the tag is suggested. `Prune`
therefore first reads the `Catalog` of each space you pass and marks what its tags are built
from as used (`EmbeddingCache.Touch`), and only then deletes. A space you leave out loses
its old embeddings and embeds them again at its next rebuild: time, never correctness. If a
space's catalog fails, nothing is deleted. A read rewrites a last-used time only once it
is 24 hours stale (`tagging.CacheRefreshAfter`), so the recorded time can be up to a day
older than the last read, and a cutoff of age A keeps for certain only what was read in
the last A minus 24 hours: 30 days removes what was unread for 30 days and may remove
what was last read 29 days ago. The cutoff must therefore be at least two days old
(`tagging.MinPruneAge`), which protects the last day's reads; `Prune` refuses a more
recent one with an error, since it could delete embeddings read minutes ago.
When Taggers of several models that are all in use share one cache, call `Scope.Touch` on
the other Taggers' spaces first, since a `Prune` keeps only its own model's embeddings.

How a tag is judged:

- **Its vector** blends an embedding of its name with the mean of its members'
  vectors, weighted by member count: `w = n/(n+SmoothingK)`. A brand-new tag tags by
  its name alone, and shifts toward "tag by example" as members accumulate.
- **Its cutoff** mixes `NameMinScore` (default 0.42) and `MemberMinScore` (default
  0.58) by the same weight. Documents resemble each other far more than they resemble a
  short tag path, so each part is judged on its own scale: scoring the blended vector
  against the cutoff is exactly
  `(1-w)·(cos(name) - NameMinScore) + w·(mean cos(member) - MemberMinScore) >= 0`.
  The mean of a document's cosines to the members stays put as members are added,
  where its cosine to their centroid rises with every one, so one `MemberMinScore`
  holds for a tag with two examples and for one with fifty.
- **Every tag that clears its own cutoff is suggested**, best score first, with no limit
  on how many, but along a path of the tag tree (names split on `/`: `dev / go` is a
  child of `dev`) the better fit wins. Every ancestor meets every descendant by margin
  (score minus cutoff): a descendant at least as wide shadows the ancestor, and a
  narrower one is shadowed by it. A Go page gets `dev / go` and not also `dev`, which
  the path implies; a general programming page gets `dev` and not the `dev / web` it
  clears by less; two children that both beat their parent are both suggested.
  `TagScore.ShadowedBy` names the winner; `TaggerConfig.AllPathLevels` turns the rule
  off. A suggestion is never stored as a link: only the application's hand-made links
  teach a tag.
- **A new revision rebuilds only what moved.** The Tagger reads the catalog's tags,
  rebuilds from the cache the ones whose revision, name, model or settings changed, with
  one `Members` call per 64 tags, and drops the ones gone; the callers that ask at one
  revision together share one rebuild. So a bulk import costs one rebuild per changed
  tag, at the next call. The rebuilds of one space run one after the other: a caller at
  another revision (a second instance of your application that has not seen the latest
  change yet) waits, then rebuilds only if the space is still not at its revision. While
  two instances disagree on the revision, each call of one re-reads the catalog's tags
  after a call of the other, so let them converge.
- **A rebuild belongs to its callers.** It runs on a goroutine of the Tagger's that the
  waiting callers share. A caller whose context ends returns at once and the rebuild goes
  on for the others. When the last one has gone it goes on for
  `TaggerConfig.RebuildGrace` (30 seconds by default): a caller that asks about the space
  within that time joins the same rebuild, so a rebuild that takes longer than your
  request deadline still completes, for the requests after the ones that gave up on it,
  instead of being started over by every request and never finishing. With nobody back
  when the time is up, the rebuild's context is cancelled, so a hung embedder is not left
  running; a negative `RebuildGrace` cancels it as soon as its last caller has gone. A
  rebuild that hangs is joined by the callers that come within the grace period, each
  waiting until its own deadline, so give an embedder that can hang a timeout of its
  own. A panic of your `Catalog` or `Embedder` in a rebuild
  does not end the process: the callers get an error wrapping `ErrNotRebuilt` with the
  panic's value, and the stack goes to the `Logger`. `tagger.Close()` cancels the calls
  and rebuilds in flight, waits for them, and makes every later call return an error
  wrapping `sense.ErrClosed`.
- **A failure keeps what was done.** Texts go to the embedder 32 at a time and each batch
  is cached as it comes back, so a rebuild or a `Warm` that fails or is cancelled loses at
  most one batch, and the next call embeds only what is missing. A catalog failure, an
  embedder that is down, or an embedder failure on a tag's *name* (the error then names
  the tag) fails the call with no tags and an error wrapping `tagging.ErrNotRebuilt`; the
  index stays as it was, so the next call tries again.
- **One bad member does not block a space.** When a batch fails, its texts are embedded
  again one at a time. A member's text the embedder still fails on, or answers with a
  vector of NaN, infinite or all-zero values, is left out: its tag is built from its name
  and its other members, nothing is cached for it, and the Tagger tries it again at the
  space's next revision. `space.Incomplete(ctx)` lists the ids of the tags built that
  way, and `TaggerConfig.Logger` (a `*slog.Logger`, nil discards) gets a
  warning for each member, with the index, the partition, the tag's id and the SHA-256 and
  length of the text, so you can find the item; the text itself is not logged. `Warm`
  returns an error naming the texts it could not embed by their position, after
  embedding the rest; `Suggest` and `Scores` fail when their own text cannot be embedded.
- **An embedder that is down is not a bad member.** A member is left out only on evidence
  that the embedder works: it answered with a useless vector, or an `Embed` after the one
  that failed succeeded, of a later text or, when there is none, of the tag's name
  embedded once more as a canary. When that fails too, or the embedder's error wraps
  `sense.ErrClosed`, `context.Canceled` or `context.DeadlineExceeded` (a closed model, a
  request timeout of its own), the rebuild fails with an error wrapping
  `tagging.ErrNotRebuilt` and the cause, and nothing is recorded: the next call tries
  again, so an outage never leaves tags built from their names alone. After three texts in
  a row have failed the Tagger checks the canary before going on, so a dead embedder costs
  a handful of calls, not one per text. `Warm` ends the same way, with the embedder's
  error.
- **A cache that cannot be written still answers.** On a read-only replica or a locked
  SQLite file the vectors are read but their last-used time cannot be refreshed, and new
  embeddings cannot be stored. Neither fails a call: the Tagger uses the vectors it has
  and logs a warning.
- **One configuration per shared index:** Taggers that share a `TagIndex` (through
  `WithTagIndex`) must use the same `Model`, `NameMinScore`, `MemberMinScore`,
  `SmoothingK` and `ChunkBytes` (`AllPathLevels` may differ), or each takes the other's
  tags for stale and they rebuild at every call. Taggers of different models can share
  a cache: the model is part of every key.
- **Switching models** needs no migration: after you change `TaggerConfig.Model` the new
  model misses the cache and embeds what it needs, lazily, never mixing two models'
  vectors. `Scope.Warm` with the corpus's texts does it ahead of time, and `Tagger.Prune`
  later drops the old model's rows. Give every model its own `Model` name. If another
  model does end up under an old name, a cached vector that is not as long as the
  embedder's `Dimensions()` counts as a miss and is embedded again and overwritten. Where
  the lengths cannot be told apart beforehand (an embedder whose `Dimensions()` is 0, or a
  shared `TagIndex` built by the other model), `Suggest` and `Scores` return an error
  naming the tag rather than a score of 0. Two models of the same dimensions under one
  name cannot be told apart at all.
- **Rejected alternatives:** kNN voting on its own (it never abstains); a kNN agreement
  check (it blocks second tags on two-topic pages); one global cutoff (it fits only one
  stage, names or examples); the cosine to the members' centroid (it drifts with their
  number, so a cutoff right for five examples is too strict for two); centering the
  vectors on the scope's mean, or keeping only each document's best tags (both lift
  single-topic F1 and lose most second topics); EmbeddingGemma's classification or
  clustering prompts, and bge-m3, multilingual-e5-small and nomic-embed-text-v2 (all
  below EmbeddingGemma with its search prompts); a cap on tags per document (every
  qualifying tag is wanted).

The golden numbers below measure tagging over *essences*: a page's title, excerpt and
site name followed by the first 2 KB of its text, one per line
(`eval.Document.Essence`). Pass `Suggest`, and your catalog's `Members`, your documents
built the same way to get what the numbers say.

## Known limitations

- **Second and side topics.** Without `ChunkBytes` a page is one embedding of its first
  ~2 KB. When pairs of single-tag golden documents are stitched into one page (2.3 to
  2.9 KB), both tags are assigned 0.80 of the time, 0.87 with `ChunkBytes` 2000. On long
  items one side topic in ten is still missed with `ChunkBytes` (see **Long items**),
  and a topic a single paragraph covers rarely gets its tag, by design.
- **Documents that fit no tag** got none on the golden set, by a small margin: with the
  whole family of a document's tags removed (the tag, its parent, its siblings and its
  children), none of the 84 documents got one of the tags left, and the closest came
  within 0.016 of its cutoff. That is 15 tags in five families and one model, so expect
  a wrong tag now and then on a larger vocabulary.
- **A document whose own tag is missing gets a related one.** With only a document's own
  tags removed, 0.64 of the golden documents get no tag, and 0.81 none but an ancestor
  of their own tags (`cooking` for a recipe once `cooking / recipes` is gone), which the
  path implies. The rest get a sibling or a child, a related tag but a wrong one.
- **Siblings are the usual mistake.** Nearly every wrong suggestion is a tag of the
  right family: `cooking / baking` for a recipe, a parent next to its child. Unrelated
  tags are all but never suggested.
- **Tags that mix unrelated topics** average their members into one vector that
  matches neither topic well.
- **The two cutoffs come from a small golden set** (84 documents, 15 tags, one model);
  the model's scores sit in a narrow band, so ±0.04 on the name cutoff already costs a
  lot of precision or recall. Re-check them on real data, and measure them again for
  another model. Both the essences and the long documents score a little better at a
  name cutoff of 0.43 (F1 +0.02 and +0.03), which costs stitched two-topic pages 0.04;
  the default stays 0.42.

## Generating tags

```go
emb, err := yzma.New(ctx, cfg)          // the embedding model scores the tags
llm, err := yzma.NewCompleter(ctx, cfg) // the chat model writes them
gen, err := tagging.NewGenerator(llm, emb, tagging.GeneratorConfig{})
tags, err := gen.Generate(ctx, essence, tagging.DefaultGeneratorMinScore) // []tagging.Tag{Name, Score}, best first

// options: the language of the tags, and tags as paths of a tree
gen, err = tagging.NewGenerator(llm, emb, tagging.GeneratorConfig{},
	tagging.WithLanguageTags("en"),                       // or WithNativeLanguageTags(), the default
	tagging.WithTreeTags("dev / go", "cooking / baking")) // extend these paths; none: a tree of its own
```

**Options.** `WithNativeLanguageTags()` (the default) writes the tags in the language of the
text; `WithLanguageTags(lang)` in one language whatever the text's, `lang` a BCP 47 tag
(`en`, `es`, `pt-BR`); the prompt opens with that rule, asks the model to translate the
text's topics rather than copy its words (proper names stay), and repeats the rule last.
A 4B model copies a text's words readily: with the rule stated once, last, English tags
for the 55 German and Spanish golden documents had 88 of 550 tags still in the text's
language; stated first and last, 28, most of them place and dish names. With `WithTreeTags`,
`WithNativeLanguageTags` states its rule first and last too; without, last only (first as
well cost long texts 0.07 of side recall, under an earlier scoring rule). `WithTreeTags(existing...)` writes each tag as a path from a broad
category down (`dev / go / error wrapping`, at most `MaxTreeLevels` deep) and the tags of a
text as a small tree sharing upper levels. With existing paths, the model is shown them (up
to `MaxTreeBytes`) and asked to extend them, written as they are, and to start a new
top-level category only when none fits; a tag is scored by its whole path. A path the model
reuses, whole or as the start of a longer one, comes back in your spelling whatever case or
spacing the model wrote (`cooking/baking` becomes your `Cooking / Baking`), and is kept even
when one of its levels is longer than the four words and 40 characters a level the model
adds may have. On a Spanish
text about Go errors, with the golden tags as the tree, it wrote `dev / go / error wrapping`
and three siblings, in English although the text is Spanish: the tree's language wins over
`WithNativeLanguageTags`, although the prompt also says that new levels follow the text.
Measured 2026-11-07 with an English tree on six German and Spanish documents, every tag
came back English; `WithLanguageTags` turned the German ones German, existing levels
translated too, but left two of the three Spanish ones English. So set `WithLanguageTags` to
the tree's language. Without existing paths the native rule holds; before it was stated
first, short German and Spanish texts got English trees. On a text that fits no
branch it opened new roots (`tech / agentic web`) rather than reaching for `dev / web`. A
Generator keeps no state, so building one per call, with the tree as it is now, is cheap.

The Tagger picks among tags your application already has. The Generator needs none: it writes them
for one text, stores nothing, and returns the tags that score at least the cutoff you pass.
For a short text, hand it one built like the essences above; for a long page, pass the
title, a newline and then the full body (see **Long texts**). A text just over
`ChunkBytes` (2 KB) still costs one call, but its tags are scored per chunk, a scale
measured on long documents and not on essences.

- **The model writes, the embedder scores.** The chat model (Qwen3 4B Instruct, Apache-2.0)
  is asked for up to `GeneratorConfig.Candidates` (10) tags of one to three words, in the text's
  language unless the options above say otherwise. A model's own confidence is poorly calibrated, so each tag is scored by the
  cosine between the embedding of the text (as a query) and of the tag (as a document):
  the roles, and so the scale, of the Tagger's `NameMinScore`. One cutoff then means the same
  on every text, and a tag the model invented falls under it.
- **The reply is read leniently.** The model is asked for one tag per line, and what it
  wraps around them is dropped: a `<think>…</think>` block (one never closed takes the
  rest of the reply), code fences, a label such as `Tags:`, `Here are the tags:` or
  `标签：`, list markers, numbers and checkboxes, quotes, brackets, Markdown emphasis,
  and a remark in brackets after a tag (`sourdough (bread)`). Several tags on a line are
  split at commas and semicolons, the Chinese, Japanese and Arabic ones included
  (`、` `，` `；` `،` `؛`), and a run of hashtags (`#go #rust`) is one tag each. A reply
  that is JSON gives only the strings that are tags: the strings of an array, and in an
  object the members named `tags`, `keywords`, `labels` or `topics` and each list item's
  `name`, `tag` or `label`; scores, flags and other members (`"language": "en"`) are
  never tags, and such a reply is not read by lines as well. A reply that is only
  `None`, `No tags.`, `N/A` or the like, in English, Spanish or German, means no tags.
  What belongs to a tag stays: a colon unless a label word stands before it
  (`Topic modeling: LDA`, `Star Trek: Voyager`), a bracket that closes inside it
  (`(re)insurance`), an inch mark (`5" floppy`), a lone `<` or `>` (`x < y`). An arrow
  is kept in a plain tag (`a -> b`) and is the step to the next level in a tree tag
  (`a / b`). A name left with a bracket or a quote that does not close, with an HTML
  element, or longer than four words or 40 characters, is not a tag.
- **A bad answer is asked for again, once.** A model call that fails or replies with
  nothing is repeated, unless the context has ended or the error wraps
  `sense.ErrClosed`: a closed model is asked once. If it fails again on a short text,
  `Generate` returns the error; a model that twice says nothing is
  `tagging.ErrEmptyReply`, not "no tags". On a long text the other windows still count:
  `Generate` returns their tags *and* an error wrapping `tagging.ErrPartial` that names
  the windows it lost. A model that is gone is not asked about the rest of the text: a
  closed one stops the reading at once, and when the first two windows fail with none
  read yet the others are skipped, so a model that is down costs four calls, not two per
  window. Check the error before you discard the tags:

  ```go
  tags, err := gen.Generate(ctx, text, tagging.DefaultGeneratorMinScore)
  if err != nil && !errors.Is(err, tagging.ErrPartial) {
      return err // nothing was read
  }
  // tags are good; with ErrPartial, topics of the lost windows may be missing
  ```
- **The cutoff is yours.** `-1` returns every candidate. `tagging.DefaultGeneratorMinScore` (0.45)
  fits the default models: over ten golden documents in English, Spanish and German the
  model's tags scored 0.31 to 0.68, the main topics at 0.5 and up, and what fell under
  0.45 was side detail (the steam for a loaf, a side salad). That is a judgment by eye on a
  small set, and another embedder scores on another scale, so choose again for it.
- **Up to 64 KB are read** (`GeneratorConfig.MaxBytes`); nothing after that reaches the
  model or the embedder, so a tag is never scored against words the embedder did not see.
- **Long texts.** A text over 2 KB (`GeneratorConfig.ChunkBytes`) is read in windows of
  about 6 KB, one LLM call each, a short first line (the title) repeated at the head of
  every window, so pass the text as `title\n` + body; the names merge, and each is scored by its best cosine
  over the 2 KB chunks of the text. On 24 synthetic documents of 7 to 31 KB (93 side
  topics, all past the first 2 KB) that scored a mean F1 of 0.82 against 0.49 for reading
  the first 2 KB only: side-topic recall 0.76 against 0.24, precision 0.89 against 0.78,
  main-topic recall 1.0 in both, and no passing mention tagged (`TestGenerateLongTags`).
  Scoring by the best chunk won because the mean of the chunks and
  the top-k mean dilute exactly the section topics the windows find. 6 KB windows beat
  2 KB ones (F1 0.84 against 0.74 in the experiment, under an earlier scoring rule) because the model needs some context to name a
  section's topic, and 2 KB windows give many near-duplicates at 2.7 times the calls.
  The price is time: about 3 LLM calls and 15 s for a document of 16 KB on the test
  laptop's iGPU, and a 64 KB text is about 11 calls. `zarf/examples`' autotag pays this
  for long notes. The default chat context is 4096 tokens so that a window fits, which
  roughly doubles the model's KV cache (a few hundred MB).
- **Cost.** The chat model is a 2.5 GB download, and a process with it and the embedding
  model loaded holds about 5 GB. A text takes 7 to 13 s with the default 4 threads on an
  AMD Ryzen AI 9 HX PRO 370 (English about 7 s, German 11), nearly all of it writing the
  tags; `yzma.Runtime.Threads` raises that. Decoding is greedy, so a text gets the same tags
  every time.
- **Limits.** The tags are as good as a 4B model's: their casing is the model's, a rare
  one is malformed (`FitnessimHome`), and they come back in the text's language. They
  cover what the text says; to tag against a vocabulary you keep, use the Tagger.

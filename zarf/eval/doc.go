// Package eval measures how well a langai.Embedder serves the two tasks
// this module uses embeddings for, over a labelled dataset:
//
//   - retrieval: free-text query → relevant documents, what search.Index
//     does (documents indexed through the production chunker, every query
//     searched, recall@k / MRR / nDCG@10);
//   - tagging: document essence → expected tag paths, what a tagging.Tagger
//     does for tags without hand-tagged examples yet (tag paths
//     indexed as documents, every essence searched as a query, precision /
//     recall / F1 at a floor plus a floor sweep and top-1 accuracy).
//
// Retrieval is reported over two sets of queries, split as
// Dataset.SplitQueries does: the same-language ones (a German query for a
// German document, the search a service must get right) and the
// cross-lingual ones (a Spanish query for an English document), so a model's
// ability across languages shows next to the headline without colouring it.
// There is no per-language breakdown: the dataset is multilingual so that the
// numbers already cover every language it holds. Tagging is reported once
// over every document. Timing per short and per long text rides along.
// A second set, testdata/long (LoadLong), holds long documents with labelled
// main, side and passing-mention topics; MatchTopics scores the tags a
// tagging.Generator wrote or a tagging.Tagger picked for one of them against
// those topics (a tag counts for the topic it names, word for word, or is
// similar enough to), and MeanLongScores averages the documents: each rate
// over the documents it exists for, and the F1 as the mean of the
// documents' own.
// Everything runs over the exact in-memory store, so the numbers reflect the
// model, never an index.
//
// The checked-in dataset is testdata/golden.yaml (see golden_test.go for its
// composition rules). Five callers drive this package: the gated
// TestGoldenSetWithDefaultModel in provider/yzma, which asserts floors for
// the default model, the gated TestGenerateLongTags in provider/yzma and
// TestTaggerLongTags in tagging, which hold the long set's floors for the
// Generator and the Tagger, the gated TestGoldenTagging in tagging, which
// replays the set through a Tagger with hand-tagged examples, and the gated
// TestCompareModels in this package's own tests (make semeval), which
// compares the models of testdata/models.yaml and prints a table. The
// dataset format is plain YAML so an export of an application's own
// documents can be evaluated the same way.
package eval

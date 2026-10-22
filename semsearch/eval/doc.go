// Package eval measures how well a semsearch.Embedder serves the two tasks
// this module uses embeddings for, over a labelled dataset:
//
//   - retrieval: free-text query → relevant documents, what Index.Search
//     does (documents indexed through the production chunker, every query
//     searched, recall@k / MRR / nDCG@10);
//   - tagging: document essence → expected tag paths, what the autotag
//     package does for tags without hand-tagged examples yet (tag paths
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
// Everything runs over the exact in-memory store, so the numbers reflect the
// model, never an index.
//
// The checked-in dataset is testdata/golden.yaml (see golden_test.go for its
// composition rules). Three callers drive this package: the gated
// TestGoldenSetWithDefaultModel in semsearch/yzma, which asserts floors for
// the default model, the gated TestGoldenTagging in autotag, which replays
// the set through a Tagger with hand-tagged examples, and cmd/semeval, which
// compares several models and prints a table. The dataset format is plain
// YAML so an export of an application's own documents can be evaluated the
// same way.
package eval

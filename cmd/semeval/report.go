package main

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/go-bumbu/langai/semsearch/eval"
)

// row is one model's outcome: its result and load time, or the error that
// stopped it.
type row struct {
	Name   string        `json:"name"`
	Load   time.Duration `json:"load_ns"`
	Result *eval.Result  `json:"result,omitempty"`
	Err    string        `json:"error,omitempty"`
}

// header names the table's columns: quality first (SEARCH, XLING, TAGS with
// the FLOOR that reached it), then what the model costs (DIMS, MS/QUERY,
// MS/DOC). Every column but the trailing ERROR is a tab-terminated cell.
const header = "MODEL\tSEARCH\tXLING\tTAGS\tFLOOR\tDIMS\tMS/QUERY\tMS/DOC\t"

// legend explains each column in a line, printed under the table.
const legend = `
SEARCH    share of the documents a query should find that are in its top 5, over queries in those documents' language
XLING     the same over the queries that must cross a language (a Spanish query for an English document)
TAGS      best F1 of tag assignment over the floor sweep; FLOOR is the cutoff that reached it (autotag's NameMinScore)
DIMS      vector size
MS/QUERY  mean embedding time per query or tag path; MS/DOC per document essence
`

// writeTable prints what the dataset holds, one line per model and the
// legend; a failed model shows its error in the last column. The finer
// metrics stay in the JSON report. Every row carries as many cells as the
// header: text/tabwriter ends a column block at a line with fewer cells, and
// the rows after it would then align on their own instead of under the
// header.
func writeTable(w io.Writer, ds *eval.Dataset, rows []row) {
	same, cross := ds.SplitQueries()
	_, _ = fmt.Fprintf(w, "%d documents, %d same-language and %d cross-lingual queries, %d tags\n\n",
		len(ds.Documents), len(same), len(cross), len(ds.Tags))
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintln(tw, header+"ERROR")
	for _, r := range rows {
		if r.Result == nil {
			// the name, an empty cell under every other column, then the error under ERROR
			_, _ = fmt.Fprintf(tw, "%s%s%s\n", r.Name, strings.Repeat("\t", strings.Count(header, "\t")), r.Err)
			continue
		}
		res := r.Result
		_, _ = fmt.Fprintf(tw, "%s\t%s\t%s\t%.3f\t%.2f\t%d\t%.1f\t%.1f\t\n", r.Name,
			recall5(res.SameLanguage), recall5(res.CrossLingual), res.Tagging.Best.F1, res.Tagging.Best.Floor,
			res.Dims, ms(res.Timing.ShortPerText), ms(res.Timing.LongPerText))
	}
	_ = tw.Flush()
	_, _ = fmt.Fprint(w, legend)
}

// recall5 is a query set's recall@5, or "-" when the dataset has no query of
// that kind: an absence, not a score of zero.
func recall5(r eval.Retrieval) string {
	if r.Queries == 0 {
		return "-"
	}
	return fmt.Sprintf("%.3f", r.RecallAt[5])
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// writeJSON writes every row with its full result and load time, for diffing
// two runs or reading the metrics the table leaves out.
func writeJSON(w io.Writer, rows []row) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(rows)
}

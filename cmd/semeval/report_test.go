package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/go-bumbu/langai/semsearch/eval"
)

// cannedDataset is what the table's first line counts: three documents in
// three languages, two same-language queries and one cross-lingual, two tags.
func cannedDataset() *eval.Dataset {
	return &eval.Dataset{
		Tags: []string{"dev", "cooking"},
		Documents: []eval.Document{
			{ID: "go", Lang: "en", Title: "Go", Text: "go", Tags: []string{"dev"}},
			{ID: "pasta", Lang: "es", Title: "Pasta", Text: "pasta", Tags: []string{"cooking"}},
			{ID: "brot", Lang: "de", Title: "Brot", Text: "brot", Tags: []string{"cooking"}},
		},
		Queries: []eval.Query{
			{ID: "q-go", Lang: "en", Text: "go", Relevant: []string{"go"}},
			{ID: "q-brot", Lang: "de", Text: "brot", Relevant: []string{"brot"}},
			{ID: "q-pasta", Lang: "en", Text: "pasta", Relevant: []string{"pasta"}},
		},
	}
}

// cannedRows is a good model, a failed one and a second good model, so what
// follows the failed row has to line up under the header too. The good ones
// share one result.
func cannedRows() []row {
	res := &eval.Result{
		Dims:         384,
		SameLanguage: eval.Retrieval{Queries: 2, RecallAt: map[int]float64{1: 0.6, 3: 0.8, 5: 0.9, 10: 0.95}, MRR: 0.7, NDCG10: 0.75},
		CrossLingual: eval.Retrieval{Queries: 1, RecallAt: map[int]float64{1: 0.4, 3: 0.6, 5: 0.7, 10: 0.8}, MRR: 0.55, NDCG10: 0.6},
		Tagging:      eval.Tagging{Documents: 3, AtFloor: eval.Cut{Floor: 0.35, F1: 0.61}, Best: eval.Cut{Floor: 0.28, F1: 0.66}, Top1: 0.8},
		Timing:       eval.Timing{ShortPerText: 2 * time.Millisecond, LongPerText: 45 * time.Millisecond},
	}
	return []row{
		{Name: "org/good", Load: 1200 * time.Millisecond, Result: res},
		{Name: "org/broken", Err: "download: 404 Not Found"},
		{Name: "org/also-good", Load: 300 * time.Millisecond, Result: res},
	}
}

func TestWriteTable(t *testing.T) {
	var buf bytes.Buffer
	writeTable(&buf, cannedDataset(), cannedRows())
	out := buf.String()
	for _, want := range []string{
		"3 documents, 2 same-language and 1 cross-lingual queries, 2 tags",
		"MODEL", "SEARCH", "XLING", "TAGS", "FLOOR", "DIMS", "MS/QUERY", "MS/DOC", "ERROR",
		"org/good", "0.900", "0.700", "0.660", "0.28", "384", "2.0", "45.0",
		"org/broken", "download: 404 Not Found", "org/also-good",
		"SEARCH    share of the documents", "MS/QUERY  mean embedding time",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("table lacks %q:\n%s", want, out)
		}
	}
	// no per-language rows, no load time, no F1 at the floor, no top-1: one
	// row per model, and the finer metrics stay in the JSON
	for _, gone := range []string{"BUCKET", "LOAD", "1.2s", "0.610", "0.800"} {
		if strings.Contains(out, gone) {
			t.Errorf("table still has %q:\n%s", gone, out)
		}
	}
	if n := strings.Count(out, "org/good"); n != 1 {
		t.Errorf("want one row for org/good, found %d:\n%s", n, out)
	}
}

// TestWriteTable_EmptyQuerySetShowsDash: a dataset without cross-lingual
// queries prints "-" under XLING, not 0.000, which would read as a model that
// finds nothing.
func TestWriteTable_EmptyQuerySetShowsDash(t *testing.T) {
	ds := cannedDataset()
	ds.Queries = ds.Queries[:2] // the two same-language queries
	rows := cannedRows()[:1]
	rows[0].Result.CrossLingual = eval.Retrieval{RecallAt: map[int]float64{}}
	var buf bytes.Buffer
	writeTable(&buf, ds, rows)
	out := buf.String()
	if !strings.Contains(out, "3 documents, 2 same-language and 0 cross-lingual queries") {
		t.Errorf("first line does not count the queries:\n%s", out)
	}
	_, lines := table(t, out)
	if fields := strings.Fields(lines[0]); len(fields) < 3 || fields[0] != "org/good" || fields[2] != "-" {
		t.Errorf("row = %q, want '-' under XLING", lines[0])
	}
}

// columns are the table's columns in order; the last one holds a failed
// model's error.
var columns = []string{"MODEL", "SEARCH", "XLING", "TAGS", "FLOOR", "DIMS", "MS/QUERY", "MS/DOC", "ERROR"}

// TestWriteTable_FailedModelKeepsAligned guards a text/tabwriter rule: a line
// with fewer cells than the one above ends the column blocks, so a failed
// model's row must carry as many cells as the header or every row after it
// aligns on its own. The error has to sit under ERROR and every cell of the
// other rows under its column's header.
func TestWriteTable_FailedModelKeepsAligned(t *testing.T) {
	var buf bytes.Buffer
	writeTable(&buf, cannedDataset(), cannedRows())
	out := buf.String()

	header, rows := table(t, out)
	problems, perModel := alignmentProblems(rows, columnStarts(t, header, columns))
	if len(problems) > 0 {
		t.Errorf("table is misaligned:\n%s\n\n%s", strings.Join(problems, "\n"), out)
	}
	for model, want := range map[string]int{"org/good": 1, "org/broken": 1, "org/also-good": 1} {
		if perModel[model] != want {
			t.Errorf("%s has %d rows, want %d:\n%s", model, perModel[model], want, out)
		}
	}
}

// table splits writeTable's output into the header and the rows under it,
// which end at the blank line before the legend.
func table(t *testing.T, out string) (header string, rows []string) {
	t.Helper()
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	for i, l := range lines {
		if !strings.HasPrefix(l, "MODEL") {
			continue
		}
		for _, r := range lines[i+1:] {
			if strings.TrimSpace(r) == "" {
				break
			}
			rows = append(rows, r)
		}
		return l, rows
	}
	t.Fatalf("no table header in:\n%s", out)
	return "", nil
}

// columnStarts returns the offset at which each of cols starts in header.
func columnStarts(t *testing.T, header string, cols []string) []int {
	t.Helper()
	starts := make([]int, len(cols))
	for i, c := range cols {
		starts[i] = strings.Index(header, c)
		if starts[i] < 0 || (i > 0 && starts[i] <= starts[i-1]) {
			t.Fatalf("header lacks %q, or has it out of order: %q", c, header)
		}
	}
	return starts
}

// alignmentProblems checks the rows against the offsets the header's columns
// start at and reports how many rows each model has. The failed model
// ("org/broken") must start its error under the last column; every other row
// must start each cell before it under its column.
func alignmentProblems(rows []string, starts []int) (problems []string, perModel map[string]int) {
	perModel = map[string]int{}
	errCol := len(starts) - 1
	for _, line := range rows {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		model := fields[0]
		perModel[model]++
		if model == "org/broken" {
			if at := strings.Index(line, "download: 404 Not Found"); at != starts[errCol] {
				problems = append(problems, fmt.Sprintf("%s: the error starts at offset %d, want %d (under ERROR)", model, at, starts[errCol]))
			}
			continue
		}
		for col := 0; col < errCol; col++ {
			if !startsUnder(line, starts[col]) {
				problems = append(problems, fmt.Sprintf("%s: the %s cell is not under its header", model, columns[col]))
			}
		}
	}
	return problems, perModel
}

// startsUnder reports whether a cell begins at offset at: a non-blank there
// with a blank (or the line start) before it.
func startsUnder(line string, at int) bool {
	return at < len(line) && line[at] != ' ' && (at == 0 || line[at-1] == ' ')
}

func TestWriteJSON_RoundTrips(t *testing.T) {
	var buf bytes.Buffer
	if err := writeJSON(&buf, cannedRows()); err != nil {
		t.Fatal(err)
	}
	var back []row
	if err := json.Unmarshal(buf.Bytes(), &back); err != nil {
		t.Fatal(err)
	}
	if len(back) != 3 || back[0].Name != "org/good" || back[0].Result.SameLanguage.MRR != 0.7 || back[0].Result.Tagging.Top1 != 0.8 ||
		back[1].Err == "" || back[2].Name != "org/also-good" || back[2].Result == nil {
		t.Fatalf("round trip = %+v", back)
	}
}

package textsearch

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-bumbu/sense/search"
)

// Hit is one search result joined with its catalog entry.
type Hit struct {
	Doc   Doc     `json:"doc"`
	Score float64 `json:"score"`
	// Seq is the position of the chunk that matched best among the
	// document's chunks, Heading its heading path ("Title › Section", empty
	// for a text without headings) and Text the rest of it.
	Seq     int    `json:"seq"`
	Heading string `json:"heading"`
	Text    string `json:"text"`
}

// Result is what a search returns.
type Result struct {
	// Query is the text searched for; empty for "more like this".
	Query string `json:"query,omitempty"`
	// Doc is the document "more like this" started from.
	Doc *Doc `json:"doc,omitempty"`
	// Stored says the query was the document's stored chunk vectors, so no
	// model ran.
	Stored bool `json:"stored,omitempty"`
	// ElapsedUs is what the search took, embedding the query included, in
	// microseconds: a search with stored vectors takes well under a
	// millisecond on a small corpus.
	ElapsedUs int64 `json:"elapsedUs"`
	Hits      []Hit `json:"hits"`
}

// Search embeds the query and returns the documents closest to it, each
// with the chunk that matched best. limit <= 0 means the Library's default.
func (l *Library) Search(ctx context.Context, query string, limit int) (Result, error) {
	start := time.Now()
	matches, err := l.cfg.Index.Search(ctx, query, l.limit(limit))
	if err != nil {
		return Result{}, err
	}
	r := l.result(matches, start)
	r.Query = query
	return r, nil
}

// Similar returns the documents most like the catalog document id, searching
// with its stored chunk vectors (search.Index.SearchIndexed): no model runs.
// The document itself is never among the hits. A document the index does
// not hold is search.ErrNotIndexed.
func (l *Library) Similar(ctx context.Context, id string, limit int) (Result, error) {
	doc, ok := l.Doc(id)
	if !ok {
		return Result{}, ErrNotFound
	}
	start := time.Now()
	matches, err := l.cfg.Index.SearchIndexed(ctx, id, l.limit(limit))
	if err != nil {
		return Result{}, err
	}
	r := l.result(matches, start)
	r.Doc, r.Stored = &doc, true
	return r, nil
}

func (l *Library) limit(limit int) int {
	if limit <= 0 {
		return l.cfg.Limit
	}
	return limit
}

// result joins the matches with the catalog; a match whose file vanished
// since the scan keeps its id only.
func (l *Library) result(matches []search.Match, start time.Time) Result {
	r := Result{ElapsedUs: time.Since(start).Microseconds(), Hits: make([]Hit, 0, len(matches))}
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, m := range matches {
		h := Hit{Doc: Doc{ID: m.DocID, Name: filepath.Base(m.DocID), Title: m.DocID}, Score: m.Score, Seq: m.Seq}
		if d, ok := l.docs[m.DocID]; ok {
			h.Doc = *d
		}
		h.Heading, h.Text = splitHeading(m.Text, l.heads[m.DocID])
		r.Hits = append(r.Hits, h)
	}
	return r
}

// splitHeading tells a chunk's heading path from its text. The chunker
// starts a chunk with the path of the headings it sits under ("Title ›
// Section") and a blank line; a chunk of a text without headings has no
// such line. The first paragraph is the path when it is one line made of
// the document's headings.
func splitHeading(chunk string, heads map[string]bool) (heading, text string) {
	first, rest, _ := strings.Cut(chunk, "\n\n")
	if len(heads) == 0 || strings.Contains(first, "\n") {
		return "", chunk
	}
	for part := range strings.SplitSeq(first, " › ") {
		if !heads[strings.TrimSpace(part)] {
			return "", chunk
		}
	}
	return first, rest
}

// ChunkView is one chunk of a document as it was embedded.
type ChunkView struct {
	Seq     int    `json:"seq"`
	Heading string `json:"heading"`
	Text    string `json:"text"`
	// Bytes is the length of the whole chunk, heading path included: what
	// the chunker's MaxChars caps.
	Bytes int `json:"bytes"`
}

// DocView is a document with its text and the chunks the chunker cuts it
// into.
type DocView struct {
	Doc    Doc         `json:"doc"`
	Text   string      `json:"text"`
	Chunks []ChunkView `json:"chunks"`
	// Changed says the file is not what the last scan read (another size
	// or time): the chunks shown are those of the file as it is now, which
	// the index gets at the next scan.
	Changed bool `json:"changed"`
}

// View returns the document id with its text and its chunks: the file as it
// is now through the index's chunker, which is what was embedded as long as
// the file has not changed since the scan.
func (l *Library) View(id string) (DocView, error) {
	doc, ok := l.Doc(id)
	if !ok {
		return DocView{}, ErrNotFound
	}
	path := filepath.Join(l.cfg.Dir, filepath.FromSlash(id))
	info, err := os.Stat(path) //nolint:gosec // the id is a catalog key, so path is a file the scan found under the documents directory
	if err != nil {
		return DocView{}, err
	}
	if info.Size() > maxDocBytes {
		return DocView{Doc: doc, Chunks: []ChunkView{}}, nil
	}
	data, err := os.ReadFile(path) //nolint:gosec // the id is a catalog key, so path is a file the scan found under the documents directory
	if err != nil {
		return DocView{}, err
	}
	text := clean(string(data))
	heads := headings(text)
	chunks := l.chunker.Chunk(text)
	v := DocView{Doc: doc, Text: text, Chunks: make([]ChunkView, len(chunks)), Changed: info.Size() != doc.Bytes || !info.ModTime().Equal(doc.ModTime)}
	for i, c := range chunks {
		v.Chunks[i] = ChunkView{Seq: i, Bytes: len(c)}
		v.Chunks[i].Heading, v.Chunks[i].Text = splitHeading(c, heads)
	}
	return v, nil
}

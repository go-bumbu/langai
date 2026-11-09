package textsearch

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-bumbu/sense/zarf/eval"
)

// WriteSamples writes the eval golden texts into dir as Markdown files, so
// the page has documents to search without the user's own: the short
// labelled documents of goldenFile (zarf/eval/testdata/golden.yaml: English,
// Spanish and German, a few paragraphs each) under golden/, and the long
// documents of longDir (zarf/eval/testdata/long: 7 to 31 KB, cut into
// sections by "## " headings) under long/. Each file is its document's
// title as a "# " heading and its text. A file that is there already is
// kept, so it is safe to run again; an empty goldenFile or longDir skips
// that set. It returns how many files it wrote.
func WriteSamples(dir, goldenFile, longDir string) (int, error) {
	written := 0
	if goldenFile != "" {
		ds, err := eval.Load(goldenFile)
		if err != nil {
			return 0, fmt.Errorf("textsearch: samples: %w", err)
		}
		for _, d := range ds.Documents {
			body := strings.TrimSpace(d.Excerpt) + "\n\n" + strings.TrimSpace(d.Text)
			ok, err := writeSample(filepath.Join(dir, "golden", d.ID+".md"), d.Title, body)
			if err != nil {
				return written, err
			}
			if ok {
				written++
			}
		}
	}
	if longDir != "" {
		docs, err := eval.LoadLong(longDir)
		if err != nil {
			return written, fmt.Errorf("textsearch: samples: %w", err)
		}
		for _, d := range docs {
			ok, err := writeSample(filepath.Join(dir, "long", d.ID+".md"), d.Title, d.Text)
			if err != nil {
				return written, err
			}
			if ok {
				written++
			}
		}
	}
	return written, nil
}

// writeSample writes one document unless its file exists, and reports
// whether it wrote.
func writeSample(path, title, body string) (bool, error) {
	if _, err := os.Stat(path); err == nil {
		return false, nil
	} else if !errors.Is(err, fs.ErrNotExist) {
		return false, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return false, err
	}
	text := "# " + strings.TrimSpace(title) + "\n\n" + strings.TrimSpace(body) + "\n"
	if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
		return false, err
	}
	return true, nil
}

// SampleQueries returns up to n queries of goldenFile that have to cross a
// language to find their document (a Spanish question whose answer is an
// English text), for the page to offer: each from another language in turn.
// A missing or unreadable file gives none.
func SampleQueries(goldenFile string, n int) []string {
	ds, err := eval.Load(goldenFile)
	if err != nil {
		return nil
	}
	_, cross := ds.SplitQueries()
	byLang := map[string][]string{}
	var langs []string
	for _, q := range cross {
		if _, ok := byLang[q.Lang]; !ok {
			langs = append(langs, q.Lang)
		}
		byLang[q.Lang] = append(byLang[q.Lang], q.Text)
	}
	var out []string
	for i := 0; len(out) < n; i++ {
		took := false
		for _, lang := range langs {
			if i < len(byLang[lang]) && len(out) < n {
				out = append(out, byLang[lang][i])
				took = true
			}
		}
		if !took {
			break
		}
	}
	return out
}

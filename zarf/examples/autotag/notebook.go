// Package autotag is the automatic-tagging page of the example service: a
// notebook of notes and tags. Every note shows the tags a tagging.Tagger picks
// for it among the notebook's own (option one: an existing set of tags, which
// learns from the tags a person puts on notes by hand), and on demand the new
// tags a tagging.Generator writes for it, each with its score (option two:
// recommended tags).
//
// The notebook owns the notes and tags in a JSON file, and is the Tagger's
// Catalog: the Tagger reads the tags and their hand-made members from it when
// the notebook's revision changes, and keeps only embeddings, in a cache that
// a deleted or fresh one (another embedding model, say) refills as it goes.
package autotag

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/zarf/eval"
)

// Errors the handler maps to statuses.
var (
	ErrNotFound  = errors.New("autotag: no such note or tag")
	ErrBadInput  = errors.New("autotag: bad input")
	ErrNoSamples = errors.New("autotag: no samples file configured")
)

// Info is what the page says about the models and the cutoffs.
type Info struct {
	Embedder       string  `json:"embedder"`
	Completer      string  `json:"completer"`
	Accel          string  `json:"accel"`
	NameMinScore   float64 `json:"nameMinScore"`
	MemberMinScore float64 `json:"memberMinScore"`
	MinScore       float64 `json:"minScore"` // the Generator's cutoff
	// ChunkBytes is the Tagger's TaggerConfig.ChunkBytes: a longer note is
	// read in pieces
	ChunkBytes int  `json:"chunkBytes"`
	Samples    bool `json:"samples"`
}

// Config wires a Notebook.
type Config struct {
	// Cache keeps the embeddings of the Tagger New builds over the notebook.
	Cache tagging.EmbeddingCache
	// Tagging configures that Tagger; its Model is required.
	Tagging tagging.TaggerConfig
	// Chat writes the recommended tags and Embedder scores them: Recommend
	// builds a tagging.Generator over them with the options of each request.
	// Embedder is the Tagger's embedder too.
	Chat     langai.Completer
	Embedder langai.Embedder
	// File is the notebook's JSON file; empty keeps it in memory only.
	File string
	// SamplesFile is an eval dataset (zarf/eval/testdata/golden.yaml) that
	// LoadSamples imports; empty disables it.
	SamplesFile string
	// ChatState says where the chat model's lazy loader stands ("not
	// started", "loading", "loaded", or the error of a load that failed);
	// nil leaves it out.
	ChatState func() string
	// EmbedderState says the same of the embedding model, and RunsOn what
	// it runs on, which a lazy model knows once it is loaded; nil leaves
	// them out, and the page then shows Info.Accel.
	EmbedderState func() string
	RunsOn        func() string
	Info          Info
	Logger        *slog.Logger
}

// Tag is one of the notebook's tags.
type Tag struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// Note is one text of the notebook and the tags a person put on it.
type Note struct {
	ID    string `json:"id"`
	Title string `json:"title,omitempty"`
	Text  string `json:"text"`
	// Tags are the ids of the tags put on it by hand.
	Tags []string `json:"tags"`
	// Expected are the tag names a labelled sample carries, for comparison.
	Expected []string `json:"expected,omitempty"`
}

type catalog struct {
	Tags  []Tag  `json:"tags"`
	Notes []Note `json:"notes"`
	Next  int    `json:"next"`
}

// The Tagger space the notebook is: it has only one.
const (
	scopeIndex     = "examples"
	scopePartition = "notebook"
)

// Notebook keeps the notes and tags, serves them to its Tagger as the
// Catalog, and asks the Tagger and the Generator for tags.
type Notebook struct {
	cfg    Config
	logger *slog.Logger
	tagger *tagging.Tagger
	scope  *tagging.Scope
	mu     sync.RWMutex
	cat    catalog
	// rev is the notebook's revision, which every change moves up. tagRev
	// holds each tag's revision: the notebook's revision when its members
	// last changed, or when it was made, so a tag deleted and made again
	// under the same id never comes back at a revision the Tagger has seen.
	rev    int64
	tagRev map[string]int64
}

// New loads the notebook file, if there is one, builds a Tagger over it (the
// notebook is the Catalog) and warms it: the tags are built and every note is
// embedded, unless the cache holds it already. A fresh cache, or another
// model, embeds everything again. When the cache holds everything, the
// embedder is never called, so a lazy one stays unloaded. An embedder that
// fails does not keep the notebook from opening: the failure is logged and
// each note shows it until the embedder works.
func New(ctx context.Context, cfg Config) (*Notebook, error) {
	if cfg.Cache == nil || cfg.Chat == nil || cfg.Embedder == nil {
		return nil, errors.New("autotag: Config needs a Cache, a Chat model and an Embedder")
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	cfg.Info.Samples = cfg.SamplesFile != ""
	nb := &Notebook{cfg: cfg, logger: logger, rev: 1, tagRev: map[string]int64{}}
	if cfg.File != "" {
		data, err := os.ReadFile(cfg.File)
		switch {
		case errors.Is(err, os.ErrNotExist):
		case err != nil:
			return nil, fmt.Errorf("autotag: read notebook: %w", err)
		default:
			if err := json.Unmarshal(data, &nb.cat); err != nil {
				return nil, fmt.Errorf("autotag: read notebook %s: %w", cfg.File, err)
			}
		}
	}
	for _, t := range nb.cat.Tags {
		nb.tagRev[t.ID] = 1
	}
	tg, err := tagging.NewTagger(cfg.Embedder, nb, cfg.Cache, cfg.Tagging)
	if err != nil {
		return nil, fmt.Errorf("autotag: %w", err)
	}
	nb.tagger, nb.scope = tg, tg.Scope(tagging.Space{Index: scopeIndex, Partition: scopePartition})

	start := time.Now()
	texts := make([]string, len(nb.cat.Notes))
	for i, n := range nb.cat.Notes {
		texts[i] = itemText(n)
	}
	if err := nb.scope.Warm(ctx, revision(nb.rev), texts...); err != nil {
		if ctx.Err() != nil {
			_ = tg.Close()
			return nil, fmt.Errorf("autotag: warm the tagger: %w", err)
		}
		nb.logger.Warn("notebook not warmed: its notes are embedded when the embedder works", "err", err)
	}
	nb.logger.Info("notebook loaded", "tags", len(nb.cat.Tags), "notes", len(nb.cat.Notes), "elapsed", time.Since(start).Round(time.Millisecond))
	return nb, nil
}

// Close stops the notebook's Tagger and waits for what it has running (a
// rebuild after a change, a View in flight) to end, so nothing reads the
// embedder or the cache after it: call it before closing those. Every later
// View and LoadSamples fails with langai.ErrClosed. It does not close the
// Embedder, the Chat model or the Cache, which the caller made.
func (nb *Notebook) Close() error { return nb.tagger.Close() }

// PruneCache removes from the Tagger's cache the embeddings last used before
// unusedSince, and returns how many: the texts of notes edited or deleted
// since, and what another embedding model left there. It goes through
// Tagger.Prune with the notebook's space, not the cache's own Prune, so the
// names and members of tags that have not changed since, which the Tagger
// never reads again, are marked as used first. unusedSince must be at least
// tagging.MinPruneAge in the past.
func (nb *Notebook) PruneCache(ctx context.Context, unusedSince time.Time) (int64, error) {
	return nb.tagger.Prune(ctx, unusedSince, nb.scope.Space())
}

// KeepCachePruned prunes the cache of what was unused for unusedFor (see
// PruneCache) now and then once every interval, until ctx ends, and logs how
// many embeddings each run removed. It returns when ctx ends or the notebook
// is closed, so run it on a goroutine and cancel ctx before Close.
func (nb *Notebook) KeepCachePruned(ctx context.Context, unusedFor, interval time.Duration) {
	tick := time.NewTicker(interval)
	defer tick.Stop()
	for {
		removed, err := nb.PruneCache(ctx, time.Now().Add(-unusedFor))
		switch {
		case ctx.Err() != nil || errors.Is(err, langai.ErrClosed):
			return
		case err != nil:
			nb.logger.Warn("tag cache not pruned", "err", err)
		default:
			nb.logger.Info("tag cache pruned", "removed", removed, "unused_for", unusedFor)
		}
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}

// revision formats a revision counter as the Tagger reads it.
func revision(r int64) string { return strconv.FormatInt(r, 10) }

// changed moves the notebook's revision up, and the revision of each of
// tagIDs to it: their members changed. The caller holds the write lock.
func (nb *Notebook) changed(tagIDs ...string) {
	nb.rev++
	for _, id := range tagIDs {
		nb.tagRev[id] = nb.rev
	}
}

// Tags implements tagging.Catalog: the notebook's tags, each at its own
// revision. The notebook is a single space, so index and partition are
// ignored.
func (nb *Notebook) Tags(context.Context, string, string) ([]tagging.CatalogTag, error) {
	nb.mu.RLock()
	defer nb.mu.RUnlock()
	out := make([]tagging.CatalogTag, len(nb.cat.Tags))
	for i, t := range nb.cat.Tags {
		out[i] = tagging.CatalogTag{ID: t.ID, Name: t.Name, Revision: revision(nb.tagRev[t.ID])}
	}
	return out, nil
}

// Members implements tagging.Catalog: for each of tagIDs, the texts of the
// notes that carry it by hand.
func (nb *Notebook) Members(_ context.Context, _, _ string, tagIDs []string) (map[string][]string, error) {
	nb.mu.RLock()
	defer nb.mu.RUnlock()
	out := make(map[string][]string, len(tagIDs))
	for _, id := range tagIDs {
		out[id] = nil
	}
	for _, n := range nb.cat.Notes {
		for _, id := range n.Tags {
			if texts, ok := out[id]; ok {
				out[id] = append(texts, itemText(n))
			}
		}
	}
	return out, nil
}

// itemText is what the Tagger and the Generator read for a note.
func itemText(n Note) string {
	return strings.TrimSpace(n.Title + "\n" + n.Text)
}

// Info describes the models and cutoffs.
func (nb *Notebook) Info() Info { return nb.cfg.Info }

// ChatState says whether the chat model is loaded.
func (nb *Notebook) ChatState() string {
	if nb.cfg.ChatState == nil {
		return ""
	}
	return nb.cfg.ChatState()
}

// EmbedderState says whether the embedding model is loaded.
func (nb *Notebook) EmbedderState() string {
	if nb.cfg.EmbedderState == nil {
		return ""
	}
	return nb.cfg.EmbedderState()
}

// RunsOn says what the embedding model runs on: what the Notebook was told
// (Config.RunsOn), or Info.Accel.
func (nb *Notebook) RunsOn() string {
	if nb.cfg.RunsOn == nil {
		return nb.cfg.Info.Accel
	}
	return nb.cfg.RunsOn()
}

// TagView is a tag with how many notes carry it by hand and how many the
// Tagger picks it for.
type TagView struct {
	Tag
	Hand int `json:"hand"`
	Auto int `json:"auto"`
}

// NoteView is a note with the tags the Tagger picks for it, best first, and
// its score against every tag.
type NoteView struct {
	Note
	Auto   []string `json:"auto"`
	Scores []Score  `json:"scores"`
	// Bytes is the length of what the Tagger reads, and Parts how many pieces
	// it is embedded in (see Info.ChunkBytes).
	Bytes int    `json:"bytes"`
	Parts int    `json:"parts"`
	Error string `json:"error,omitempty"`
}

// Score is a note's score against one tag and the cutoff the tag picks at.
// ShadowedBy is the id of a tag on its path the note fits better, when the
// score reaches the cutoff and the Tagger picks that tag instead (see
// tagging.TagScore).
type Score struct {
	TagID      string  `json:"tagId"`
	Score      float64 `json:"score"`
	Cutoff     float64 `json:"cutoff"`
	ShadowedBy string  `json:"shadowedBy,omitempty"`
}

// View is the whole notebook with every note's automatic tags.
type View struct {
	Tags      []TagView  `json:"tags"`
	Notes     []NoteView `json:"notes"`
	SuggestMs int64      `json:"suggestMs"`
}

// View returns every tag and note, asking the Tagger for each note's scores.
// Scores reads cached embeddings only, so this embeds nothing unless a note or
// a tag's name is new, or was embedded under another model or chunk size.
//
// It scores a copy of the notebook taken under the read lock, with the lock
// released: the Tagger reads the notebook back through Tags and Members, which
// take the lock themselves, and a read lock held across that would deadlock
// against a writer waiting in between.
func (nb *Notebook) View(ctx context.Context) (View, error) {
	nb.mu.RLock()
	tags := slices.Clone(nb.cat.Tags)
	notes := slices.Clone(nb.cat.Notes)
	for i := range notes {
		// a write edits a note's tags in place
		notes[i].Tags = slices.Clone(notes[i].Tags)
	}
	rev := revision(nb.rev)
	nb.mu.RUnlock()

	known := make(map[string]int, len(tags))
	v := View{Tags: make([]TagView, len(tags)), Notes: make([]NoteView, 0, len(notes))}
	for i, t := range tags {
		known[t.ID] = i
		v.Tags[i] = TagView{Tag: t}
	}
	start := time.Now()
	for _, n := range notes {
		text := itemText(n)
		nv := NoteView{Note: n, Auto: []string{}, Scores: []Score{}, Bytes: len(text), Parts: len(tagging.Chunk(text, nb.cfg.Info.ChunkBytes))}
		for _, id := range n.Tags {
			if i, ok := known[id]; ok {
				v.Tags[i].Hand++
			}
		}
		scores, err := nb.scope.Scores(ctx, rev, text)
		if err != nil {
			if ctx.Err() != nil {
				return View{}, ctx.Err()
			}
			nv.Error = err.Error()
		}
		for _, s := range scores {
			// a tag made since the copy was taken is left out
			i, ok := known[s.ID]
			if !ok {
				continue
			}
			nv.Scores = append(nv.Scores, Score{TagID: s.ID, Score: s.Score, Cutoff: s.Cutoff, ShadowedBy: s.ShadowedBy})
			if s.Picked() {
				nv.Auto = append(nv.Auto, s.ID)
				v.Tags[i].Auto++
			}
		}
		v.Notes = append(v.Notes, nv)
	}
	v.SuggestMs = time.Since(start).Milliseconds()
	return v, nil
}

// AddTag creates a tag, or returns the one with that name (case does not
// matter), so accepting a recommended tag twice makes one tag. With parents,
// a path's missing ancestors are created too: "dev / go / errors" makes
// "dev" and "dev / go" when they are not tags yet, so an accepted tree
// recommendation leaves the whole branch. A new level takes the spelling of
// the existing ancestors: "cooking / baking" under the tag "Cooking" becomes
// "Cooking / baking".
func (nb *Notebook) AddTag(_ context.Context, name string, parents bool) (Tag, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return Tag{}, fmt.Errorf("%w: blank tag name", ErrBadInput)
	}
	nb.mu.Lock()
	defer nb.mu.Unlock()
	if !parents {
		t := nb.addTag(name, nb.tagByName)
		return t, nb.save()
	}
	var t Tag
	for _, level := range pathPrefixes(name) {
		next := leaf(level)
		if t.Name != "" {
			next = t.Name + " / " + next
		}
		// levels match exactly: a root "web" is not the tag "dev / web"
		t = nb.addTag(next, nb.tagByPath)
	}
	return t, nb.save()
}

// addTag is AddTag for one name, reusing the tag find finds; the caller holds
// the lock and saves.
func (nb *Notebook) addTag(name string, find func(string) (Tag, bool)) Tag {
	if t, ok := find(name); ok {
		return t
	}
	t := Tag{ID: nb.uniqueTagID(slug(name)), Name: name}
	nb.cat.Tags = append(nb.cat.Tags, t)
	nb.changed(t.ID)
	return t
}

// pathPrefixes returns a path's ancestors and the path itself, root first:
// "a / b / c" gives "a", "a / b" and "a / b / c".
func pathPrefixes(path string) []string {
	var levels, out []string
	for _, l := range strings.Split(path, "/") {
		if l = strings.TrimSpace(l); l != "" {
			levels = append(levels, l)
			out = append(out, strings.Join(levels, " / "))
		}
	}
	return out
}

// RenameTag gives a tag a new name, which the Tagger embeds as its prior.
func (nb *Notebook) RenameTag(_ context.Context, id, name string) (Tag, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return Tag{}, fmt.Errorf("%w: blank tag name", ErrBadInput)
	}
	nb.mu.Lock()
	defer nb.mu.Unlock()
	i := nb.tagIndex(id)
	if i < 0 {
		return Tag{}, ErrNotFound
	}
	if other, ok := nb.tagByName(name); ok && other.ID != id {
		return Tag{}, fmt.Errorf("%w: a tag named %q exists", ErrBadInput, other.Name)
	}
	nb.cat.Tags[i].Name = name
	nb.changed() // a rename keeps the tag's revision: its members are the same
	return nb.cat.Tags[i], nb.save()
}

// DeleteTag deletes a tag and takes it off every note.
func (nb *Notebook) DeleteTag(_ context.Context, id string) error {
	nb.mu.Lock()
	defer nb.mu.Unlock()
	i := nb.tagIndex(id)
	if i < 0 {
		return ErrNotFound
	}
	nb.cat.Tags = slices.Delete(nb.cat.Tags, i, i+1)
	for j := range nb.cat.Notes {
		nb.cat.Notes[j].Tags = slices.DeleteFunc(nb.cat.Notes[j].Tags, func(t string) bool { return t == id })
	}
	nb.changed()
	delete(nb.tagRev, id)
	return nb.save()
}

// AddNote adds a note; the Tagger embeds it on the next View.
func (nb *Notebook) AddNote(_ context.Context, title, text string) (Note, error) {
	title, text = strings.TrimSpace(title), strings.TrimSpace(text)
	if text == "" {
		return Note{}, fmt.Errorf("%w: blank note text", ErrBadInput)
	}
	nb.mu.Lock()
	defer nb.mu.Unlock()
	nb.cat.Next++
	n := Note{ID: "n" + strconv.Itoa(nb.cat.Next), Title: title, Text: text, Tags: []string{}}
	for nb.noteIndex(n.ID) >= 0 {
		nb.cat.Next++
		n.ID = "n" + strconv.Itoa(nb.cat.Next)
	}
	nb.cat.Notes = append(nb.cat.Notes, n)
	nb.changed()
	return n, nb.save()
}

// DeleteNote deletes a note; the tags it carried by hand forget it.
func (nb *Notebook) DeleteNote(_ context.Context, id string) error {
	nb.mu.Lock()
	defer nb.mu.Unlock()
	i := nb.noteIndex(id)
	if i < 0 {
		return ErrNotFound
	}
	carried := nb.cat.Notes[i].Tags
	nb.cat.Notes = slices.Delete(nb.cat.Notes, i, i+1)
	nb.changed(carried...)
	return nb.save()
}

// TagNote puts a tag on a note by hand, which teaches the tag.
func (nb *Notebook) TagNote(_ context.Context, noteID, tagID string) error {
	nb.mu.Lock()
	defer nb.mu.Unlock()
	i := nb.noteIndex(noteID)
	if i < 0 || nb.tagIndex(tagID) < 0 {
		return ErrNotFound
	}
	if slices.Contains(nb.cat.Notes[i].Tags, tagID) {
		return nil
	}
	nb.cat.Notes[i].Tags = append(nb.cat.Notes[i].Tags, tagID)
	nb.changed(tagID)
	return nb.save()
}

// UntagNote takes a hand-made tag off a note; one the note does not carry is
// a no-op.
func (nb *Notebook) UntagNote(_ context.Context, noteID, tagID string) error {
	nb.mu.Lock()
	defer nb.mu.Unlock()
	i := nb.noteIndex(noteID)
	if i < 0 {
		return ErrNotFound
	}
	if !slices.Contains(nb.cat.Notes[i].Tags, tagID) {
		return nil
	}
	nb.cat.Notes[i].Tags = slices.DeleteFunc(nb.cat.Notes[i].Tags, func(t string) bool { return t == tagID })
	nb.changed(tagID)
	return nb.save()
}

// Recommended is one tag the Generator wrote for a note.
type Recommended struct {
	Name  string  `json:"name"`
	Score float64 `json:"score"`
	// Above says the score reaches the cutoff.
	Above bool `json:"above"`
	// TagID is the notebook's tag of that name (or, for a flat name, leaf
	// name), if one exists.
	TagID string `json:"tagId,omitempty"`
	// Known is how many leading levels of a tree tag are a tag of the
	// notebook already, compared as whole paths: 2 for "dev / go / errors"
	// when "dev / go" exists, whether or not "dev" does.
	Known int `json:"known"`
}

// RecommendOptions say how the tags are written (see tagging.GeneratorOption).
type RecommendOptions struct {
	// Language is the BCP 47 tag of the language the tags are written in, such
	// as "en"; empty writes them in the note's own language.
	Language string `json:"language"`
	// Tree writes paths that extend the notebook's tags, as a small tree.
	Tree bool `json:"tree"`
}

// Recommendation is what Recommend found for a note.
type Recommendation struct {
	Tags     []Recommended    `json:"tags"`
	MinScore float64          `json:"minScore"`
	Ms       int64            `json:"ms"`
	Options  RecommendOptions `json:"options"`
	// Error is set when the model failed on a part of a long note: Tags are
	// those of the rest, so topics of that part may be missing (see
	// tagging.ErrPartial).
	Error string `json:"error,omitempty"`
}

// Recommend has a Generator write new tags for a note, as opts say. Every tag
// the model wrote comes back, best first, those under the cutoff marked: a
// debug view wants to see what was dropped. The first call loads the chat
// model. A Generator keeps no state, so each call builds its own: in tree mode
// the notebook's tags as they are now are the tree it extends. When the model
// fails on some windows of a long note and not on others, the tags of the
// windows it read come back with Recommendation.Error saying so, as a note's
// scores do in a View; any other failure is the error of Recommend.
func (nb *Notebook) Recommend(ctx context.Context, noteID string, opts RecommendOptions) (Recommendation, error) {
	nb.mu.RLock()
	i := nb.noteIndex(noteID)
	if i < 0 {
		nb.mu.RUnlock()
		return Recommendation{}, ErrNotFound
	}
	text := itemText(nb.cat.Notes[i])
	genOpts := []tagging.GeneratorOption{tagging.WithNativeLanguageTags()}
	if opts.Language != "" {
		genOpts = []tagging.GeneratorOption{tagging.WithLanguageTags(opts.Language)}
	}
	if opts.Tree {
		paths := make([]string, len(nb.cat.Tags))
		for j, t := range nb.cat.Tags {
			paths[j] = t.Name
		}
		slices.Sort(paths) // siblings next to each other: the model reads a tree
		genOpts = append(genOpts, tagging.WithTreeTags(paths...))
	}
	nb.mu.RUnlock()

	gen, err := tagging.NewGenerator(nb.cfg.Chat, nb.cfg.Embedder, tagging.GeneratorConfig{}, genOpts...)
	if err != nil {
		return Recommendation{}, fmt.Errorf("%w: %w", ErrBadInput, err)
	}
	start := time.Now()
	tags, err := gen.Generate(ctx, text, -1)
	partial := ""
	switch {
	case errors.Is(err, tagging.ErrPartial):
		// the model failed on some windows of a long note: the tags of the
		// others are good, so they are shown, with what went wrong
		nb.logger.Warn("recommended tags cover a part of the note only", "note", noteID, "err", err)
		partial = err.Error()
	case err != nil:
		return Recommendation{}, err
	}
	res := Recommendation{Tags: make([]Recommended, 0, len(tags)), MinScore: nb.cfg.Info.MinScore, Ms: time.Since(start).Milliseconds(), Options: opts, Error: partial}
	nb.mu.RLock()
	defer nb.mu.RUnlock()
	for _, t := range tags {
		r := Recommended{Name: t.Name, Score: t.Score, Above: t.Score >= nb.cfg.Info.MinScore}
		if existing, ok := nb.tagByName(t.Name); ok {
			r.TagID = existing.ID
		}
		if strings.Contains(t.Name, "/") {
			for k, p := range pathPrefixes(t.Name) {
				if _, ok := nb.tagByPath(p); ok {
					r.Known = k + 1
				}
			}
		}
		res.Tags = append(res.Tags, r)
	}
	return res, nil
}

// SampleStats says what LoadSamples imported.
type SampleStats struct {
	Tags     int   `json:"tags"`
	Notes    int   `json:"notes"`
	HandTags int   `json:"handTags"`
	Ms       int64 `json:"ms"`
}

// LoadSamples imports the labelled samples: their tag taxonomy as tags, and
// their documents as notes, the expected tags kept for comparison. What the
// notebook has already is skipped. With perTag above zero, each tag is also
// put by hand on the first perTag documents that expect it, in file order, so
// the tags start with examples to learn from; the rest of the notes show what
// the Tagger makes of them. A sample note imported earlier gets its hand tags
// too, so the baseline can be trained in place.
func (nb *Notebook) LoadSamples(ctx context.Context, perTag int) (SampleStats, error) {
	if nb.cfg.SamplesFile == "" {
		return SampleStats{}, ErrNoSamples
	}
	if perTag < 0 {
		return SampleStats{}, fmt.Errorf("%w: negative examples per tag", ErrBadInput)
	}
	ds, err := eval.Load(nb.cfg.SamplesFile)
	if err != nil {
		return SampleStats{}, fmt.Errorf("autotag: samples: %w", err)
	}
	start := time.Now()
	st, texts, rev, err := nb.importSamples(ds, perTag)
	if err != nil {
		return st, err
	}
	// embed the new notes now, with the lock released, so the next View does
	// not pay for them
	if err := nb.scope.Warm(ctx, rev, texts...); err != nil {
		return st, fmt.Errorf("autotag: warm the tagger: %w", err)
	}
	st.Ms = time.Since(start).Milliseconds()
	nb.logger.Info("samples loaded", "tags", st.Tags, "notes", st.Notes, "handTags", st.HandTags, "elapsed", time.Since(start).Round(time.Millisecond))
	return st, nil
}

// importSamples is LoadSamples under the lock: it adds what the notebook
// lacks of ds and saves, and returns the texts of the notes it added and the
// notebook's revision after them.
func (nb *Notebook) importSamples(ds *eval.Dataset, perTag int) (SampleStats, []string, string, error) {
	var st SampleStats
	var texts []string
	nb.mu.Lock()
	defer nb.mu.Unlock()
	for _, name := range ds.Tags {
		if _, ok := nb.tagByName(name); ok {
			continue
		}
		t := Tag{ID: nb.uniqueTagID(slug(name)), Name: name}
		nb.cat.Tags = append(nb.cat.Tags, t)
		nb.changed(t.ID)
		st.Tags++
	}
	for _, d := range ds.Documents {
		if nb.noteIndex(d.ID) >= 0 {
			continue
		}
		n := Note{ID: d.ID, Title: d.Title, Text: strings.TrimSpace(d.Excerpt + "\n\n" + d.Text), Tags: []string{}, Expected: d.Tags}
		nb.cat.Notes = append(nb.cat.Notes, n)
		nb.changed()
		texts = append(texts, itemText(n))
		st.Notes++
	}
	nb.handTagSamples(ds.Documents, perTag, &st)
	return st, texts, revision(nb.rev), nb.save()
}

// handTagSamples puts each tag by hand on the first perTag documents that
// expect it. The caller holds the lock.
func (nb *Notebook) handTagSamples(docs []eval.Document, perTag int, st *SampleStats) {
	used := map[string]int{}
	for _, d := range docs {
		i := nb.noteIndex(d.ID)
		if i < 0 {
			continue
		}
		for _, name := range d.Tags {
			if used[name] >= perTag {
				continue
			}
			used[name]++
			tag, ok := nb.tagByName(name)
			if !ok || slices.Contains(nb.cat.Notes[i].Tags, tag.ID) {
				continue
			}
			nb.cat.Notes[i].Tags = append(nb.cat.Notes[i].Tags, tag.ID)
			nb.changed(tag.ID)
			st.HandTags++
		}
	}
}

// tagByName finds a tag by its name or, for a name that is not a path,
// failing that by its leaf: a generated "baking" matches the tag "cooking /
// baking", while the path "dev / go / errors" matches only itself. Case and
// the spacing around slashes do not matter. The caller holds the lock.
func (nb *Notebook) tagByName(name string) (Tag, bool) {
	if t, ok := nb.tagByPath(name); ok {
		return t, true
	}
	name = strings.ToLower(strings.TrimSpace(name))
	if strings.Contains(name, "/") {
		return Tag{}, false
	}
	for _, t := range nb.cat.Tags {
		if strings.ToLower(leaf(t.Name)) == name {
			return t, true
		}
	}
	return Tag{}, false
}

// tagByPath finds the tag whose whole name is name, compared by pathKey:
// no leaf matching. The caller holds the lock.
func (nb *Notebook) tagByPath(name string) (Tag, bool) {
	key := pathKey(name)
	for _, t := range nb.cat.Tags {
		if pathKey(t.Name) == key {
			return t, true
		}
	}
	return Tag{}, false
}

// pathKey is how two tag names compare: lowercased, levels trimmed, so
// "Dev/Go" and "dev / go" are one tag.
func pathKey(name string) string {
	p := pathPrefixes(name)
	if len(p) == 0 {
		return ""
	}
	return strings.ToLower(p[len(p)-1])
}

func leaf(name string) string {
	if i := strings.LastIndex(name, "/"); i >= 0 {
		return strings.TrimSpace(name[i+1:])
	}
	return name
}

func (nb *Notebook) tagIndex(id string) int {
	return slices.IndexFunc(nb.cat.Tags, func(t Tag) bool { return t.ID == id })
}

func (nb *Notebook) noteIndex(id string) int {
	return slices.IndexFunc(nb.cat.Notes, func(n Note) bool { return n.ID == id })
}

// uniqueTagID returns base, or base-2, base-3... if it is taken.
func (nb *Notebook) uniqueTagID(base string) string {
	if base == "" {
		base = "tag"
	}
	id := base
	for i := 2; nb.tagIndex(id) >= 0; i++ {
		id = base + "-" + strconv.Itoa(i)
	}
	return id
}

// slug turns a tag name into an id: "cooking / baking" becomes
// "cooking-baking". Letters of any script stay.
func slug(name string) string {
	var b strings.Builder
	dash := false
	for _, r := range strings.ToLower(name) {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			dash = b.Len() > 0
			continue
		}
		if dash {
			b.WriteByte('-')
			dash = false
		}
		b.WriteRune(r)
	}
	return b.String()
}

// save writes the catalog, if there is a file; the caller holds the lock.
// It writes a temporary file and renames it, so a crash never leaves half a
// notebook.
func (nb *Notebook) save() error {
	if nb.cfg.File == "" {
		return nil
	}
	data, err := json.MarshalIndent(nb.cat, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(nb.cfg.File), 0o750); err != nil {
		return fmt.Errorf("autotag: save notebook: %w", err)
	}
	tmp := nb.cfg.File + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("autotag: save notebook: %w", err)
	}
	if err := os.Rename(tmp, nb.cfg.File); err != nil {
		return fmt.Errorf("autotag: save notebook: %w", err)
	}
	return nil
}

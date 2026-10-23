// Package storetest is the conformance suite for autotag.Store. Every backend
// runs it from its own tests, so that all of them keep the contract the same
// way:
//
//	func TestConformance(t *testing.T) {
//		storetest.Run(t, func(t *testing.T) autotag.Store { return open(t) })
//	}
package storetest

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"testing"

	"github.com/go-bumbu/langai/autotag"
)

// idx and part are the scope the tests work in, unless a test is about telling
// scopes apart.
const idx, part = "tenant", "space"

// v1 and v2 are vectors at the edges of float32: the largest value, the
// smallest normal one, a negative and a zero. No test changes them; one that
// needs a vector to change clones it first.
var (
	v1 = []float32{1.5, -2.25, 3.4028235e38}
	v2 = []float32{1.1754944e-38, 0, -1}
)

// Run runs the suite against the Store newStore returns. Each subtest calls
// newStore with its own t, so a backend can tie what it opens to t.Cleanup, and
// the store it returns must hold nothing yet.
func Run(t *testing.T, newStore func(t *testing.T) autotag.Store) {
	cases := []struct {
		name string
		run  func(t *testing.T, s autotag.Store)
	}{
		{"TagLifecycle", tagLifecycle},
		{"SetTagVectorsChecksTheVersion", setTagVectorsChecksTheVersion},
		{"TagsAreOrderedByID", tagsAreOrderedByID},
		{"DeleteTagDropsItsLinks", deleteTagDropsItsLinks},
		{"ItemLifecycle", itemLifecycle},
		{"LinkBumpsTheTagOnce", linkBumpsTheTagOnce},
		{"LinkRejectsUnknownIDs", linkRejectsUnknownIDs},
		{"PutItemBumpsItsTags", putItemBumpsItsTags},
		{"DeleteItemBumpsItsTagsAndDropsLinks", deleteItemBumpsItsTagsAndDropsLinks},
		{"MembersAreOrderedByItemID", membersAreOrderedByItemID},
		{"ScopesAreIsolated", scopesAreIsolated},
		{"VectorsRoundTrip", vectorsRoundTrip},
		{"EmptyIndexIsAnError", emptyIndexIsAnError},
		{"StoredSlicesAreCopies", storedSlicesAreCopies},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) { c.run(t, newStore(t)) })
	}
}

// tagLifecycle: a tag is created at Version 1, a rename bumps it and drops the
// name vector, and the current name changes nothing.
func tagLifecycle(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)

	if !sc.putTag("t1", "a") {
		t.Error("PutTag of a new tag reported no change")
	}
	if got, want := sc.tag("t1"), (autotag.TagRecord{ID: "t1", Name: "a", Version: 1}); !reflect.DeepEqual(got, want) {
		t.Errorf("new tag = %+v, want %+v", got, want)
	}

	if sc.putTag("t1", "a") {
		t.Error("PutTag with the current name reported a change")
	}
	sc.wantVersion("t1", 1, "after PutTag with the current name")

	if !sc.setTagVectors("t1", 1, autotag.TagVectors{NameVec: v1, Vec: v2, Cutoff: 0.5, Model: "m"}) {
		t.Fatal("SetTagVectors at the tag's Version stored nothing")
	}
	if !sc.putTag("t1", "b") {
		t.Error("PutTag with a new name reported no change")
	}
	if got := sc.tag("t1"); got.Name != "b" || got.Version != 2 || got.NameVec != nil {
		t.Errorf("renamed tag = %+v, want Name b, Version 2, NameVec nil", got)
	}

	if _, ok := sc.lookupTag("unknown"); ok {
		t.Error("Tag found an id that was never put")
	}
}

// setTagVectorsChecksTheVersion: vectors are stored only for the Version they
// were computed from, so a change that lands meanwhile is never overwritten.
func setTagVectorsChecksTheVersion(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)
	sc.putTag("t1", "a")

	if !sc.setTagVectors("t1", 1, autotag.TagVectors{NameVec: v1, Vec: v2, Cutoff: 0.5, Model: "m"}) {
		t.Fatal("SetTagVectors at the tag's Version stored nothing")
	}
	want := autotag.TagRecord{ID: "t1", Name: "a", NameVec: v1, Vec: v2, Cutoff: 0.5, Model: "m", Version: 1, Built: 1}
	if got := sc.tag("t1"); !reflect.DeepEqual(got, want) {
		t.Fatalf("tag after SetTagVectors = %+v, want %+v", got, want)
	}

	for _, stale := range []int64{0, 2} {
		if sc.setTagVectors("t1", stale, autotag.TagVectors{NameVec: v2, Vec: v1, Cutoff: 0.9, Model: "other"}) {
			t.Errorf("SetTagVectors at Version %d of a tag at Version 1 stored its vectors", stale)
		}
	}
	if got := sc.tag("t1"); !reflect.DeepEqual(got, want) {
		t.Errorf("tag after the refused SetTagVectors = %+v, want it unchanged: %+v", got, want)
	}

	if sc.setTagVectors("unknown", 1, autotag.TagVectors{NameVec: v1, Vec: v2, Model: "m"}) {
		t.Error("SetTagVectors of a tag that was never put stored its vectors")
	}
}

// tagsAreOrderedByID: Tags lists the tags by ID, whatever order they were put in.
func tagsAreOrderedByID(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)
	if got := sc.tags(); len(got) != 0 {
		t.Fatalf("Tags of an empty scope = %+v, want none", got)
	}

	for _, id := range []string{"b", "a", "c"} {
		sc.putTag(id, "tag "+id)
	}
	if got, want := tagIDs(sc.tags()), []string{"a", "b", "c"}; !slices.Equal(got, want) {
		t.Errorf("ids from Tags = %v, want %v", got, want)
	}
}

// deleteTagDropsItsLinks: a deleted tag takes its links with it, so a tag
// created later under the same id starts without members.
func deleteTagDropsItsLinks(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)
	sc.putTag("t1", "a")
	sc.putItem(newItem("i1"))
	sc.link("i1", "t1")

	sc.deleteTag("t1")
	if _, ok := sc.lookupTag("t1"); ok {
		t.Error("Tag found the tag after DeleteTag")
	}

	sc.putTag("t1", "a")
	if got := sc.members("t1"); len(got) != 0 {
		t.Errorf("Members of a tag created after its id was deleted = %+v, want none", got)
	}
	if got := sc.putItem(newItem("i1")); len(got) != 0 {
		t.Errorf("PutItem returned %v after the item's only tag was deleted, want none", got)
	}

	sc.deleteTag("unknown")
}

// itemLifecycle: PutItem creates an item and a second PutItem replaces it.
func itemLifecycle(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)

	first := autotag.ItemRecord{ID: "i1", Text: "x", Vec: v1, Model: "m"}
	if got := sc.putItem(first); len(got) != 0 {
		t.Errorf("PutItem of an item no tag is linked to returned %v, want none", got)
	}
	if got := sc.item("i1"); !reflect.DeepEqual(got, first) {
		t.Errorf("Item = %+v, want %+v", got, first)
	}

	second := autotag.ItemRecord{ID: "i1", Text: "y", Vec: v2, Model: "m"}
	sc.putItem(second)
	if got := sc.item("i1"); !reflect.DeepEqual(got, second) {
		t.Errorf("Item after a second PutItem = %+v, want %+v", got, second)
	}

	if _, ok := sc.lookupItem("unknown"); ok {
		t.Error("Item found an id that was never put")
	}
}

// linkBumpsTheTagOnce: a link changes the tag's Version when it is made and
// when it is removed, never when the call changes nothing.
func linkBumpsTheTagOnce(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)
	sc.putTag("t1", "a")
	sc.putItem(autotag.ItemRecord{ID: "i1", Text: "x", Vec: v1, Model: "m"})

	if !sc.link("i1", "t1") {
		t.Error("Link of two unlinked ids reported no change")
	}
	sc.wantVersion("t1", 2, "after Link")
	if sc.link("i1", "t1") {
		t.Error("Link of an existing link reported a change")
	}
	sc.wantVersion("t1", 2, "after Link of an existing link")

	want := []autotag.Member{{ItemID: "i1", Vec: v1, Model: "m"}}
	if got := sc.members("t1"); !reflect.DeepEqual(got, want) {
		t.Errorf("Members = %+v, want %+v", got, want)
	}

	if !sc.unlink("i1", "t1") {
		t.Error("Unlink of a link reported no change")
	}
	sc.wantVersion("t1", 3, "after Unlink")
	if sc.unlink("i1", "t1") {
		t.Error("Unlink without a link reported a change")
	}
	sc.wantVersion("t1", 3, "after Unlink without a link")
}

// linkRejectsUnknownIDs: Link names the id it cannot find, the item first, and
// a refused Link changes nothing. Unlink of ids that are not there is no error.
func linkRejectsUnknownIDs(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)
	sc.putTag("t1", "a")
	sc.putItem(newItem("i1"))
	ctx := t.Context()

	cases := []struct {
		name          string
		itemID, tagID string
		want          error
	}{
		{"an unknown item", "nope", "t1", autotag.ErrUnknownItem},
		{"an unknown tag", "i1", "nope", autotag.ErrUnknownTag},
		{"both unknown, the item is checked first", "nope", "nope", autotag.ErrUnknownItem},
	}
	for _, c := range cases {
		if _, err := s.Link(ctx, idx, part, c.itemID, c.tagID); !errors.Is(err, c.want) {
			t.Errorf("Link with %s: err = %v, want %v", c.name, err, c.want)
		}
	}
	sc.wantVersion("t1", 1, "after the refused Links")

	if changed, err := s.Unlink(ctx, idx, part, "nope", "nope"); changed || err != nil {
		t.Errorf("Unlink of unknown ids = %v, %v; want false, nil", changed, err)
	}
}

// putItemBumpsItsTags: replacing an item changes what every tag linked to it
// is built from, so those tags, and only those, are bumped and reported.
func putItemBumpsItsTags(t *testing.T, s autotag.Store) {
	sc := seedLinked(t, s)
	before := sc.versions("t1", "t2", "t3")

	replaced := autotag.ItemRecord{ID: "i1", Text: "changed", Vec: v2, Model: "m"}
	if got, want := sc.putItem(replaced), []string{"t1", "t2"}; !slices.Equal(got, want) {
		t.Errorf("PutItem returned %v, want %v", got, want)
	}
	if got, want := sc.versions("t1", "t2", "t3"), []int64{before[0] + 1, before[1] + 1, before[2]}; !slices.Equal(got, want) {
		t.Errorf("Versions of t1, t2, t3 after PutItem = %v, want %v", got, want)
	}
}

// deleteItemBumpsItsTagsAndDropsLinks: deleting an item bumps and reports the
// tags that were linked to it, and removes it from their members.
func deleteItemBumpsItsTagsAndDropsLinks(t *testing.T, s autotag.Store) {
	sc := seedLinked(t, s)
	before := sc.versions("t1", "t2", "t3")

	if got, want := sc.deleteItem("i1"), []string{"t1", "t2"}; !slices.Equal(got, want) {
		t.Errorf("DeleteItem returned %v, want %v", got, want)
	}
	if got, want := sc.versions("t1", "t2", "t3"), []int64{before[0] + 1, before[1] + 1, before[2]}; !slices.Equal(got, want) {
		t.Errorf("Versions of t1, t2, t3 after DeleteItem = %v, want %v", got, want)
	}
	for _, id := range []string{"t1", "t2"} {
		if got := sc.members(id); len(got) != 0 {
			t.Errorf("Members of %s after DeleteItem = %+v, want none", id, got)
		}
	}
	if _, ok := sc.lookupItem("i1"); ok {
		t.Error("Item found the item after DeleteItem")
	}

	if got := sc.deleteItem("unknown"); len(got) != 0 {
		t.Errorf("DeleteItem of an unknown item returned %v, want none", got)
	}
}

// membersAreOrderedByItemID: Members lists the items by ItemID, whatever order
// they were linked in, each with the vector and model it was put with.
func membersAreOrderedByItemID(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)
	sc.putTag("t1", "a")
	sc.putItem(autotag.ItemRecord{ID: "i1", Text: "x", Vec: v1, Model: "m1"})
	sc.putItem(autotag.ItemRecord{ID: "i2", Text: "y", Vec: v2, Model: "m2"})
	sc.link("i2", "t1")
	sc.link("i1", "t1")

	want := []autotag.Member{{ItemID: "i1", Vec: v1, Model: "m1"}, {ItemID: "i2", Vec: v2, Model: "m2"}}
	if got := sc.members("t1"); !reflect.DeepEqual(got, want) {
		t.Errorf("Members = %+v, want %+v", got, want)
	}
}

// scopesAreIsolated: the same ids in two partitions of one index, or in one
// partition of two indexes, are different tags and items, and nothing done in
// one scope shows in another.
func scopesAreIsolated(t *testing.T, s autotag.Store) {
	scopes := []struct {
		scope
		name, text string
	}{
		{newScope(t, s, "tenant", "space"), "tag in tenant/space", "text in tenant/space"},
		{newScope(t, s, "tenant", "other"), "tag in tenant/other", "text in tenant/other"},
		{newScope(t, s, "other", "space"), "tag in other/space", "text in other/space"},
	}
	for _, sc := range scopes {
		sc.putTag("t1", sc.name)
		sc.putItem(autotag.ItemRecord{ID: "i1", Text: sc.text, Vec: v1, Model: "m"})
	}
	for _, sc := range scopes {
		if got := sc.tag("t1").Name; got != sc.name {
			t.Errorf("Tag in %s/%s has the name %q, want %q", sc.index, sc.partition, got, sc.name)
		}
		if got := sc.item("i1").Text; got != sc.text {
			t.Errorf("Item in %s/%s has the text %q, want %q", sc.index, sc.partition, got, sc.text)
		}
		if got := sc.tags(); len(got) != 1 || got[0].Name != sc.name {
			t.Errorf("Tags in %s/%s = %+v, want only %q", sc.index, sc.partition, got, sc.name)
		}
	}

	scopes[0].link("i1", "t1")
	for _, sc := range scopes[1:] {
		if got := sc.members("t1"); len(got) != 0 {
			t.Errorf("Members in %s/%s after a Link in another scope = %+v, want none", sc.index, sc.partition, got)
		}
		sc.wantVersion("t1", 1, fmt.Sprintf("in %s/%s after a Link in another scope", sc.index, sc.partition))
	}

	// An item of one partition cannot be linked to a tag of another.
	scopes[0].putItem(newItem("i2"))
	scopes[1].putTag("t2", "tag of tenant/other")
	ctx := t.Context()
	if _, err := s.Link(ctx, "tenant", "other", "i2", "t2"); !errors.Is(err, autotag.ErrUnknownItem) {
		t.Errorf("Link of an item of tenant/space in tenant/other: err = %v, want %v", err, autotag.ErrUnknownItem)
	}
	if _, err := s.Link(ctx, "tenant", "space", "i2", "t2"); !errors.Is(err, autotag.ErrUnknownTag) {
		t.Errorf("Link to a tag of tenant/other in tenant/space: err = %v, want %v", err, autotag.ErrUnknownTag)
	}
}

// vectorsRoundTrip: every vector comes back exactly as it was stored, in the
// item, in its members entry and in the tag, and a vector never set stays nil.
func vectorsRoundTrip(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)
	sc.putTag("t1", "a")
	sc.putTag("never-built", "b")
	if rec := sc.tag("never-built"); rec.NameVec != nil || rec.Vec != nil {
		t.Errorf("tag never built has NameVec %v and Vec %v, want nil and nil", rec.NameVec, rec.Vec)
	}

	sc.putItem(newItem("i1"))
	sc.link("i1", "t1")
	for _, v := range [][]float32{v1, v2, {0.1}} {
		sc.putItem(autotag.ItemRecord{ID: "i1", Text: "x", Vec: v, Model: "m"})
		if got := sc.item("i1").Vec; !slices.Equal(got, v) {
			t.Errorf("Item vector = %v, want %v", got, v)
		}
		if got := sc.members("t1"); len(got) != 1 || !slices.Equal(got[0].Vec, v) {
			t.Errorf("Members vectors = %+v, want [%v]", got, v)
		}

		sc.setVectors("t1", autotag.TagVectors{NameVec: v, Vec: v, Model: "m"})
		if got := sc.tag("t1"); !slices.Equal(got.NameVec, v) || !slices.Equal(got.Vec, v) {
			t.Errorf("tag vectors = %v and %v, want %v twice", got.NameVec, got.Vec, v)
		}
	}
}

// emptyIndexIsAnError: no method works without an index, since it is what
// keeps one tenant's data from another's. The error is about the index: the
// store is empty, so a Link that skipped the check would fail on an unknown
// item, and an error alone would not show it.
func emptyIndexIsAnError(t *testing.T, s autotag.Store) {
	ctx := t.Context()
	calls := []struct {
		name string
		call func() error
	}{
		{"Tag", func() error { _, _, err := s.Tag(ctx, "", part, "t1"); return err }},
		{"Tags", func() error { _, err := s.Tags(ctx, "", part); return err }},
		{"PutTag", func() error { _, err := s.PutTag(ctx, "", part, "t1", "a"); return err }},
		{"SetTagVectors", func() error {
			_, err := s.SetTagVectors(ctx, "", part, "t1", 1, autotag.TagVectors{NameVec: v1, Vec: v2, Model: "m"})
			return err
		}},
		{"DeleteTag", func() error { return s.DeleteTag(ctx, "", part, "t1") }},
		{"Item", func() error { _, _, err := s.Item(ctx, "", part, "i1"); return err }},
		{"PutItem", func() error { _, err := s.PutItem(ctx, "", part, newItem("i1")); return err }},
		{"DeleteItem", func() error { _, err := s.DeleteItem(ctx, "", part, "i1"); return err }},
		{"Link", func() error { _, err := s.Link(ctx, "", part, "i1", "t1"); return err }},
		{"Unlink", func() error { _, err := s.Unlink(ctx, "", part, "i1", "t1"); return err }},
		{"Members", func() error { _, err := s.Members(ctx, "", part, "t1"); return err }},
	}
	for _, c := range calls {
		err := c.call()
		if err == nil {
			t.Errorf("%s with an empty index returned no error", c.name)
		} else if errors.Is(err, autotag.ErrUnknownItem) || errors.Is(err, autotag.ErrUnknownTag) {
			t.Errorf("%s with an empty index returned %q, which blames an id and not the index", c.name, err)
		}
	}
}

// storedSlicesAreCopies: a store keeps its own copy of every vector it is
// given, and hands out a copy of every vector it holds, so a caller that
// reuses a slice cannot change what is stored.
func storedSlicesAreCopies(t *testing.T, s autotag.Store) {
	sc := newScope(t, s, idx, part)
	item := autotag.ItemRecord{ID: "i1", Text: "x", Vec: slices.Clone(v1), Model: "m"}
	vecs := autotag.TagVectors{NameVec: slices.Clone(v1), Vec: slices.Clone(v2), Cutoff: 0.5, Model: "m"}
	sc.putTag("t1", "a")
	sc.putItem(item)
	sc.link("i1", "t1")
	sc.setVectors("t1", vecs)

	scribble(item.Vec, vecs.NameVec, vecs.Vec)
	wantStoredVectors(t, sc, "after the slices passed in were changed")

	scribble(sc.item("i1").Vec)
	rec := sc.tag("t1")
	scribble(rec.NameVec, rec.Vec)
	for _, r := range sc.tags() {
		scribble(r.NameVec, r.Vec)
	}
	for _, m := range sc.members("t1") {
		scribble(m.Vec)
	}
	wantStoredVectors(t, sc, "after the slices read were changed")
}

// wantStoredVectors checks that the scope still holds v1 as the vector of item
// i1 and as the name vector of tag t1, and v2 as the vector of t1.
func wantStoredVectors(t *testing.T, sc scope, when string) {
	t.Helper()
	if got := sc.item("i1").Vec; !slices.Equal(got, v1) {
		t.Errorf("Item vector %s = %v, want %v", when, got, v1)
	}
	if got := sc.tag("t1"); !slices.Equal(got.NameVec, v1) || !slices.Equal(got.Vec, v2) {
		t.Errorf("tag vectors %s = %v and %v, want %v and %v", when, got.NameVec, got.Vec, v1, v2)
	}
	if got := sc.members("t1"); len(got) != 1 || !slices.Equal(got[0].Vec, v1) {
		t.Errorf("Members vectors %s = %+v, want [%v]", when, got, v1)
	}
}

// scribble overwrites every element of vs, as a caller reusing its buffers would.
func scribble(vs ...[]float32) {
	for _, v := range vs {
		for i := range v {
			v[i] = -42
		}
	}
}

// newItem returns an item with the given id, and a text, vector and model that
// no test looks at.
func newItem(id string) autotag.ItemRecord {
	return autotag.ItemRecord{ID: id, Text: "text of " + id, Vec: v1, Model: "m"}
}

func tagIDs(tags []autotag.TagRecord) []string {
	ids := make([]string, len(tags))
	for i, tg := range tags {
		ids[i] = tg.ID
	}
	return ids
}

// seedLinked fills the default scope with the tags t1, t2 and t3 and the item
// i1, linked to t2 and then t1 but not to t3.
func seedLinked(t *testing.T, s autotag.Store) scope {
	sc := newScope(t, s, idx, part)
	for _, id := range []string{"t1", "t2", "t3"} {
		sc.putTag(id, "tag "+id)
	}
	sc.putItem(newItem("i1"))
	sc.link("i1", "t2")
	sc.link("i1", "t1")
	return sc
}

// scope is one (index, partition) of a store. Its methods make the calls a test
// expects to succeed and stop the test when one fails, so that a test reads as
// the calls it is about.
type scope struct {
	t                *testing.T
	store            autotag.Store
	index, partition string
}

func newScope(t *testing.T, s autotag.Store, index, partition string) scope {
	return scope{t: t, store: s, index: index, partition: partition}
}

func (sc scope) putTag(id, name string) bool {
	sc.t.Helper()
	changed, err := sc.store.PutTag(sc.t.Context(), sc.index, sc.partition, id, name)
	if err != nil {
		sc.t.Fatalf("PutTag(%q, %q): %v", id, name, err)
	}
	return changed
}

func (sc scope) setTagVectors(id string, version int64, v autotag.TagVectors) bool {
	sc.t.Helper()
	stored, err := sc.store.SetTagVectors(sc.t.Context(), sc.index, sc.partition, id, version, v)
	if err != nil {
		sc.t.Fatalf("SetTagVectors(%q, %d): %v", id, version, err)
	}
	return stored
}

// setVectors stores v for the tag at the Version it has now, and stops the test
// if the store refuses.
func (sc scope) setVectors(id string, v autotag.TagVectors) {
	sc.t.Helper()
	if !sc.setTagVectors(id, sc.tag(id).Version, v) {
		sc.t.Fatalf("SetTagVectors(%q) at the tag's current Version stored nothing", id)
	}
}

// lookupTag returns the tag and whether the store has it.
func (sc scope) lookupTag(id string) (autotag.TagRecord, bool) {
	sc.t.Helper()
	rec, ok, err := sc.store.Tag(sc.t.Context(), sc.index, sc.partition, id)
	if err != nil {
		sc.t.Fatalf("Tag(%q): %v", id, err)
	}
	return rec, ok
}

// tag returns a tag the test knows is there.
func (sc scope) tag(id string) autotag.TagRecord {
	sc.t.Helper()
	rec, ok := sc.lookupTag(id)
	if !ok {
		sc.t.Fatalf("Tag(%q) found nothing", id)
	}
	return rec
}

func (sc scope) tags() []autotag.TagRecord {
	sc.t.Helper()
	recs, err := sc.store.Tags(sc.t.Context(), sc.index, sc.partition)
	if err != nil {
		sc.t.Fatalf("Tags: %v", err)
	}
	return recs
}

// versions returns the Version of each of the tags.
func (sc scope) versions(ids ...string) []int64 {
	sc.t.Helper()
	out := make([]int64, len(ids))
	for i, id := range ids {
		out[i] = sc.tag(id).Version
	}
	return out
}

// wantVersion checks the Version of a tag; when says at which point of the
// test.
func (sc scope) wantVersion(id string, want int64, when string) {
	sc.t.Helper()
	if got := sc.tag(id).Version; got != want {
		sc.t.Errorf("Version of %s %s = %d, want %d", id, when, got, want)
	}
}

func (sc scope) deleteTag(id string) {
	sc.t.Helper()
	if err := sc.store.DeleteTag(sc.t.Context(), sc.index, sc.partition, id); err != nil {
		sc.t.Fatalf("DeleteTag(%q): %v", id, err)
	}
}

// lookupItem returns the item and whether the store has it.
func (sc scope) lookupItem(id string) (autotag.ItemRecord, bool) {
	sc.t.Helper()
	rec, ok, err := sc.store.Item(sc.t.Context(), sc.index, sc.partition, id)
	if err != nil {
		sc.t.Fatalf("Item(%q): %v", id, err)
	}
	return rec, ok
}

// item returns an item the test knows is there.
func (sc scope) item(id string) autotag.ItemRecord {
	sc.t.Helper()
	rec, ok := sc.lookupItem(id)
	if !ok {
		sc.t.Fatalf("Item(%q) found nothing", id)
	}
	return rec
}

func (sc scope) putItem(item autotag.ItemRecord) []string {
	sc.t.Helper()
	linked, err := sc.store.PutItem(sc.t.Context(), sc.index, sc.partition, item)
	if err != nil {
		sc.t.Fatalf("PutItem(%q): %v", item.ID, err)
	}
	return linked
}

func (sc scope) deleteItem(id string) []string {
	sc.t.Helper()
	linked, err := sc.store.DeleteItem(sc.t.Context(), sc.index, sc.partition, id)
	if err != nil {
		sc.t.Fatalf("DeleteItem(%q): %v", id, err)
	}
	return linked
}

func (sc scope) link(itemID, tagID string) bool {
	sc.t.Helper()
	changed, err := sc.store.Link(sc.t.Context(), sc.index, sc.partition, itemID, tagID)
	if err != nil {
		sc.t.Fatalf("Link(%q, %q): %v", itemID, tagID, err)
	}
	return changed
}

func (sc scope) unlink(itemID, tagID string) bool {
	sc.t.Helper()
	changed, err := sc.store.Unlink(sc.t.Context(), sc.index, sc.partition, itemID, tagID)
	if err != nil {
		sc.t.Fatalf("Unlink(%q, %q): %v", itemID, tagID, err)
	}
	return changed
}

func (sc scope) members(tagID string) []autotag.Member {
	sc.t.Helper()
	ms, err := sc.store.Members(sc.t.Context(), sc.index, sc.partition, tagID)
	if err != nil {
		sc.t.Fatalf("Members(%q): %v", tagID, err)
	}
	return ms
}

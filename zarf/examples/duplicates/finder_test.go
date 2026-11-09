package duplicates

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/image/draw"

	"github.com/go-bumbu/langai/search"
)

// scene is a picture with structure for a hash to hold on to: a diagonal
// colour gradient with a dozen rectangles on it, all decided by seed, so
// that two seeds give unrelated pictures.
func scene(seed uint64, w, h int) *image.RGBA {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	r0, g0, b0 := rng.IntN(256), rng.IntN(256), rng.IntN(256)
	for y := range h {
		for x := range w {
			t := (x*255/w + y*255/h) / 2
			img.Set(x, y, color.RGBA{uint8((r0 + t) % 256), uint8((g0 + 2*t) % 256), uint8((b0 + 255 - t) % 256), 255})
		}
	}
	for range 12 {
		x, y := rng.IntN(w), rng.IntN(h)
		rect := image.Rect(x, y, x+w/8+rng.IntN(w/3), y+h/8+rng.IntN(h/3))
		c := color.RGBA{uint8(rng.IntN(256)), uint8(rng.IntN(256)), uint8(rng.IntN(256)), 255}
		draw.Draw(img, rect, image.NewUniform(c), image.Point{}, draw.Src)
	}
	return img
}

// half returns src scaled to half its size, with a mark stamped on it: a
// copy that every hash still takes for the picture, a few bits away.
func half(src image.Image) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx()/2, b.Dy()/2))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, b, draw.Src, nil)
	mark := image.Rect(dst.Bounds().Dx()*7/8, dst.Bounds().Dy()*7/8, dst.Bounds().Dx()-2, dst.Bounds().Dy()-2)
	draw.Draw(dst, mark, image.NewUniform(color.White), image.Point{}, draw.Src)
	return dst
}

func pngBytes(t *testing.T, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func jpegBytes(t *testing.T, img image.Image, quality int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func write(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// fixture is a directory with a picture, the same bytes under another name
// (an exact duplicate), a half-size JPEG of it with a mark, in a
// subdirectory (a near-exact one), two unrelated pictures, a broken image,
// and files the scan must leave alone.
type fixture struct {
	dir    string
	finder *Finder
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	dir := t.TempDir()
	a := scene(1, 256, 192)
	write(t, filepath.Join(dir, "a.png"), pngBytes(t, a))
	write(t, filepath.Join(dir, "a-again.png"), pngBytes(t, a))
	write(t, filepath.Join(dir, "sub dir", "a-small.jpg"), jpegBytes(t, half(a), 70))
	write(t, filepath.Join(dir, "b.png"), pngBytes(t, scene(2, 256, 192)))
	write(t, filepath.Join(dir, "c.png"), pngBytes(t, scene(3, 200, 200)))
	write(t, filepath.Join(dir, "broken.jpg"), []byte("not a jpeg"))
	write(t, filepath.Join(dir, "notes.txt"), []byte("ignored"))
	write(t, filepath.Join(dir, ".hidden", "a.png"), pngBytes(t, a))
	f, err := New(Config{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	return &fixture{dir: dir, finder: f}
}

func scanned(t *testing.T) *fixture {
	t.Helper()
	f := newFixture(t)
	if _, err := f.finder.Scan(context.Background()); err != nil {
		t.Fatal(err)
	}
	return f
}

func ids(g Group) string {
	names := make([]string, len(g.Members))
	for i, m := range g.Members {
		names[i] = m.ID
	}
	return strings.Join(names, ",")
}

func TestNew_ValidatesAndListsTheFourHashes(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Error("an empty directory name was accepted")
	}
	f, err := New(Config{Dir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	kinds := f.Kinds()
	if len(kinds) != 4 || kinds[0].Name != DefaultKind || kinds[0].Bits != 64 || kinds[3].Name != "dhash-256" || kinds[3].Algorithm != "dhash" {
		t.Errorf("kinds: %+v", kinds)
	}
	if f.Limit() != 12 || len(f.Collections()) != 0 {
		t.Errorf("defaults: limit %d", f.Limit())
	}
}

func TestScan_HashesEveryPictureWithAllFourHashes(t *testing.T) {
	f := newFixture(t)
	stats, err := f.finder.Scan(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.Hashed != 5 || stats.Failed != 1 || stats.Unchanged != 0 || stats.Removed != 0 {
		t.Errorf("first scan: %+v", stats)
	}
	pics := f.finder.Pictures()
	if len(pics) != 6 || pics[0].ID != "a-again.png" || pics[5].ID != "sub dir/a-small.jpg" {
		t.Fatalf("catalog: %+v", pics)
	}
	for _, p := range pics {
		if p.ID == "broken.jpg" {
			if p.Error == "" {
				t.Error("the broken picture has no error")
			}
			continue
		}
		if p.Error != "" || p.Width == 0 || len(p.hashes) != 4 {
			t.Errorf("%s: %+v", p.ID, p)
		}
	}
}

func TestScan_KeepsUnchangedAndDropsRemoved(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	// nothing changed: nothing is decoded again, and the broken one is retried
	stats, err := f.finder.Scan(ctx)
	if err != nil || stats.Hashed != 0 || stats.Unchanged != 5 || stats.Failed != 1 {
		t.Errorf("second scan: %+v, %v", stats, err)
	}
	// one removed, one replaced by another picture
	if err := os.Remove(filepath.Join(f.dir, "b.png")); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(f.dir, "c.png"), pngBytes(t, scene(4, 300, 200)))
	stats, err = f.finder.Scan(ctx)
	if err != nil || stats.Hashed != 1 || stats.Unchanged != 3 || stats.Removed != 1 {
		t.Errorf("scan after changes: %+v, %v", stats, err)
	}
	if got := f.finder.LastScan(); got.Removed != 1 {
		t.Errorf("last scan: %+v", got)
	}
	if _, ok := f.finder.Path("b.png"); ok {
		t.Error("the removed picture is still served")
	}
	if path, ok := f.finder.Path("sub dir/a-small.jpg"); !ok || !strings.HasSuffix(filepath.ToSlash(path), "sub dir/a-small.jpg") {
		t.Errorf("path of a catalog picture: %q %v", path, ok)
	}
}

func TestScan_OneAtATimeAndAMissingDirectory(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t)
	f.finder.scanning.Store(true)
	if _, err := f.finder.Scan(ctx); !errors.Is(err, ErrScanning) {
		t.Errorf("a second scan: %v", err)
	}
	if !f.finder.Scanning() {
		t.Error("Scanning is false during a scan")
	}
	f.finder.scanning.Store(false)

	done, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := f.finder.Scan(done); !errors.Is(err, context.Canceled) {
		t.Errorf("a cancelled scan: %v", err)
	}

	empty, err := New(Config{Dir: filepath.Join(t.TempDir(), "missing")})
	if err != nil {
		t.Fatal(err)
	}
	if stats, err := empty.Scan(ctx); err != nil || stats.Hashed != 0 {
		t.Errorf("scan of a missing directory: %+v, %v", stats, err)
	}
	if g, err := empty.Groups("", -1); err != nil || len(g.Groups) != 0 || g.Nearest != nil || g.Pairs != 0 {
		t.Errorf("groups of nothing: %+v, %v", g, err)
	}
}

func TestGroups_ExactDuplicatesAtZeroBits(t *testing.T) {
	f := scanned(t)
	for _, k := range f.finder.Kinds() {
		// threshold 0: only the two files with the same pixels
		g, err := f.finder.Groups(k.Name, 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(g.Groups) != 1 || !g.Groups[0].Exact || ids(g.Groups[0]) != "a-again.png,a.png" || g.Groups[0].Max != 0 || g.MinScore != 1 {
			t.Errorf("%s at 0 bits: %+v", k.Name, g)
		}
		if g.Pictures != 5 || g.Pairs != 10 || g.Duplicates != 2 || g.Nearest == nil {
			t.Errorf("%s at 0 bits: %d pictures, %d pairs, %d duplicates", k.Name, g.Pictures, g.Pairs, g.Duplicates)
		}
	}
}

func TestGroups_NearExactDuplicatesAtTheDefaultThreshold(t *testing.T) {
	f := scanned(t)
	for _, k := range f.finder.Kinds() {
		// the hash's own threshold: the half-size JPEG joins, nothing else
		g, err := f.finder.Groups(k.Name, -1)
		if err != nil {
			t.Fatal(err)
		}
		if g.Threshold != k.Threshold || len(g.Groups) != 1 || ids(g.Groups[0]) != "a-again.png,a.png,sub dir/a-small.jpg" || g.Groups[0].Exact {
			t.Fatalf("%s at its default: %+v", k.Name, g)
		}
		gr := g.Groups[0]
		if gr.Distances[0][1] != 0 || gr.Distances[0][2] != gr.Max || gr.Distances[2][0] != gr.Max || gr.Max == 0 || gr.Max > k.Threshold {
			t.Errorf("%s distances: %v, max %d", k.Name, gr.Distances, gr.Max)
		}
		if len(gr.Members[0].Hash) != k.Bits/4 {
			t.Errorf("%s hash %q is not %d hex digits", k.Name, gr.Members[0].Hash, k.Bits/4)
		}
		// unrelated pictures are far from the threshold
		if g.Nearest == nil || g.Nearest.Distance <= 2*k.Threshold {
			t.Errorf("%s: closest pair outside the group: %+v", k.Name, g.Nearest)
		}
	}
}

func TestGroups_AWideThresholdLinksThroughACloserMember(t *testing.T) {
	f := scanned(t)
	// every picture is within the whole hash of every other: one group,
	// whose furthest pair is the largest distance there is
	g, err := f.finder.Groups("phash-64", 64)
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Groups) != 1 || len(g.Groups[0].Members) != 5 || g.Nearest != nil || g.MinScore != -1 {
		t.Errorf("groups at 64 bits: %+v", g)
	}
	if _, err := f.finder.Groups("phash-64", 65); !errors.Is(err, ErrBadInput) {
		t.Errorf("a threshold above the hash's size: %v", err)
	}
	if _, err := f.finder.Groups("md5", 0); !errors.Is(err, ErrBadInput) {
		t.Errorf("an unknown hash: %v", err)
	}
}

// upload is a copy of the fixture's a.png that is not in the catalog: the
// same picture as a JPEG.
func upload(t *testing.T) []byte {
	t.Helper()
	return jpegBytes(t, scene(1, 256, 192), 60)
}

func TestSearch_FindsCopiesThroughTheMediaIndex(t *testing.T) {
	f := scanned(t)
	if len(f.finder.Collections()) != 0 {
		t.Error("an index was opened before any search")
	}
	res, err := f.finder.Search(context.Background(), "", bytes.NewReader(upload(t)), -1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Kind.Name != DefaultKind || res.Threshold != 4 || res.MinScore != 0.875 || len(res.QueryHash) != 16 || res.Stored || res.Query != nil {
		t.Errorf("result: %+v", res)
	}
	if len(res.Hits) != 5 || res.Copies != 3 {
		t.Fatalf("%d hits, %d copies, want 5 and 3: %+v", len(res.Hits), res.Copies, res.Hits)
	}
	for i, h := range res.Hits {
		// the cosine is the Hamming distance read another way, and the raw
		// hashes agree with the index
		if want := 1 - 2*float64(h.Hamming)/64; h.Score != want {
			t.Errorf("hit %d: score %v is not 1 - 2·%d/64", i, h.Score, h.Hamming)
		}
		if h.Distance != h.Hamming || h.Hash == "" {
			t.Errorf("hit %d: Hash.Distance %d, from the cosine %d", i, h.Distance, h.Hamming)
		}
		if copyOfA := strings.HasPrefix(h.Picture.Name, "a"); h.Within != copyOfA || (i < 3) != copyOfA {
			t.Errorf("hit %d %s: within %v", i, h.Picture.ID, h.Within)
		}
	}
}

func TestSearch_OpensACollectionPerHash(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	if _, err := f.finder.Search(ctx, "", bytes.NewReader(upload(t)), -1, 0); err != nil {
		t.Fatal(err)
	}
	// the search opened that hash's index, and only that one
	cols := f.finder.Collections()
	if len(cols) != 1 || cols[0].Kind != DefaultKind || cols[0].Coord.Collection != "duplicates-phash-64" {
		t.Fatalf("collections: %+v", cols)
	}
	if model, err := cols[0].Store.Model(ctx, cols[0].Coord); err != nil || model != DefaultKind {
		t.Errorf("the store records %q, %v", model, err)
	}
	docs, err := cols[0].Store.Docs(ctx, cols[0].Coord, "", 0)
	if err != nil || len(docs) != 5 {
		t.Errorf("the index holds %d pictures, %v", len(docs), err)
	}

	// 256 bits is another space, with a collection of its own
	res, err := f.finder.Search(ctx, "dhash-256", bytes.NewReader(upload(t)), 20, 2)
	if err != nil || len(res.Hits) != 2 || res.Kind.Bits != 256 || len(res.QueryHash) != 64 || res.Copies != 2 {
		t.Errorf("dhash-256: %+v, %v", res, err)
	}
	if len(f.finder.Collections()) != 2 {
		t.Errorf("collections after a second hash: %+v", f.finder.Collections())
	}
}

func TestSearch_RefusesBadInput(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	if _, err := f.finder.Search(ctx, "", strings.NewReader("not a picture"), -1, 0); err == nil {
		t.Error("a file that is no picture was searched with")
	}
	if _, err := f.finder.Search(ctx, "md5", bytes.NewReader(upload(t)), -1, 0); !errors.Is(err, ErrBadInput) {
		t.Errorf("an unknown hash: %v", err)
	}
	if _, err := f.finder.Search(ctx, "", bytes.NewReader(upload(t)), 65, 0); !errors.Is(err, ErrBadInput) {
		t.Errorf("a threshold above the hash's size: %v", err)
	}
}

func TestSimilar_SearchesWithTheStoredVector(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	res, err := f.finder.Similar(ctx, "", "a.png", -1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Stored || res.Query == nil || res.Query.ID != "a.png" || len(res.Hits) != 4 || res.Copies != 2 {
		t.Fatalf("similar: %+v", res)
	}
	if res.Hits[0].Picture.ID != "a-again.png" || res.Hits[0].Score != 1 || res.Hits[0].Hamming != 0 || res.Hits[0].Distance != 0 {
		t.Errorf("the exact duplicate: %+v", res.Hits[0])
	}
	for _, h := range res.Hits {
		if h.Picture.ID == "a.png" {
			t.Error("the picture is among its own copies")
		}
	}
	if _, err := f.finder.Similar(ctx, "", "nope.png", -1, 0); !errors.Is(err, ErrNotFound) {
		t.Errorf("similar to an unknown picture: %v", err)
	}
	// a picture that could not be hashed is in the catalog, not in the index
	if _, err := f.finder.Similar(ctx, "", "broken.jpg", -1, 0); !errors.Is(err, search.ErrNotIndexed) {
		t.Errorf("similar to a picture that is not indexed: %v", err)
	}
	if _, err := f.finder.Similar(ctx, "md5", "a.png", -1, 0); !errors.Is(err, ErrBadInput) {
		t.Errorf("an unknown hash: %v", err)
	}
}

func TestIndex_FollowsTheCatalogAndResets(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	if _, err := f.finder.Similar(ctx, "", "a.png", -1, 0); err != nil {
		t.Fatal(err)
	}
	col := f.finder.Collections()[0]
	count := func() int {
		docs, err := col.Store.Docs(ctx, col.Coord, "", 0)
		if err != nil {
			t.Fatal(err)
		}
		return len(docs)
	}
	// a scan after a picture left and another changed brings the index in step
	if err := os.Remove(filepath.Join(f.dir, "b.png")); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(f.dir, "c.png"), pngBytes(t, scene(1, 256, 192)))
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(filepath.Join(f.dir, "c.png"), later, later); err != nil {
		t.Fatal(err)
	}
	if _, err := f.finder.Scan(ctx); err != nil {
		t.Fatal(err)
	}
	if n := count(); n != 4 {
		t.Errorf("the index holds %d pictures after the scan, want 4", n)
	}
	res, err := f.finder.Similar(ctx, "", "a.png", -1, 0)
	if err != nil || res.Copies != 3 {
		t.Errorf("c.png is now a copy of a.png: %d copies, %v", res.Copies, err)
	}

	if err := f.finder.Reset(ctx, DefaultKind); err != nil {
		t.Fatal(err)
	}
	if n := count(); n != 0 {
		t.Errorf("the index holds %d pictures after the reset", n)
	}
	// the next search builds it again
	if res, err := f.finder.Similar(ctx, "", "a.png", -1, 0); err != nil || res.Copies != 3 || count() != 4 {
		t.Errorf("after the reset: %d copies, %v", res.Copies, err)
	}
	// resetting a hash no search opened is a no-op, an unknown one an error
	if err := f.finder.Reset(ctx, "dhash-64"); err != nil {
		t.Errorf("reset of an unopened hash: %v", err)
	}
	if err := f.finder.Reset(ctx, "md5"); !errors.Is(err, ErrBadInput) {
		t.Errorf("reset of an unknown hash: %v", err)
	}
}

func TestDecode_RefusesWhatIsNoPicture(t *testing.T) {
	dir := t.TempDir()
	if _, err := decode(filepath.Join(dir, "missing.png")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a missing file: %v", err)
	}
	// a PNG header that announces more pixels than the scan decodes
	huge := pngBytes(t, image.NewGray(image.Rect(0, 0, 1, 1)))
	copy(huge[16:24], []byte{0, 0, 0x80, 0, 0, 0, 0x80, 0}) // 32768 × 32768
	binary.BigEndian.PutUint32(huge[29:33], crc32.ChecksumIEEE(huge[12:29]))
	write(t, filepath.Join(dir, "huge.png"), huge)
	if _, err := decode(filepath.Join(dir, "huge.png")); err == nil || !strings.Contains(err.Error(), "megapixels") {
		t.Errorf("a picture of too many pixels: %v", err)
	}
	// a header that decodes and pixels that do not
	cut := pngBytes(t, scene(1, 64, 64))[:60]
	write(t, filepath.Join(dir, "cut.png"), cut)
	if _, err := decode(filepath.Join(dir, "cut.png")); err == nil {
		t.Error("a truncated picture was decoded")
	}
}

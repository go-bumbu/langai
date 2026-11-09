package duplicates

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"io"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/go-bumbu/sense/provider/phash"
	"github.com/go-bumbu/sense/search"
)

// Member is one picture of a group.
type Member struct {
	Picture
	// Hash is the picture's hash in hexadecimal (phash.Hash.String): for a
	// 64-bit hash every two digits are one row of its 8×8 grid.
	Hash string `json:"hash"`
}

// Group is a set of pictures linked by pairs within the threshold.
type Group struct {
	Members []Member `json:"members"`
	// Distances[i][j] is the Hamming distance between members i and j.
	Distances [][]int `json:"distances"`
	// Max is the largest distance between two members. It can be above the
	// threshold: a picture joins a group through its closest member.
	Max int `json:"max"`
	// Exact says every member has the same hash: to the algorithm they are
	// one picture.
	Exact bool `json:"exact"`
}

// Pair is two pictures and the Hamming distance between them.
type Pair struct {
	A        string `json:"a"`
	B        string `json:"b"`
	Distance int    `json:"distance"`
}

// Groups is the duplicates of the catalog under one hash and threshold.
type Groups struct {
	Kind      Kind `json:"kind"`
	Threshold int  `json:"threshold"`
	// MinScore is the threshold as the cosine floor a MediaIndex over the
	// same hash would take: 1 - 2·threshold/bits.
	MinScore float64 `json:"minScore"`
	// Pictures is how many pictures were compared, Pairs how many pairs
	// that makes, and Duplicates how many pictures are in a group.
	Pictures   int     `json:"pictures"`
	Pairs      int     `json:"pairs"`
	Duplicates int     `json:"duplicates"`
	Groups     []Group `json:"groups"`
	// Nearest is the closest pair of pictures that are not in one group:
	// how far the threshold is from linking unrelated pictures. Nil when
	// every picture is in one group, or there are fewer than two.
	Nearest *Pair `json:"nearest,omitempty"`
	// ElapsedUs is what comparing every pair took, in microseconds.
	ElapsedUs int64 `json:"elapsedUs"`
}

// score is a Hamming distance as the cosine of the two hash vectors.
func score(distance, bits int) float64 { return 1 - 2*float64(distance)/float64(bits) }

// hamming is a cosine of two hash vectors as the Hamming distance.
func hamming(score float64, bits int) int {
	return int(math.Round((1 - score) * float64(bits) / 2))
}

// Groups compares every pair of hashed pictures by the Hamming distance of
// their raw hashes (phash.Hash.Distance) and links the ones at most
// threshold bits apart into groups: no store and no vectors. kindName ""
// is DefaultKind; a negative threshold is the hash's own default. A group
// whose hashes are all equal is exact; the others are near-exact
// duplicates.
func (f *Finder) Groups(kindName string, threshold int) (Groups, error) {
	k, err := f.kind(kindName)
	if err != nil {
		return Groups{}, err
	}
	if threshold < 0 {
		threshold = k.Threshold
	}
	if threshold > k.Bits {
		return Groups{}, fmt.Errorf("%w: a threshold of %d bits for a hash of %d", ErrBadInput, threshold, k.Bits)
	}
	start := time.Now()
	pics := f.Pictures()
	pics = slices.DeleteFunc(pics, func(p Picture) bool { return p.Error != "" })
	hashes := make([]phash.Hash, len(pics))
	for i, p := range pics {
		hashes[i] = p.hashes[k.Name]
	}

	// union-find over the pairs within the threshold
	parent := make([]int, len(pics))
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(i int) int {
		if parent[i] != i {
			parent[i] = find(parent[i])
		}
		return parent[i]
	}
	for i := range pics {
		for j := i + 1; j < len(pics); j++ {
			if hashes[i].Distance(hashes[j]) <= threshold {
				parent[find(i)] = find(j)
			}
		}
	}

	res := Groups{Kind: k.Kind, Threshold: threshold, MinScore: score(threshold, k.Bits), Pictures: len(pics), Pairs: len(pics) * (len(pics) - 1) / 2, Groups: []Group{}}
	byRoot := map[int][]int{}
	for i := range pics {
		byRoot[find(i)] = append(byRoot[find(i)], i)
	}
	for _, members := range byRoot {
		if len(members) < 2 {
			continue
		}
		res.Groups = append(res.Groups, group(pics, hashes, members))
		res.Duplicates += len(members)
	}
	// exact groups first, then the tightest, then by name
	slices.SortFunc(res.Groups, func(a, b Group) int {
		if a.Max != b.Max {
			return a.Max - b.Max
		}
		return strings.Compare(a.Members[0].ID, b.Members[0].ID)
	})
	// the closest pair across two groups: a pair above the threshold can
	// still be in one group, linked through a third picture
	for i := range pics {
		for j := i + 1; j < len(pics); j++ {
			if find(i) == find(j) {
				continue
			}
			if d := hashes[i].Distance(hashes[j]); res.Nearest == nil || d < res.Nearest.Distance {
				res.Nearest = &Pair{A: pics[i].ID, B: pics[j].ID, Distance: d}
			}
		}
	}
	res.ElapsedUs = time.Since(start).Microseconds()
	return res, nil
}

// group builds a Group from the indexes of its members, which are in
// catalog order.
func group(pics []Picture, hashes []phash.Hash, members []int) Group {
	g := Group{Members: make([]Member, len(members)), Distances: make([][]int, len(members)), Exact: true}
	for i, m := range members {
		g.Members[i] = Member{Picture: pics[m], Hash: hashes[m].String()}
		g.Distances[i] = make([]int, len(members))
		for j, n := range members {
			d := hashes[m].Distance(hashes[n])
			g.Distances[i][j] = d
			g.Max = max(g.Max, d)
			// comparable: equal hashes are one picture to the algorithm
			if hashes[m] != hashes[n] {
				g.Exact = false
			}
		}
	}
	return g
}

// Hit is one picture a search found.
type Hit struct {
	Picture Picture `json:"picture"`
	// Score is the cosine the MediaIndex returned, and Hamming the distance
	// it stands for: (1 - score)·bits/2.
	Score   float64 `json:"score"`
	Hamming int     `json:"hamming"`
	// Distance is the Hamming distance of the two raw hashes
	// (phash.Hash.Distance), computed without the index; -1 when a hash is
	// missing. It equals Hamming, except for a JPEG with an EXIF
	// orientation: the index turns such a picture upright before it hashes
	// it, and a raw Hash takes the pixels as they were decoded.
	Distance int `json:"distance"`
	// Within says the hit is at most the threshold away: a copy.
	Within bool   `json:"within"`
	Hash   string `json:"hash"`
}

// Result is what a search for copies returns.
type Result struct {
	Kind      Kind `json:"kind"`
	Threshold int  `json:"threshold"`
	// MinScore is the threshold as a cosine: 1 - 2·threshold/bits.
	MinScore float64 `json:"minScore"`
	// Query is the catalog picture searched with; nil for an upload.
	Query *Picture `json:"query,omitempty"`
	// QueryHash is the raw hash of the query picture, in hexadecimal.
	QueryHash string `json:"queryHash"`
	// Stored says the query was the picture's vector from the index, so
	// nothing was hashed.
	Stored bool `json:"stored,omitempty"`
	// Copies is how many hits are within the threshold.
	Copies    int   `json:"copies"`
	ElapsedUs int64 `json:"elapsedUs"`
	Hits      []Hit `json:"hits"`
}

// index returns the MediaIndex of a hash, in step with the catalog; the
// first call for a hash builds it.
func (f *Finder) index(ctx context.Context, k *kind) (*search.MediaIndex, error) {
	if err := f.sync(ctx, k, true); err != nil {
		return nil, err
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.index, nil
}

// Search finds the copies of an uploaded picture among the catalog's: the
// picture is hashed by a search.MediaIndex over the hash as an
// ImageEmbedder (SearchImage), exactly as a model's index would embed it,
// and each hit's cosine is shown next to the Hamming distance of the raw
// hashes. Every hit comes back, the ones within the threshold marked: a
// debug view wants to see what a floor would drop. A negative threshold is
// the hash's default; limit <= 0 the Finder's.
func (f *Finder) Search(ctx context.Context, kindName string, r io.Reader, threshold, limit int) (Result, error) {
	k, threshold, err := f.searchArgs(kindName, threshold)
	if err != nil {
		return Result{}, err
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return Result{}, err
	}
	ix, err := f.index(ctx, k)
	if err != nil {
		return Result{}, err
	}
	start := time.Now()
	matches, err := ix.SearchImage(ctx, bytes.NewReader(data), f.limit(limit))
	if err != nil {
		return Result{}, err
	}
	elapsed := time.Since(start)
	// the raw hash of the same bytes, to set next to the index's cosine
	var query *phash.Hash
	if img, _, err := image.Decode(bytes.NewReader(data)); err == nil {
		if h, err := k.hasher.Hash(img); err == nil {
			query = &h
		}
	}
	res := f.result(k, threshold, matches, query)
	res.ElapsedUs = elapsed.Microseconds()
	return res, nil
}

// Similar finds the copies of the catalog picture id, searching with its
// stored vector (search.MediaIndex.SearchIndexed): nothing is hashed, and
// the picture itself is not among the hits.
func (f *Finder) Similar(ctx context.Context, kindName, id string, threshold, limit int) (Result, error) {
	k, threshold, err := f.searchArgs(kindName, threshold)
	if err != nil {
		return Result{}, err
	}
	f.mu.Lock()
	p, ok := f.pictures[id]
	var pic Picture
	if ok {
		pic = *p
	}
	f.mu.Unlock()
	if !ok {
		return Result{}, ErrNotFound
	}
	ix, err := f.index(ctx, k)
	if err != nil {
		return Result{}, err
	}
	start := time.Now()
	matches, err := ix.SearchIndexed(ctx, id, f.limit(limit))
	if err != nil {
		return Result{}, err
	}
	elapsed := time.Since(start)
	var query *phash.Hash
	if h, ok := pic.hashes[k.Name]; ok {
		query = &h
	}
	res := f.result(k, threshold, matches, query)
	res.Query, res.Stored, res.ElapsedUs = &pic, true, elapsed.Microseconds()
	return res, nil
}

func (f *Finder) searchArgs(kindName string, threshold int) (*kind, int, error) {
	k, err := f.kind(kindName)
	if err != nil {
		return nil, 0, err
	}
	if threshold < 0 {
		threshold = k.Threshold
	}
	if threshold > k.Bits {
		return nil, 0, fmt.Errorf("%w: a threshold of %d bits for a hash of %d", ErrBadInput, threshold, k.Bits)
	}
	return k, threshold, nil
}

func (f *Finder) limit(limit int) int {
	if limit <= 0 {
		return f.cfg.Limit
	}
	return limit
}

// result joins the matches with the catalog and sets each cosine next to
// the Hamming distance it stands for, and to the distance of the raw
// hashes when the query's is known.
func (f *Finder) result(k *kind, threshold int, matches []search.MediaMatch, query *phash.Hash) Result {
	res := Result{Kind: k.Kind, Threshold: threshold, MinScore: score(threshold, k.Bits), Hits: make([]Hit, 0, len(matches))}
	if query != nil {
		res.QueryHash = query.String()
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, m := range matches {
		h := Hit{Picture: Picture{ID: m.DocID, Name: filepath.Base(m.DocID)}, Score: m.Score, Hamming: hamming(m.Score, k.Bits), Distance: -1}
		if p, ok := f.pictures[m.DocID]; ok {
			h.Picture = *p
			if own, ok := p.hashes[k.Name]; ok {
				h.Hash = own.String()
				if query != nil {
					h.Distance = query.Distance(own)
				}
			}
		}
		h.Within = h.Hamming <= threshold
		if h.Within {
			res.Copies++
		}
		res.Hits = append(res.Hits, h)
	}
	return res
}

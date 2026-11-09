package search

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"strings"
	"time"

	// the image formats IndexImage and SearchImage decode; an application
	// registers more with image.RegisterFormat
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/go-bumbu/langai"
	"github.com/go-bumbu/langai/media"
	"github.com/go-bumbu/langai/vector"
)

// maxImagePixels is the largest picture IndexImage and SearchImage decode:
// 256 megapixels, a gigabyte of RGBA, which is beyond any camera and well
// short of what a crafted one-kilobyte PNG can claim to be.
const maxImagePixels = 256 << 20

// MediaIndex is the Index for pictures: it embeds images and the sampled
// frames of videos with a langai.ImageEmbedder, keeps the vectors in a
// vector.Store and answers a query image or video with the indexed items that
// look most like it, across both kinds. It knows nothing about what the
// items are: callers hand it opaque ids, image bytes and video paths.
//
// Every item is a set of frames, an image being one frame at time zero, and
// each frame is one vector.Chunk whose Seq is the frame's time in
// milliseconds. A search embeds the query's frames, asks the store for the
// best chunk of each of the closest items for each frame and keeps the best
// pair per item, so a video is
// found by the one frame that resembles the query most. The coordinates
// (index, partition, collection) work as for Index; vectors of an image
// model must not share a collection with a text model's.
type MediaIndex struct {
	embedder langai.ImageEmbedder
	sampler  media.Sampler
	video    media.VideoEmbedder // whole segments instead of sampled frames, when set
	text     langai.Embedder     // text queries, when set
	store    vector.Store
	minScore float64
	coord    vector.Coord
	model    string // the image embedder's, when it names it
}

// MediaOption customizes a MediaIndex.
type MediaOption func(*MediaIndex)

// WithSampler sets the Sampler that turns a video file into frames; without
// one (or a VideoEmbedder), IndexVideo and SearchVideo fail. media/ffmpeg
// has the implementation over the ffmpeg binary.
func WithSampler(s media.Sampler) MediaOption {
	return func(ix *MediaIndex) { ix.sampler = s }
}

// WithVideoEmbedder makes IndexVideo and SearchVideo embed a video as whole
// segments through v, a model that takes video, instead of sampling frames
// for the image embedder: each segment is one chunk whose Seq is the
// segment's start, so a match says which stretch of the video it is in. v
// must share the image embedder's space (the same model behind both) and
// its dimensions. An index built this way must not share a collection with
// one built on frames: the same video gets different vectors.
func WithVideoEmbedder(v media.VideoEmbedder) MediaOption {
	return func(ix *MediaIndex) { ix.video = v }
}

// WithTextEmbedder enables SearchText: e embeds the query text into the
// image embedder's space (the text tower of the same model), with the same
// dimensions.
func WithTextEmbedder(e langai.Embedder) MediaOption {
	return func(ix *MediaIndex) { ix.text = e }
}

// WithMediaMinScore drops search hits scoring below min, for the reasons
// WithMinScore gives: a nearest-neighbour query always fills its limit, and
// only a floor turns "closest" into "looks alike". Where the floor sits
// depends on the model: with the pinned SigLIP 2 model a resized or
// recompressed copy scores above 0.9 and unrelated pictures well below.
func WithMediaMinScore(min float64) MediaOption {
	return func(ix *MediaIndex) { ix.minScore = min }
}

// NewMediaIndex returns a MediaIndex using the given embedder and store, over
// the collection at index, partition and collection, which match exactly as
// for New. When the image embedder names its model (langai.Identified), the
// store records it with the first vectors written to the collection and
// refuses another model's afterwards.
func NewMediaIndex(embedder langai.ImageEmbedder, store vector.Store, index, partition, collection string, opts ...MediaOption) (*MediaIndex, error) {
	if embedder == nil {
		return nil, errors.New("search: nil embedder")
	}
	if store == nil {
		return nil, errors.New("search: nil store")
	}
	ix := &MediaIndex{
		embedder: embedder, store: store, minScore: math.Inf(-1),
		coord: vector.Coord{Index: index, Partition: partition, Collection: collection},
		model: modelOf(embedder),
	}
	for _, opt := range opts {
		opt(ix)
	}
	if ix.video != nil && ix.video.Dimensions() != embedder.Dimensions() {
		return nil, fmt.Errorf("search: the video embedder's vectors have %d dimensions, the image embedder's %d: they must be one model", ix.video.Dimensions(), embedder.Dimensions())
	}
	if ix.text != nil && ix.text.Dimensions() != embedder.Dimensions() {
		return nil, fmt.Errorf("search: the text embedder's vectors have %d dimensions, the image embedder's %d: they must be one model", ix.text.Dimensions(), embedder.Dimensions())
	}
	return ix, nil
}

// MediaMatch is one search hit: the item and how much it looks like the
// query. For a video, At is the time of its frame that matched best; for a
// video query, QueryAt is the time of the query frame that frame matched.
// Both are 0 for images.
type MediaMatch struct {
	DocID   string
	Score   float64
	At      time.Duration
	QueryAt time.Duration
}

// IndexImage decodes the image r holds (any registered format), embeds it and
// stores it under docID, replacing whatever was indexed there before. It is a
// cheap no-op when the bytes are unchanged since the last call, so it is safe
// to call eagerly.
func (ix *MediaIndex) IndexImage(ctx context.Context, docID string, r io.Reader) error {
	if docID == "" {
		return ErrEmptyID
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return fmt.Errorf("search: read image %q: %w", docID, err)
	}
	hash := bytesHash(data)
	if same, err := ix.unchanged(ctx, docID, hash); err != nil || same {
		return err
	}
	img, err := decodeImage(bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("search: decode image %q: %w", docID, err)
	}
	return ix.indexFrames(ctx, docID, hash, []media.Frame{{Image: img}})
}

// IndexVideo embeds the video at path and stores it under docID, replacing
// whatever was indexed there before: as whole segments through the
// VideoEmbedder when one is set (see WithVideoEmbedder), otherwise as the
// frames the Sampler picks (see WithSampler), each embedded as an image. It
// is a no-op when the file's bytes are unchanged since the last call;
// hashing the file is far cheaper than decoding it.
func (ix *MediaIndex) IndexVideo(ctx context.Context, docID, path string) error {
	if docID == "" {
		return ErrEmptyID
	}
	if ix.sampler == nil && ix.video == nil {
		return errors.New("search: no sampler: pass WithSampler or WithVideoEmbedder to NewMediaIndex to index videos")
	}
	hash, err := fileHash(path)
	if err != nil {
		return fmt.Errorf("search: hash video %q: %w", docID, err)
	}
	if same, err := ix.unchanged(ctx, docID, hash); err != nil || same {
		return err
	}
	if ix.video != nil {
		segments, err := ix.video.EmbedVideo(ctx, path)
		if err != nil {
			return fmt.Errorf("search: embed video %q: %w", docID, err)
		}
		chunks := make([]vector.Chunk, len(segments))
		for i, s := range segments {
			if err := vector.Validate(s.Vector, ix.embedder.Dimensions()); err != nil {
				return fmt.Errorf("search: embed video %q: segment %d: %w", docID, i, err)
			}
			chunks[i] = vector.Chunk{Seq: int(s.Start / time.Millisecond), Vector: s.Vector}
		}
		if err := ix.store.Upsert(ctx, ix.coord, ix.model, vector.Doc{ID: docID, Hash: hash, Chunks: chunks}); err != nil {
			return fmt.Errorf("search: store %q: %w", docID, err)
		}
		return nil
	}
	frames, err := ix.sampler.Sample(ctx, path)
	if err != nil {
		return fmt.Errorf("search: sample video %q: %w", docID, err)
	}
	return ix.indexFrames(ctx, docID, hash, frames)
}

// IndexFrames embeds frames the caller extracted itself (its own decoder or
// sampler) and stores them under docID. contentHash is the caller's
// fingerprint of the source: when it equals the stored one the call is a
// no-op, and an empty one always indexes. No frames store the hash alone, so
// the item is known but never found.
func (ix *MediaIndex) IndexFrames(ctx context.Context, docID, contentHash string, frames []media.Frame) error {
	if docID == "" {
		return ErrEmptyID
	}
	if contentHash != "" {
		if same, err := ix.unchanged(ctx, docID, contentHash); err != nil || same {
			return err
		}
	}
	return ix.indexFrames(ctx, docID, contentHash, frames)
}

func (ix *MediaIndex) indexFrames(ctx context.Context, docID, hash string, frames []media.Frame) error {
	vectors, err := ix.embed(ctx, frames)
	if err != nil {
		return fmt.Errorf("search: embed %q: %w", docID, err)
	}
	chunks := make([]vector.Chunk, len(frames))
	for i, f := range frames {
		if err := vector.Validate(vectors[i], ix.embedder.Dimensions()); err != nil {
			return fmt.Errorf("search: embed %q: frame %d: %w", docID, i, err)
		}
		chunks[i] = vector.Chunk{Seq: int(f.At / time.Millisecond), Vector: vectors[i]}
	}
	if err := ix.store.Upsert(ctx, ix.coord, ix.model, vector.Doc{ID: docID, Hash: hash, Chunks: chunks}); err != nil {
		return fmt.Errorf("search: store %q: %w", docID, err)
	}
	return nil
}

// unchanged reports whether the store holds hash for docID, so indexing it
// again would change nothing.
func (ix *MediaIndex) unchanged(ctx context.Context, docID, hash string) (bool, error) {
	stored, err := ix.store.Hashes(ctx, ix.coord, docID)
	if err != nil {
		return false, fmt.Errorf("search: read content hash: %w", err)
	}
	h, ok := stored[docID]
	return ok && h == hash, nil
}

// embed runs the frames' images through the embedder and checks it answered
// one vector per frame.
func (ix *MediaIndex) embed(ctx context.Context, frames []media.Frame) ([][]float32, error) {
	if len(frames) == 0 {
		return nil, nil
	}
	images := make([]image.Image, len(frames))
	for i, f := range frames {
		if f.Image == nil {
			return nil, fmt.Errorf("frame %d has no image", i)
		}
		images[i] = f.Image
	}
	vectors, err := ix.embedder.EmbedImages(ctx, images)
	if err != nil {
		return nil, err
	}
	if len(vectors) != len(images) {
		return nil, fmt.Errorf("embedder returned %d vectors for %d images", len(vectors), len(images))
	}
	return vectors, nil
}

// Delete removes an item and its frames. Deleting an unknown id is not an
// error.
func (ix *MediaIndex) Delete(ctx context.Context, docID string) error {
	if docID == "" {
		return ErrEmptyID
	}
	if err := ix.store.Delete(ctx, ix.coord, docID); err != nil {
		return fmt.Errorf("search: delete %q: %w", docID, err)
	}
	return nil
}

// SearchImage decodes the image r holds and returns the items that look most
// like it, best first, at most limit (10 when <= 0).
func (ix *MediaIndex) SearchImage(ctx context.Context, r io.Reader, limit int) ([]MediaMatch, error) {
	img, err := decodeImage(r)
	if err != nil {
		return nil, fmt.Errorf("search: decode query image: %w", err)
	}
	return ix.SearchFrames(ctx, []media.Frame{{Image: img}}, limit)
}

// SearchVideo returns the items that look most like the video at path, best
// first, at most limit (10 when <= 0): by its segments through the
// VideoEmbedder when one is set, otherwise by its sampled frames, each
// embedded as an image. QueryAt is the start of the query segment or the
// time of the query frame that matched.
func (ix *MediaIndex) SearchVideo(ctx context.Context, path string, limit int) ([]MediaMatch, error) {
	if ix.video != nil {
		segments, err := ix.video.EmbedVideo(ctx, path)
		if err != nil {
			return nil, fmt.Errorf("search: embed query video: %w", err)
		}
		queries := make([]queryVector, len(segments))
		for i, s := range segments {
			queries[i] = queryVector{vector: s.Vector, at: s.Start}
		}
		return ix.search(ctx, queries, limit)
	}
	if ix.sampler == nil {
		return nil, errors.New("search: no sampler: pass WithSampler or WithVideoEmbedder to NewMediaIndex to search with videos")
	}
	frames, err := ix.sampler.Sample(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("search: sample query video: %w", err)
	}
	return ix.SearchFrames(ctx, frames, limit)
}

// SearchText returns the items that look most like what the text describes,
// best first, at most limit (10 when <= 0). It needs the text embedder of
// the same model (WithTextEmbedder); the text is embedded as a query.
func (ix *MediaIndex) SearchText(ctx context.Context, text string, limit int) ([]MediaMatch, error) {
	if ix.text == nil {
		return nil, errors.New("search: no text embedder: pass WithTextEmbedder to NewMediaIndex to search by text")
	}
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}
	vectors, err := ix.text.Embed(ctx, langai.RoleQuery, []string{text})
	if err != nil {
		return nil, fmt.Errorf("search: embed query text: %w", err)
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("search: text embedder returned %d vectors for one text", len(vectors))
	}
	return ix.search(ctx, []queryVector{{vector: vectors[0]}}, limit)
}

// SearchIndexed returns the items that look most like the already indexed
// item docID, best first, at most limit (10 when <= 0), searching with its
// stored vectors instead of embedding the file again: no model runs, so it
// costs only the store lookups. For a video every stored frame or segment
// is a query and QueryAt is its time, as with SearchVideo. docID itself is
// never among the hits; an id with nothing stored is ErrNotIndexed.
func (ix *MediaIndex) SearchIndexed(ctx context.Context, docID string, limit int) ([]MediaMatch, error) {
	chunks, err := storedVectors(ctx, ix.store, ix.coord, docID)
	if err != nil {
		return nil, err
	}
	queries := make([]queryVector, len(chunks))
	for i, c := range chunks {
		queries[i] = queryVector{vector: c.Vector, at: time.Duration(c.Seq) * time.Millisecond}
	}
	return ix.search(ctx, queries, limit, docID)
}

// SearchFrames is the search behind SearchImage and SearchVideo for frames
// the caller extracted itself. Every frame is embedded and looked up; each
// item is scored by its best frame against any query frame, and the hits
// come back best first, at most limit (10 when <= 0), those under the floor
// (WithMediaMinScore) left out. An indexed item that is the query itself
// comes first with a score near 1: to leave it out, search with its stored
// vectors through SearchIndexed.
func (ix *MediaIndex) SearchFrames(ctx context.Context, frames []media.Frame, limit int) ([]MediaMatch, error) {
	if len(frames) == 0 {
		return nil, nil
	}
	vectors, err := ix.embed(ctx, frames)
	if err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	queries := make([]queryVector, len(frames))
	for i, f := range frames {
		queries[i] = queryVector{vector: vectors[i], at: f.At}
	}
	return ix.search(ctx, queries, limit)
}

// queryVector is one embedded query frame, segment or text, with the time it
// stands for in a video query.
type queryVector struct {
	vector []float32
	at     time.Duration
}

// search looks every query vector up and keeps the best pair per item, as
// SearchFrames describes, leaving out the ids in exclude.
func (ix *MediaIndex) search(ctx context.Context, queries []queryVector, limit int, exclude ...string) ([]MediaMatch, error) {
	if len(queries) == 0 {
		return nil, nil
	}
	vectors := make([][]float32, len(queries))
	for i, q := range queries {
		vectors[i] = q.vector
	}
	hits, err := rank(ctx, ix.store, ix.coord, vector.Query{Limit: limit, Model: ix.model, Exclude: exclude}, vectors, ix.minScore)
	if err != nil {
		return nil, err
	}
	out := make([]MediaMatch, len(hits))
	for i, h := range hits {
		out[i] = MediaMatch{DocID: h.match.DocID, Score: h.match.Score, At: time.Duration(h.match.Seq) * time.Millisecond, QueryAt: queries[h.query].at}
	}
	return out, nil
}

// decodeImage decodes one picture in any registered format, refusing one
// that claims more than maxImagePixels before any pixel is allocated.
func decodeImage(r io.Reader) (image.Image, error) {
	var head bytes.Buffer
	cfg, _, err := image.DecodeConfig(io.TeeReader(r, &head))
	if err != nil {
		return nil, err
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || cfg.Width*cfg.Height > maxImagePixels {
		return nil, fmt.Errorf("image is %dx%d pixels, more than %d", cfg.Width, cfg.Height, maxImagePixels)
	}
	img, _, err := image.Decode(io.MultiReader(&head, r))
	return img, err
}

// bytesHash is the fingerprint IndexImage stores: the SHA-256 of the encoded
// image, hex.
func bytesHash(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// fileHash is the fingerprint IndexVideo stores: the SHA-256 of the file,
// hex, streamed.
func fileHash(path string) (string, error) {
	f, err := os.Open(path) //nolint:gosec // path is the caller's video file; it is only read and hashed
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

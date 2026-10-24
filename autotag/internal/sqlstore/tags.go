package sqlstore

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math/rand/v2"

	"github.com/go-bumbu/langai/autotag"
)

// newVersion returns the Version a tag starts at when it is created: a random
// one, so that a tag created again under a deleted one's id never reuses a
// Version the deleted one had (see autotag.TagRecord). It is at most 2^62, so
// the bumps of the version column, a 64-bit integer, never overflow it.
func newVersion() int64 {
	return rand.Int64N(1<<62) + 1 //nolint:gosec // G404: a Version has to differ from a deleted tag's, not be unguessable
}

// tagCols are the columns of a tag, in the order scanTag reads them.
const tagCols = `tag_id, name, name_vec, vec, cutoff, model, params, version, built`

// scanTag reads a row of tagCols.
func scanTag(row scanner) (autotag.TagRecord, error) {
	var rec autotag.TagRecord
	var nameVec, vec []byte
	if err := row.Scan(&rec.ID, &rec.Name, &nameVec, &vec, &rec.Cutoff, &rec.Model, &rec.Params, &rec.Version, &rec.Built); err != nil {
		return autotag.TagRecord{}, err
	}
	var err error
	if rec.NameVec, err = decode(nameVec); err != nil {
		return autotag.TagRecord{}, fmt.Errorf("autotag: name vector of tag %q: %w", rec.ID, err)
	}
	if rec.Vec, err = decode(vec); err != nil {
		return autotag.TagRecord{}, fmt.Errorf("autotag: vector of tag %q: %w", rec.ID, err)
	}
	return rec, nil
}

func (s *Store) Tag(ctx context.Context, index, partition, tagID string) (autotag.TagRecord, bool, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return autotag.TagRecord{}, false, err
	}
	rec, err := scanTag(s.db.QueryRowContext(ctx,
		s.q(`SELECT %s FROM %s WHERE partition = ? AND tag_id = ?`, tagCols, t.tags),
		partition, tagID))
	if errors.Is(err, sql.ErrNoRows) {
		return autotag.TagRecord{}, false, nil
	}
	if err != nil {
		return autotag.TagRecord{}, false, err
	}
	return rec, true, nil
}

func (s *Store) Tags(ctx context.Context, index, partition string) ([]autotag.TagRecord, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(ctx,
		s.q(`SELECT %s FROM %s WHERE partition = ? ORDER BY tag_id%s`, tagCols, t.tags, s.d.Collate),
		partition)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []autotag.TagRecord
	for rows.Next() {
		rec, err := scanTag(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

// PutTag needs no transaction: the INSERT creates the tag unless it exists, and
// the UPDATE renames it unless the name is the one it has. Each is one
// statement, so each is atomic by itself. A new tag's Version comes from Go
// (see newVersion), bound like every other value.
func (s *Store) PutTag(ctx context.Context, index, partition, tagID, name string) (bool, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return false, err
	}
	created, err := changed(s.db.ExecContext(ctx,
		s.q(`INSERT INTO %s (partition, tag_id, name, version) VALUES (?, ?, ?, ?)
			ON CONFLICT (partition, tag_id) DO NOTHING`, t.tags),
		partition, tagID, name, newVersion()))
	if created || err != nil {
		return created, err
	}
	// A rename keeps what a build left in the row (Vec, Cutoff, Model, Params and
	// Built) and drops the name vector, which no longer matches the name.
	return changed(s.db.ExecContext(ctx,
		s.q(`UPDATE %s SET name = ?, name_vec = NULL, version = version + 1
			WHERE partition = ? AND tag_id = ? AND name <> ?`, t.tags),
		name, partition, tagID, name))
}

// SetTagVectors is one UPDATE that names the Version it was built from, so it
// stores nothing for a tag that changed or went since.
func (s *Store) SetTagVectors(ctx context.Context, index, partition, tagID string, version int64, v autotag.TagVectors) (bool, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return false, err
	}
	return changed(s.db.ExecContext(ctx,
		s.q(`UPDATE %s SET name_vec = ?, vec = ?, cutoff = ?, model = ?, params = ?, built = version
			WHERE partition = ? AND tag_id = ? AND version = ?`, t.tags),
		encode(v.NameVec), encode(v.Vec), v.Cutoff, v.Model, v.Params,
		partition, tagID, version))
}

func (s *Store) DeleteTag(ctx context.Context, index, partition, tagID string) error {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return err
	}
	return s.inTx(ctx, func(tx *sql.Tx) error {
		// The tag goes first: its row lock keeps a concurrent Link out, and one
		// that already holds it commits before the links are deleted. A tag that
		// is not there is a no-op, links and all (see DeleteItem).
		deleted, err := changed(tx.ExecContext(ctx,
			s.q(`DELETE FROM %s WHERE partition = ? AND tag_id = ?`, t.tags),
			partition, tagID))
		if err != nil || !deleted {
			return err
		}
		return exec(ctx, tx, s.q(`DELETE FROM %s WHERE partition = ? AND tag_id = ?`, t.links), partition, tagID)
	})
}

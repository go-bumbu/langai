package sqlstore

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/go-bumbu/langai/autotag"
)

func (s *Store) Item(ctx context.Context, index, partition, itemID string) (autotag.ItemRecord, bool, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return autotag.ItemRecord{}, false, err
	}
	rec := autotag.ItemRecord{ID: itemID}
	var vec []byte
	err = s.db.QueryRowContext(ctx,
		s.q(`SELECT text, vec, model FROM %s WHERE partition = ? AND item_id = ?`, t.items),
		partition, itemID).Scan(&rec.Text, &vec, &rec.Model)
	if errors.Is(err, sql.ErrNoRows) {
		return autotag.ItemRecord{}, false, nil
	}
	if err != nil {
		return autotag.ItemRecord{}, false, err
	}
	if rec.Vec, err = decode(vec); err != nil {
		return autotag.ItemRecord{}, false, fmt.Errorf("autotag: vector of item %q: %w", itemID, err)
	}
	return rec, true, nil
}

func (s *Store) PutItem(ctx context.Context, index, partition string, item autotag.ItemRecord) ([]string, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return nil, err
	}
	var linked []string
	err = s.inTx(ctx, func(tx *sql.Tx) error {
		if err := exec(ctx, tx,
			s.q(`INSERT INTO %s (partition, item_id, text, vec, model) VALUES (?, ?, ?, ?, ?)
				ON CONFLICT (partition, item_id) DO UPDATE
				SET text = excluded.text, vec = excluded.vec, model = excluded.model`, t.items),
			partition, item.ID, item.Text, encode(item.Vec), item.Model); err != nil {
			return err
		}
		var err error
		linked, err = s.bumpLinked(ctx, tx, t, partition, item.ID)
		return err
	})
	if err != nil {
		return nil, err
	}
	return linked, nil
}

// SetItemVector compares the text inside the UPDATE, so no other write can land
// between the check and the write, and then bumps the tags in the same
// transaction. An item that is gone or has other text matches no row, and the
// transaction ends having changed nothing.
func (s *Store) SetItemVector(ctx context.Context, index, partition, itemID, text string, vec []float32, model string) ([]string, bool, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return nil, false, err
	}
	var linked []string
	var stored bool
	err = s.inTx(ctx, func(tx *sql.Tx) error {
		var err error
		stored, err = changed(tx.ExecContext(ctx,
			s.q(`UPDATE %s SET vec = ?, model = ? WHERE partition = ? AND item_id = ? AND text%s = ?`, t.items, s.d.Collate),
			encode(vec), model, partition, itemID, text))
		if err != nil || !stored {
			return err
		}
		linked, err = s.bumpLinked(ctx, tx, t, partition, itemID)
		return err
	})
	if err != nil || !stored {
		return nil, false, err
	}
	return linked, true, nil
}

func (s *Store) DeleteItem(ctx context.Context, index, partition, itemID string) ([]string, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return nil, err
	}
	var linked []string
	err = s.inTx(ctx, func(tx *sql.Tx) error {
		// The item goes first: its row lock keeps a concurrent Link out, and one
		// that already holds it commits before its links are read here. An item
		// that is not there is a no-op, links and all. It locked nothing, and the
		// statements after it would see rows committed since: a link made since
		// belongs to an item made since, and is not this delete's to remove.
		deleted, err := changed(tx.ExecContext(ctx,
			s.q(`DELETE FROM %s WHERE partition = ? AND item_id = ?`, t.items),
			partition, itemID))
		if err != nil || !deleted {
			return err
		}
		if linked, err = s.bumpLinked(ctx, tx, t, partition, itemID); err != nil {
			return err
		}
		return exec(ctx, tx, s.q(`DELETE FROM %s WHERE partition = ? AND item_id = ?`, t.links), partition, itemID)
	})
	if err != nil {
		return nil, err
	}
	return linked, nil
}

func (s *Store) Link(ctx context.Context, index, partition, itemID, tagID string) (bool, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return false, err
	}
	var linked bool
	err = s.inTx(ctx, func(tx *sql.Tx) error {
		// Both rows are read and locked, the item first, so that neither is
		// deleted before the link lands.
		ok, err := exists(ctx, tx,
			s.q(`SELECT 1 FROM %s WHERE partition = ? AND item_id = ?%s`, t.items, s.d.ForUpdate),
			partition, itemID)
		if err != nil {
			return err
		}
		if !ok {
			return autotag.ErrUnknownItem
		}
		if ok, err = s.lockTag(ctx, tx, t, partition, tagID); err != nil {
			return err
		}
		if !ok {
			return autotag.ErrUnknownTag
		}
		linked, err = changed(tx.ExecContext(ctx,
			s.q(`INSERT INTO %s (partition, item_id, tag_id) VALUES (?, ?, ?) ON CONFLICT DO NOTHING`, t.links),
			partition, itemID, tagID))
		if err != nil || !linked {
			return err
		}
		_, err = s.bump(ctx, tx, t, partition, tagID)
		return err
	})
	if err != nil {
		return false, err
	}
	return linked, nil
}

func (s *Store) Unlink(ctx context.Context, index, partition, itemID, tagID string) (bool, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return false, err
	}
	var unlinked bool
	err = s.inTx(ctx, func(tx *sql.Tx) error {
		// The tag is locked before its link goes, as Link and DeleteTag do, so
		// that none of them waits for a row another holds while it holds one.
		// A tag that is not there has no link.
		if ok, err := s.lockTag(ctx, tx, t, partition, tagID); err != nil || !ok {
			return err
		}
		var err error
		unlinked, err = changed(tx.ExecContext(ctx,
			s.q(`DELETE FROM %s WHERE partition = ? AND item_id = ? AND tag_id = ?`, t.links),
			partition, itemID, tagID))
		if err != nil || !unlinked {
			return err
		}
		_, err = s.bump(ctx, tx, t, partition, tagID)
		return err
	})
	if err != nil {
		return false, err
	}
	return unlinked, nil
}

// Members joins the links of the tag to their items, so a tag the store does
// not hold has none, and no error.
func (s *Store) Members(ctx context.Context, index, partition, tagID string) ([]autotag.Member, error) {
	t, err := s.ensure(ctx, index)
	if err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(ctx,
		s.q(`SELECT i.item_id, i.vec, i.model
			FROM %s l JOIN %s i ON i.partition = l.partition AND i.item_id = l.item_id
			WHERE l.partition = ? AND l.tag_id = ?
			ORDER BY i.item_id%s`, t.links, t.items, s.d.Collate),
		partition, tagID)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []autotag.Member
	for rows.Next() {
		var m autotag.Member
		var vec []byte
		if err := rows.Scan(&m.ItemID, &vec, &m.Model); err != nil {
			return nil, err
		}
		if m.Vec, err = decode(vec); err != nil {
			return nil, fmt.Errorf("autotag: vector of item %q: %w", m.ItemID, err)
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// lockTag reports whether the tag exists, and locks its row, where the Dialect
// has row locks, until the transaction ends.
func (s *Store) lockTag(ctx context.Context, tx *sql.Tx, t tables, partition, tagID string) (bool, error) {
	return exists(ctx, tx,
		s.q(`SELECT 1 FROM %s WHERE partition = ? AND tag_id = ?%s`, t.tags, s.d.ForUpdate),
		partition, tagID)
}

// bump adds one to the Version of the tag, and reports whether the tag was
// there.
func (s *Store) bump(ctx context.Context, tx *sql.Tx, t tables, partition, tagID string) (bool, error) {
	return changed(tx.ExecContext(ctx,
		s.q(`UPDATE %s SET version = version + 1 WHERE partition = ? AND tag_id = ?`, t.tags),
		partition, tagID))
}

// bumpLinked adds one to the Version of every tag linked to the item, and
// returns the ids of the tags it bumped, ordered. It bumps them one at a time,
// in that order, so that writers whose items share tags take the tags' row locks
// in the same order, whatever order the database would visit them in: a single
// UPDATE over all of them could lock them in opposite orders in two writers,
// which would then wait for each other. A tag that is gone by the time it is
// reached is not bumped, and is not returned.
func (s *Store) bumpLinked(ctx context.Context, tx *sql.Tx, t tables, partition, itemID string) ([]string, error) {
	ids, err := queryIDs(ctx, tx,
		s.q(`SELECT tag_id FROM %s WHERE partition = ? AND item_id = ? ORDER BY tag_id%s`, t.links, s.d.Collate),
		partition, itemID)
	if err != nil {
		return nil, err
	}
	var bumped []string
	for _, id := range ids {
		ok, err := s.bump(ctx, tx, t, partition, id)
		if err != nil {
			return nil, err
		}
		if ok {
			bumped = append(bumped, id)
		}
	}
	return bumped, nil
}

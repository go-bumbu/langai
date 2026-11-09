package sqlstore

import (
	"database/sql"
	"errors"
	"strings"
	"testing"
)

func TestQ_NumbersPlaceholders(t *testing.T) {
	for _, tc := range []struct {
		numbered bool
		in, want string
	}{
		{false, "a = ? AND b IN (?)", "a = ? AND b IN (?)"},
		{true, "a = ? AND b IN (?)", "a = $1 AND b IN ($2)"},
		{true, "no placeholders", "no placeholders"},
	} {
		s := &Store{d: Dialect{Numbered: tc.numbered}}
		if got := s.q(tc.in); got != tc.want {
			t.Errorf("q(%q) numbered=%v = %q, want %q", tc.in, tc.numbered, got, tc.want)
		}
	}
}

func TestNew_RejectsBadArguments(t *testing.T) {
	db := &sql.DB{}
	for _, tc := range []struct {
		name       string
		db         *sql.DB
		dims       int
		generation string
	}{
		{"nil handle", nil, 3, ""},
		{"zero dims", db, 0, ""},
		{"bad generation", db, 3, "G2"},
	} {
		if _, err := New(tc.db, Dialect{}, tc.dims, tc.generation); err == nil {
			t.Errorf("%s: New accepted it", tc.name)
		}
	}
}

// TestRetry: a write transaction runs again while it fails with an error
// the Dialect calls retryable, up to maxAttempts runs, and never for any
// other error or without a Retryable.
func TestRetry(t *testing.T) {
	transient, fatal := errors.New("deadlock detected"), errors.New("disk full")
	retryable := func(err error) bool { return errors.Is(err, transient) }
	for _, tc := range []struct {
		name      string
		retryable func(error) bool
		fails     []error // what the runs return, in order; nil after the list
		want      error
		runs      int
	}{
		{"succeeds", retryable, nil, nil, 1},
		{"transient once", retryable, []error{transient}, nil, 2},
		{"transient always", retryable, []error{transient, transient, transient, transient}, transient, maxAttempts},
		{"other error", retryable, []error{fatal}, fatal, 1},
		{"no Retryable", nil, []error{transient}, transient, 1},
	} {
		s := &Store{d: Dialect{Retryable: tc.retryable}}
		runs := 0
		err := s.retry(t.Context(), func() error {
			runs++
			if runs <= len(tc.fails) {
				return tc.fails[runs-1]
			}
			return nil
		})
		if !errors.Is(err, tc.want) || runs != tc.runs {
			t.Errorf("%s: %v after %d runs, want %v after %d", tc.name, err, runs, tc.want, tc.runs)
		}
	}
}

func TestJSONIDs(t *testing.T) {
	got, err := jsonIDs([]string{"a", `quote"d`, "é"})
	if err != nil || !strings.HasPrefix(got, `["a","quote\"d",`) {
		t.Fatalf("jsonIDs = %s, %v", got, err)
	}
}

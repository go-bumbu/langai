package sqlstore

import (
	"database/sql"
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

func TestJSONIDs(t *testing.T) {
	got, err := jsonIDs([]string{"a", `quote"d`, "é"})
	if err != nil || !strings.HasPrefix(got, `["a","quote\"d",`) {
		t.Fatalf("jsonIDs = %s, %v", got, err)
	}
}

package tablename

import (
	"regexp"
	"strings"
	"testing"
)

// TestFor_NameIsStable pins the derivation for a known index outside any
// generation: existing deployments already hold their vectors under these
// names, so changing it would silently orphan every index's tables. The
// token is the first 16 hex digits of sha256("1"), computed outside this code.
func TestFor_NameIsStable(t *testing.T) {
	got, err := For("", "1", "docs")
	if err != nil {
		t.Fatal(err)
	}
	if want := "ss_6b86b273ff34fce1_docs"; got != want {
		t.Fatalf("For(%q, %q, %q) = %q, want %q", "", "1", "docs", got, want)
	}
}

// TestFor_GenerationNamesItsOwnTables: a generation's tables carry its token
// in clear ahead of the index hash, so the same index can hold one table set
// per model while a migration builds the next one, and a retired generation's
// tables can be found by name. The hash is TestFor_NameIsStable's.
func TestFor_GenerationNamesItsOwnTables(t *testing.T) {
	got, err := For("g2", "1", "docs")
	if err != nil {
		t.Fatal(err)
	}
	if want := "ss_g2_6b86b273ff34fce1_docs"; got != want {
		t.Fatalf("For(%q, %q, %q) = %q, want %q", "g2", "1", "docs", got, want)
	}
}

// TestFor_AcceptsShortLowercaseTokens: the documented range, digits and the
// twelve-character maximum included.
func TestFor_AcceptsShortLowercaseTokens(t *testing.T) {
	for _, gen := range []string{"2", "e5small", "abcdefghijkl"} {
		if got, err := For(gen, "1", "docs"); err != nil {
			t.Errorf("For(%q, %q, %q) = %q, %v; want a name", gen, "1", "docs", got, err)
		}
	}
}

// TestFor_RejectsMalformedGeneration: the generation reaches SQL unhashed,
// so it must be a short lowercase bare token. Upper case would fold onto its
// lowercase twin (SQL identifiers are case-insensitive), an underscore would
// put one generation's tables under another's prefix ("a" would match
// "a_b"), and a longer token would push the derived index names towards
// Postgres's 63-byte identifier limit.
func TestFor_RejectsMalformedGeneration(t *testing.T) {
	for _, gen := range []string{"G2", "a_b", "g-2", "abcdefghijklm", `x"; DROP TABLE ss_docs; --`} {
		if got, err := For(gen, "1", "docs"); err == nil {
			t.Errorf("For(%q, %q, %q) = %q, want an error", gen, "1", "docs", got)
		}
	}
}

// TestFor_CaseDistinctIndexesGetDistinctTables: SQL matches identifiers
// case-insensitively, so two indexes differing only in case must not get
// names differing only in case, or one index would read the other's rows.
func TestFor_CaseDistinctIndexesGetDistinctTables(t *testing.T) {
	upper, err := For("", "A", "chunks")
	if err != nil {
		t.Fatal(err)
	}
	lower, err := For("", "a", "chunks")
	if err != nil {
		t.Fatal(err)
	}
	if strings.EqualFold(upper, lower) {
		t.Fatalf("indexes %q and %q resolve to the same table: %q, %q", "A", "a", upper, lower)
	}
}

// TestFor_NameIsABareIdentifier: the name is interpolated into SQL, so
// whatever the index holds must come out as a bare identifier.
func TestFor_NameIsABareIdentifier(t *testing.T) {
	bare := regexp.MustCompile(`^[a-z0-9_]+$`)
	for _, index := range []string{
		"3f2c9a1e-7b4d-4c2a-9e1f-0a5b6c7d8e9f", // a hyphenated UUID
		`x"; DROP TABLE ss_docs; --`,
	} {
		got, err := For("g2", index, "vec")
		if err != nil {
			t.Fatal(err)
		}
		if !bare.MatchString(got) {
			t.Errorf("For(%q, %q, %q) = %q, not a bare identifier", "g2", index, "vec", got)
		}
	}
}

// TestFor_EmptyIndexIsAnError: an empty index would hand every caller that
// forgot to set one the same shared tables.
func TestFor_EmptyIndexIsAnError(t *testing.T) {
	if got, err := For("", "", "docs"); err == nil {
		t.Fatalf("For(%q, %q, %q) = %q, want an error", "", "", "docs", got)
	}
}

package langai

import "testing"

func TestRoleString(t *testing.T) {
	if RoleDocument.String() != "document" || RoleQuery.String() != "query" || Role(7).String() != "Role(7)" {
		t.Fatalf("Role strings: %s %s %s", RoleDocument, RoleQuery, Role(7))
	}
}

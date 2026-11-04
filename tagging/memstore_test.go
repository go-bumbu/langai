package tagging_test

import (
	"testing"

	"github.com/go-bumbu/langai/tagging"
	"github.com/go-bumbu/langai/tagging/storetest"
)

func TestMemStore(t *testing.T) {
	storetest.Run(t, func(*testing.T) tagging.Store { return tagging.NewMemStore() })
}

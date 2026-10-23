package autotag_test

import (
	"testing"

	"github.com/go-bumbu/langai/autotag"
	"github.com/go-bumbu/langai/autotag/storetest"
)

func TestMemStore(t *testing.T) {
	storetest.Run(t, func(*testing.T) autotag.Store { return autotag.NewMemStore() })
}

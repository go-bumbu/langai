package mediasearch

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/go-bumbu/sense/search"
)

func TestReset_EmptiesTheIndexAndTheNextScanEmbedsAgain(t *testing.T) {
	ctx := context.Background()
	f := scanned(t)
	f.emb.calls.Store(0)
	if err := f.lib.Reset(ctx); err != nil {
		t.Fatal(err)
	}
	if got := f.lib.LastScan(); !got.At.IsZero() {
		t.Errorf("last scan after a reset: %+v", got)
	}
	// the catalog keeps its items, which have nothing to be found by
	if _, ok := f.lib.Item("red.png"); !ok {
		t.Error("the reset dropped the catalog")
	}
	if _, err := f.lib.Similar(ctx, "red.png", 0, false); !errors.Is(err, search.ErrNotIndexed) {
		t.Errorf("similar after a reset: %v", err)
	}
	// the hashes are forgotten on disk too: a restart does not take the
	// files for unchanged
	if data, err := os.ReadFile(f.lib.cfg.CatalogFile); err != nil || strings.TrimSpace(string(data)) != "{}" {
		t.Errorf("catalog file after a reset: %q, %v", data, err)
	}

	stats, err := f.lib.Scan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Indexed != 3 || stats.Skipped != 0 {
		t.Errorf("scan after a reset: %+v", stats)
	}
	if got := f.emb.calls.Load(); got != 4 { // two images and two frames, embedded again
		t.Errorf("the scan after a reset embedded %d pictures, want 4", got)
	}

	f.lib.scanning.Store(true)
	if err := f.lib.Reset(ctx); !errors.Is(err, ErrScanning) {
		t.Errorf("a reset during a scan: %v", err)
	}
}

package fetch

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

// DirInstall describes the install of a release tarball's libraries into a
// directory. What differs between providers is here: which archive, how its
// layout is unpacked, and what the record of the install says.
type DirInstall struct {
	// Dir is the directory to fill; whatever it held is replaced.
	Dir string
	// What names the install in the download's log line and in errors
	// ("llama.cpp b11429").
	What string
	// Archive is the gzipped release tarball.
	Archive Asset
	// Extract unpacks the verified archive into dir, an empty directory.
	Extract func(archive, dir string) error
	// RecordFile is the name of the file written next to the libraries with
	// Record as its content, so a later run can tell what Dir holds.
	RecordFile string
	Record     []byte
}

// InstallDir downloads in.Archive, unpacks it and moves the result into
// in.Dir. Download and extraction happen next to the final directory, which
// is renamed into place at the end: an interrupted install never leaves a
// half-filled in.Dir.
func (d Downloader) InstallDir(ctx context.Context, in DirInstall) error {
	parent := filepath.Dir(in.Dir)
	if err := os.MkdirAll(parent, 0o750); err != nil {
		return err
	}
	staging, err := os.MkdirTemp(parent, ".partial-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(staging) }()
	archive := filepath.Join(staging, "libs.tar.gz")
	if err := d.File(ctx, in.What, in.Archive, archive); err != nil {
		return fmt.Errorf("download %s: %w", in.What, err)
	}
	libs := filepath.Join(staging, "libs")
	if err := os.Mkdir(libs, 0o750); err != nil {
		return err
	}
	if err := in.Extract(archive, libs); err != nil {
		return fmt.Errorf("extract %s: %w", in.What, err)
	}
	if err := os.WriteFile(filepath.Join(libs, in.RecordFile), in.Record, 0o600); err != nil {
		return err
	}
	if err := os.RemoveAll(in.Dir); err != nil {
		return err
	}
	return os.Rename(libs, in.Dir)
}

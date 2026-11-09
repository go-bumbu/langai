// Package assets holds what the example service's pages share: the base
// stylesheet that makes them read as one application, and a script with the
// navigation between them and the small DOM helpers the newer pages are
// written with. Handler serves both; every page links them as
// "../assets/base.css" and "../assets/ui.js", so the service mounts it at
// "/assets/".
package assets

import (
	"embed"
	"io/fs"
	"net/http"
)

//go:embed ui
var uiFiles embed.FS

// Handler serves the shared stylesheet and script at a mount point.
func Handler() http.Handler {
	ui, err := fs.Sub(uiFiles, "ui")
	if err != nil {
		panic(err) // the embedded tree is fixed at build time
	}
	return http.FileServerFS(ui)
}

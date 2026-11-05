package mediasearch

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
	"strconv"
)

//go:embed ui
var uiFiles embed.FS

// maxUpload caps an uploaded query file.
const maxUpload = 256 << 20

// Handler serves the Library at a mount point: the page at "/", the API
// under "/api/" and the pictures under "/thumb/", "/storyboard/" and
// "/file/". Ids are catalog keys, so only files the scan found are served.
func Handler(l *Library) http.Handler {
	ui, err := fs.Sub(uiFiles, "ui")
	if err != nil {
		panic(err) // the embedded tree is fixed at build time
	}
	h := &handler{lib: l}
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServerFS(ui))
	mux.HandleFunc("GET /api/info", h.info)
	mux.HandleFunc("GET /api/items", h.items)
	mux.HandleFunc("GET /api/similar/{id...}", h.similar)
	mux.HandleFunc("POST /api/search", h.search)
	mux.HandleFunc("GET /api/text", h.text)
	mux.HandleFunc("POST /api/rescan", h.rescan)
	mux.HandleFunc("GET /api/frames/{id...}", h.frames)
	mux.HandleFunc("GET /thumb/{id...}", h.thumb)
	mux.HandleFunc("GET /storyboard/{id...}", h.storyboard)
	mux.HandleFunc("GET /file/{id...}", h.file)
	return mux
}

type handler struct {
	lib *Library
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeError answers an error with the status its kind calls for.
func writeError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	switch {
	case errors.Is(err, ErrNotFound):
		status = http.StatusNotFound
	case errors.Is(err, ErrNotVideo), errors.Is(err, ErrUnsupported):
		status = http.StatusBadRequest
	case errors.Is(err, ErrScanning):
		status = http.StatusConflict
	}
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func (h *handler) info(w http.ResponseWriter, _ *http.Request) {
	images, videos := 0, 0
	for _, it := range h.lib.Items() {
		if it.Kind == KindVideo {
			videos++
		} else {
			images++
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"info":     h.lib.Info(),
		"limit":    h.lib.Limit(),
		"dir":      h.lib.Dir(),
		"images":   images,
		"videos":   videos,
		"lastScan": h.lib.LastScan(),
	})
}

func (h *handler) items(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, h.lib.Items())
}

func limitOf(r *http.Request) int {
	n, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	return n
}

func (h *handler) similar(w http.ResponseWriter, r *http.Request) {
	res, err := h.lib.Similar(r.Context(), r.PathValue("id"), limitOf(r))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (h *handler) search(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxUpload)
	file, header, err := r.FormFile("file")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "multipart field \"file\" missing or too large: " + err.Error()})
		return
	}
	defer func() { _ = file.Close() }()
	res, err := h.lib.Search(r.Context(), header.Filename, file, limitOf(r))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (h *handler) text(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query().Get("q")
	if q == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query parameter \"q\" missing"})
		return
	}
	res, err := h.lib.SearchText(r.Context(), q, limitOf(r))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (h *handler) rescan(w http.ResponseWriter, r *http.Request) {
	// the scan outlives a closed tab
	stats, err := h.lib.Scan(context.WithoutCancel(r.Context()))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (h *handler) frames(w http.ResponseWriter, r *http.Request) {
	b, err := h.lib.Frames(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, b)
}

func (h *handler) thumb(w http.ResponseWriter, r *http.Request) {
	path, err := h.lib.Thumb(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, err)
		return
	}
	http.ServeFile(w, r, path) //nolint:gosec // path is a cache file the Library named after a catalog key, never the request's path
}

func (h *handler) storyboard(w http.ResponseWriter, r *http.Request) {
	path, err := h.lib.Storyboard(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, err)
		return
	}
	http.ServeFile(w, r, path) //nolint:gosec // see thumb
}

func (h *handler) file(w http.ResponseWriter, r *http.Request) {
	path, ok := h.lib.Path(r.PathValue("id"))
	if !ok {
		writeError(w, ErrNotFound)
		return
	}
	http.ServeFile(w, r, path)
}

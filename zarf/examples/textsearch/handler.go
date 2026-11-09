package textsearch

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
	"strconv"

	"github.com/go-bumbu/langai/search"
	"github.com/go-bumbu/langai/vector"
)

//go:embed ui
var uiFiles embed.FS

// Handler serves the Library at a mount point: the page at "/" and the API
// under "/api/". Ids are catalog keys, so only files the scan found are
// read.
func Handler(l *Library) http.Handler {
	ui, err := fs.Sub(uiFiles, "ui")
	if err != nil {
		panic(err) // the embedded tree is fixed at build time
	}
	h := &handler{lib: l}
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServerFS(ui))
	mux.HandleFunc("GET /api/info", h.info)
	mux.HandleFunc("GET /api/docs", h.docs)
	mux.HandleFunc("GET /api/doc/{id...}", h.doc)
	mux.HandleFunc("GET /api/search", h.search)
	mux.HandleFunc("GET /api/similar/{id...}", h.similar)
	mux.HandleFunc("POST /api/rescan", h.rescan)
	mux.HandleFunc("POST /api/reset", h.reset)
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

// writeError answers an error with the status its kind calls for. A model
// mismatch is a conflict the page resolves with a reset, so it is marked.
func writeError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	body := map[string]any{"error": err.Error()}
	switch {
	case errors.Is(err, ErrNotFound), errors.Is(err, search.ErrNotIndexed):
		status = http.StatusNotFound
	case errors.Is(err, ErrScanning):
		status = http.StatusConflict
	case errors.Is(err, vector.ErrModelMismatch):
		status = http.StatusConflict
		body["mismatch"] = true
	}
	writeJSON(w, status, body)
}

func (h *handler) info(w http.ResponseWriter, r *http.Request) {
	docs := h.lib.Docs()
	chunks := 0
	for _, d := range docs {
		chunks += d.Chunks
	}
	info := h.lib.Info()
	last := h.lib.LastScan()
	body := map[string]any{
		"info":       info,
		"limit":      h.lib.Limit(),
		"dir":        h.lib.Dir(),
		"docs":       len(docs),
		"chunks":     chunks,
		"lastScan":   last,
		"scanning":   h.lib.Scanning(),
		"modelState": h.lib.ModelState(),
		"examples":   h.lib.Examples(),
	}
	// the model that built the collection, when it is not the one the page
	// runs: what the reset is for
	if last.Mismatch {
		body["storedModel"] = h.lib.StoredModel(r.Context())
	}
	writeJSON(w, http.StatusOK, body)
}

func (h *handler) docs(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, h.lib.Docs())
}

func (h *handler) doc(w http.ResponseWriter, r *http.Request) {
	v, err := h.lib.View(r.PathValue("id"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, v)
}

func limitOf(r *http.Request) int {
	n, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	return n
}

func (h *handler) search(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query().Get("q")
	if q == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query parameter \"q\" missing"})
		return
	}
	res, err := h.lib.Search(r.Context(), q, limitOf(r))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (h *handler) similar(w http.ResponseWriter, r *http.Request) {
	res, err := h.lib.Similar(r.Context(), r.PathValue("id"), limitOf(r))
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

func (h *handler) reset(w http.ResponseWriter, r *http.Request) {
	if err := h.lib.Reset(context.WithoutCancel(r.Context())); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

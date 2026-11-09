package status

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
)

//go:embed ui
var uiFiles embed.FS

// Handler serves the Service at a mount point: the page at "/" and the API
// under "/api/". It is the service's home page, so the other pages' mount
// points must be registered next to it with longer patterns.
func Handler(s *Service) http.Handler {
	ui, err := fs.Sub(uiFiles, "ui")
	if err != nil {
		panic(err) // the embedded tree is fixed at build time
	}
	h := &handler{svc: s}
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServerFS(ui))
	mux.HandleFunc("GET /api/status", h.status)
	mux.HandleFunc("POST /api/models/{key}/load", h.load)
	mux.HandleFunc("POST /api/collections/{key}/reset", h.reset)
	mux.HandleFunc("POST /api/cache/prune", h.prune)
	return mux
}

type handler struct {
	svc *Service
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
	case errors.Is(err, ErrNotLazy), errors.Is(err, ErrNoAction):
		status = http.StatusBadRequest
	}
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func (h *handler) status(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, h.svc.Snapshot(r.Context()))
}

// load answers with the model's status after the load: a failed load is a
// state the page shows, not a failed request.
func (h *handler) load(w http.ResponseWriter, r *http.Request) {
	st, err := h.svc.Load(r.Context(), r.PathValue("key"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, st)
}

func (h *handler) reset(w http.ResponseWriter, r *http.Request) {
	// a reset is not left half done by a closed tab
	st, err := h.svc.Reset(context.WithoutCancel(r.Context()), r.PathValue("key"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, st)
}

func (h *handler) prune(w http.ResponseWriter, r *http.Request) {
	st, err := h.svc.Prune(context.WithoutCancel(r.Context()))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, st)
}

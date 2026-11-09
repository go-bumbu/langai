package duplicates

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
	"strconv"

	"github.com/go-bumbu/langai/search"
)

//go:embed ui
var uiFiles embed.FS

// maxUpload caps an uploaded query picture.
const maxUpload = 128 << 20

// Handler serves the Finder at a mount point: the page at "/", the API
// under "/api/" and the pictures under "/file/". Ids are catalog keys, so
// only files the scan found are served.
func Handler(f *Finder) http.Handler {
	ui, err := fs.Sub(uiFiles, "ui")
	if err != nil {
		panic(err) // the embedded tree is fixed at build time
	}
	h := &handler{finder: f}
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServerFS(ui))
	mux.HandleFunc("GET /api/info", h.info)
	mux.HandleFunc("GET /api/groups", h.groups)
	mux.HandleFunc("POST /api/search", h.search)
	mux.HandleFunc("GET /api/similar/{id...}", h.similar)
	mux.HandleFunc("POST /api/rescan", h.rescan)
	mux.HandleFunc("GET /file/{id...}", h.file)
	return mux
}

type handler struct {
	finder *Finder
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
	case errors.Is(err, ErrNotFound), errors.Is(err, search.ErrNotIndexed):
		status = http.StatusNotFound
	case errors.Is(err, ErrBadInput):
		status = http.StatusBadRequest
	case errors.Is(err, search.ErrTooLarge):
		status = http.StatusRequestEntityTooLarge
	case errors.Is(err, ErrScanning):
		status = http.StatusConflict
	}
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func (h *handler) info(w http.ResponseWriter, _ *http.Request) {
	pictures, failed := 0, map[string]string{}
	for _, p := range h.finder.Pictures() {
		pictures++
		if p.Error != "" {
			failed[p.ID] = p.Error
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"dir":      h.finder.Dir(),
		"kinds":    h.finder.Kinds(),
		"default":  DefaultKind,
		"limit":    h.finder.Limit(),
		"pictures": pictures,
		"failed":   failed,
		"lastScan": h.finder.LastScan(),
		"scanning": h.finder.Scanning(),
	})
}

// intOf reads a whole number from the query; missing is -1, which the
// Finder reads as "the default".
func intOf(r *http.Request, name string) (int, error) {
	s := r.URL.Query().Get(name)
	if s == "" {
		return -1, nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return 0, errors.Join(ErrBadInput, errors.New("query parameter \""+name+"\" is not a whole number from 0 up"))
	}
	return n, nil
}

func (h *handler) groups(w http.ResponseWriter, r *http.Request) {
	threshold, err := intOf(r, "threshold")
	if err != nil {
		writeError(w, err)
		return
	}
	res, err := h.finder.Groups(r.URL.Query().Get("kind"), threshold)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (h *handler) search(w http.ResponseWriter, r *http.Request) {
	threshold, err := intOf(r, "threshold")
	if err != nil {
		writeError(w, err)
		return
	}
	limit, err := intOf(r, "limit")
	if err != nil {
		writeError(w, err)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxUpload)
	file, _, err := r.FormFile("file")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "multipart field \"file\" missing or too large: " + err.Error()})
		return
	}
	defer func() { _ = file.Close() }()
	res, err := h.finder.Search(r.Context(), r.URL.Query().Get("kind"), file, threshold, limit)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (h *handler) similar(w http.ResponseWriter, r *http.Request) {
	threshold, err := intOf(r, "threshold")
	if err != nil {
		writeError(w, err)
		return
	}
	limit, err := intOf(r, "limit")
	if err != nil {
		writeError(w, err)
		return
	}
	res, err := h.finder.Similar(r.Context(), r.URL.Query().Get("kind"), r.PathValue("id"), threshold, limit)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (h *handler) rescan(w http.ResponseWriter, r *http.Request) {
	// the scan outlives a closed tab
	stats, err := h.finder.Scan(context.WithoutCancel(r.Context()))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (h *handler) file(w http.ResponseWriter, r *http.Request) {
	path, ok := h.finder.Path(r.PathValue("id"))
	if !ok {
		writeError(w, ErrNotFound)
		return
	}
	http.ServeFile(w, r, path)
}

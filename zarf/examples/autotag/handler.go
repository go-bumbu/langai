package autotag

import (
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
	"strconv"
)

//go:embed ui
var uiFiles embed.FS

// maxBody caps a JSON request body.
const maxBody = 1 << 20

// Handler serves the Notebook at a mount point: the page at "/" and the API
// under "/api/".
func Handler(nb *Notebook) http.Handler {
	ui, err := fs.Sub(uiFiles, "ui")
	if err != nil {
		panic(err) // the embedded tree is fixed at build time
	}
	h := &handler{nb: nb}
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServerFS(ui))
	mux.HandleFunc("GET /api/info", h.info)
	mux.HandleFunc("GET /api/notebook", h.view)
	mux.HandleFunc("POST /api/tags", h.addTag)
	mux.HandleFunc("PUT /api/tags/{id}", h.renameTag)
	mux.HandleFunc("DELETE /api/tags/{id}", h.deleteTag)
	mux.HandleFunc("POST /api/notes", h.addNote)
	mux.HandleFunc("DELETE /api/notes/{id}", h.deleteNote)
	mux.HandleFunc("PUT /api/notes/{id}/tags/{tag}", h.tagNote)
	mux.HandleFunc("DELETE /api/notes/{id}/tags/{tag}", h.untagNote)
	mux.HandleFunc("GET /api/notes/{id}/recommend", h.recommend)
	mux.HandleFunc("POST /api/samples", h.samples)
	return mux
}

type handler struct {
	nb *Notebook
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
	case errors.Is(err, ErrBadInput), errors.Is(err, ErrNoSamples):
		status = http.StatusBadRequest
	}
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

// readJSON decodes a small JSON body into v, answering 400 itself on failure.
func readJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, maxBody)
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "bad JSON body: " + err.Error()})
		return false
	}
	return true
}

func (h *handler) info(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"info": h.nb.Info(), "chat": h.nb.ChatState(), "embedder": h.nb.EmbedderState(), "runsOn": h.nb.RunsOn(),
	})
}

func (h *handler) view(w http.ResponseWriter, r *http.Request) {
	v, err := h.nb.View(r.Context())
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, v)
}

type nameBody struct {
	Name string `json:"name"`
}

func (h *handler) addTag(w http.ResponseWriter, r *http.Request) {
	var b struct {
		Name    string `json:"name"`
		Parents bool   `json:"parents"` // create the path's missing ancestors too
	}
	if !readJSON(w, r, &b) {
		return
	}
	t, err := h.nb.AddTag(r.Context(), b.Name, b.Parents)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, t)
}

func (h *handler) renameTag(w http.ResponseWriter, r *http.Request) {
	var b nameBody
	if !readJSON(w, r, &b) {
		return
	}
	t, err := h.nb.RenameTag(r.Context(), r.PathValue("id"), b.Name)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, t)
}

func (h *handler) deleteTag(w http.ResponseWriter, r *http.Request) {
	if err := h.nb.DeleteTag(r.Context(), r.PathValue("id")); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *handler) addNote(w http.ResponseWriter, r *http.Request) {
	var b struct {
		Title string `json:"title"`
		Text  string `json:"text"`
	}
	if !readJSON(w, r, &b) {
		return
	}
	n, err := h.nb.AddNote(r.Context(), b.Title, b.Text)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, n)
}

func (h *handler) deleteNote(w http.ResponseWriter, r *http.Request) {
	if err := h.nb.DeleteNote(r.Context(), r.PathValue("id")); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *handler) tagNote(w http.ResponseWriter, r *http.Request) {
	if err := h.nb.TagNote(r.Context(), r.PathValue("id"), r.PathValue("tag")); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *handler) untagNote(w http.ResponseWriter, r *http.Request) {
	if err := h.nb.UntagNote(r.Context(), r.PathValue("id"), r.PathValue("tag")); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// recommend takes ?lang=<BCP 47 tag> (empty: the note's own language) and
// ?tree=1 (paths extending the notebook's tags).
func (h *handler) recommend(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	opts := RecommendOptions{Language: q.Get("lang"), Tree: q.Get("tree") == "1"}
	res, err := h.nb.Recommend(r.Context(), r.PathValue("id"), opts)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

// samples imports the samples; ?hand=N puts each tag by hand on N of them.
func (h *handler) samples(w http.ResponseWriter, r *http.Request) {
	perTag := 0
	if s := r.URL.Query().Get("hand"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query parameter \"hand\" is not a number"})
			return
		}
		perTag = n
	}
	st, err := h.nb.LoadSamples(r.Context(), perTag)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, st)
}

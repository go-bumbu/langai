# The example service: `make prepare` fetches sample media and writes sample
# documents, `make run` indexes them and serves the pages. Run from this
# directory.
MEDIA_DIR  ?= data/media
# the .md and .txt documents of the text search page
DOCS_DIR   ?= data/docs
DATA_DIR   ?= data
# the ONNX models and runtime, shared with `make test-images` at the root so
# the ~380 MB download happens once
MODELS_DIR ?= $(HOME)/.cache/semsearch-onnx
# the model: google/siglip2-base-patch16-224 or facebook/dinov2-base (ONNX, videos as
# frames), or qwen3vl (Qwen3-VL-Embedding-2B through llama.cpp: text, images and
# videos as segments in one space; GGUF files and the llama.cpp build in LLAMA_DIR)
MODEL      ?= google/siglip2-base-patch16-224
LLAMA_DIR  ?= $(HOME)/.cache/semsearch-yzma
PROVIDER   ?= cpu
# what runs the GGUF model: auto (empty: a GPU when the machine has one), cpu, vulkan, cuda, metal
ACCEL      ?=
SEGMENT    ?= 20s
ADDR       ?= 127.0.0.1:8085
MIN_SCORE  ?= 0
LIMIT      ?= 12
# the cutoff of the auto tagging page's recommended tags
TAG_MIN_SCORE ?= 0.45
# the floor of the text search page's hits; 0 shows everything
TEXT_MIN_SCORE ?= 0

default: help

prepare: ## download sample photos and render sample clips into data/media, write sample documents into data/docs (needs curl and ffmpeg)
	@./prepare.sh "$(MEDIA_DIR)" "$(DOCS_DIR)"

run: ## index data/media and data/docs and serve the pages on http://127.0.0.1:8085 (first run downloads the models)
	@go run . -addr "$(ADDR)" -media "$(MEDIA_DIR)" -docs "$(DOCS_DIR)" -data "$(DATA_DIR)" -models "$(MODELS_DIR)" \
		-llama "$(LLAMA_DIR)" -model "$(MODEL)" -provider "$(PROVIDER)" -accel "$(ACCEL)" -segment "$(SEGMENT)" -min-score "$(MIN_SCORE)" -limit "$(LIMIT)" \
		-tag-min-score "$(TAG_MIN_SCORE)" -text-min-score "$(TEXT_MIN_SCORE)"

clean: ## remove the index and tag databases, the catalogs and the thumbnail cache; the media, the documents, the notebook and the models stay
	@rm -rf "$(DATA_DIR)"/index-* "$(DATA_DIR)"/autotag-* "$(DATA_DIR)"/thumbs

.PHONY: default prepare run clean help
help: # Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nSettings (environment or make arguments):\n  MEDIA_DIR=%s DOCS_DIR=%s DATA_DIR=%s MODELS_DIR=%s\n  MODEL=%s PROVIDER=%s ACCEL=%s SEGMENT=%s LLAMA_DIR=%s\n  ADDR=%s MIN_SCORE=%s LIMIT=%s TAG_MIN_SCORE=%s TEXT_MIN_SCORE=%s\n\n" \
		"$(MEDIA_DIR)" "$(DOCS_DIR)" "$(DATA_DIR)" "$(MODELS_DIR)" "$(MODEL)" "$(PROVIDER)" "$(ACCEL)" "$(SEGMENT)" "$(LLAMA_DIR)" "$(ADDR)" "$(MIN_SCORE)" "$(LIMIT)" "$(TAG_MIN_SCORE)" "$(TEXT_MIN_SCORE)"

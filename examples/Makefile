# The example service: `make prepare` fetches sample media, `make run` indexes
# it and serves the pages. Run from this directory.
MEDIA_DIR  ?= data/media
DATA_DIR   ?= data
# the ONNX models and runtime, shared with `make test-images` at the root so
# the ~380 MB download happens once
MODELS_DIR ?= $(HOME)/.cache/semsearch-onnx
MODEL      ?= google/siglip2-base-patch16-224
PROVIDER   ?= cpu
ADDR       ?= 127.0.0.1:8085
MIN_SCORE  ?= 0
LIMIT      ?= 12

default: help

prepare: ## download sample photos and render sample clips into data/media (needs curl and ffmpeg)
	@./prepare.sh "$(MEDIA_DIR)"

run: ## index data/media and serve the pages on http://127.0.0.1:8085 (first run downloads the model)
	@go run . -addr "$(ADDR)" -media "$(MEDIA_DIR)" -data "$(DATA_DIR)" -models "$(MODELS_DIR)" \
		-model "$(MODEL)" -provider "$(PROVIDER)" -min-score "$(MIN_SCORE)" -limit "$(LIMIT)"

clean: ## remove the index databases, the catalogs and the thumbnail cache; the media and the models stay
	@rm -rf "$(DATA_DIR)"/index-* "$(DATA_DIR)"/thumbs

.PHONY: default prepare run clean help
help: # Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nSettings (environment or make arguments):\n  MEDIA_DIR=%s DATA_DIR=%s MODELS_DIR=%s\n  MODEL=%s PROVIDER=%s ADDR=%s MIN_SCORE=%s LIMIT=%s\n\n" \
		"$(MEDIA_DIR)" "$(DATA_DIR)" "$(MODELS_DIR)" "$(MODEL)" "$(PROVIDER)" "$(ADDR)" "$(MIN_SCORE)" "$(LIMIT)"

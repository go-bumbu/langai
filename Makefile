COMMIT_SHA_SHORT ?= $(shell git rev-parse --short=12 HEAD)
PWD_DIR := ${CURDIR}

default: help

#==========================================================================================
##@ Testing
#==========================================================================================
test: ## run fast go tests
	@go test ./... -cover

# The real-model tests download the pinned llama.cpp build (~18 MB) and the default
# model (~329 MB) into SEMSEARCH_YZMA_TEST_DIR on first run, and the host needs libffi8,
# libgomp1 and libstdc++6. -p 1 keeps the two packages from installing llama.cpp into the
# cache at once.
SEMSEARCH_YZMA_TEST_DIR ?= $(HOME)/.cache/semsearch-yzma
test-models: ## run the real-model tests: the embedding smoke test and the golden-set quality floors
	@SEMSEARCH_MODELS=1 SEMSEARCH_YZMA_TEST_DIR=$(SEMSEARCH_YZMA_TEST_DIR) go test -p 1 ./semsearch/yzma/ ./autotag/ -run 'TestEmbedAndRank|TestGoldenSet|TestGoldenTagging' -v -count=1

lint: ## run go linter
	# depends on https://github.com/golangci/golangci-lint
	@golangci-lint run

benchmark: ## run go benchmarks
	@go test -run=^$$ -bench=. ./...

license-check: ## check for invalid licenses
	# depends on : https://github.com/elastic/go-licence-detector
	@go list -m -mod=readonly  -json all  | go-licence-detector -includeIndirect -validate -rules allowedLicenses.json -overrides overrideLicenses.json

.PHONY: verify
verify: ## run all checks; runs every check and fails if any fail
	@fail=0; \
	for target in test license-check lint benchmark coverage; do \
		echo "==================== make $$target ===================="; \
		$(MAKE) --no-print-directory $$target || fail=1; \
	done; \
	if [ $$fail -ne 0 ]; then \
		echo "❌ verify failed (see above)"; \
		exit 1; \
	fi; \
	echo "✅ verify passed"

# Default coverage threshold is 70: semsearch/yzma's calls into llama.cpp only run in the
# real-model tests (make test-models), which keeps that package's unit-test coverage short of 80.
COVERAGE_THRESHOLD ?= 70

# TESTDBS_ALL makes go-bumbu/testdbs start its container-backed databases, so the pgvec
# Postgres tests run and count here instead of skipping (needs Docker).
.PHONY: coverage
coverage: ## check code coverage per package
	@out=$$(TESTDBS_ALL=1 go test -cover -covermode=atomic ./...) || { echo "$$out"; exit 1; }; \
	echo "$$out" | awk -v threshold=$(COVERAGE_THRESHOLD) ' \
		/\[no test files\]/ { printf "⚠️  %-70s no test files\n", $$2; next } \
		/coverage:/ { \
			for (i = 1; i <= NF; i++) if ($$i == "coverage:") { cov = $$(i+1); sub(/%/, "", cov); break }; \
			if (cov + 0 < threshold) { printf "❌ %-70s %s%% (below %s%%)\n", $$2, cov, threshold; fail = 1 } \
			else { printf "✅ %-70s %s%%\n", $$2, cov } \
		} \
		END { exit fail }'

cover-report: ## generate a coverage report
	go test -covermode=count -coverpkg=./... -coverprofile cover.out  ./...
	go tool cover -html cover.out -o cover.html
	open cover.html

#==========================================================================================
##@ Evaluation
#==========================================================================================
# semeval runs TestCompareModels (semsearch/eval/compare_test.go) over every model in
# semsearch/eval/testdata/models.yaml, sharing SEMSEARCH_YZMA_TEST_DIR with the real-model
# tests. Optional settings, with paths relative to the repository root:
#   SEMEVAL_ONLY=org/a,org/b      run only these models
#   SEMEVAL_DATASET=export.yaml   another dataset in the eval format (default: the golden set)
#   SEMEVAL_MODELS=models.yaml    another models file
#   SEMEVAL_JSON=results.json     also write every metric to this file
#   SEMEVAL_THREADS=4             threads per model (default: yzma's, at most four)
#   SEMEVAL_FLOOR=0.5             tagging cutoff to report at (default: 0.46, autotag's NameMinScore)
# -timeout 0: downloading and running every candidate can outlast go test's default 10 minutes.
semeval: ## compare embedding models on the golden set (downloads candidates into ~/.cache/semsearch-yzma; needs libffi8 libgomp1 libstdc++6)
	@SEMEVAL=1 SEMSEARCH_YZMA_TEST_DIR='$(abspath $(SEMSEARCH_YZMA_TEST_DIR))' \
		SEMEVAL_ONLY='$(SEMEVAL_ONLY)' SEMEVAL_THREADS='$(SEMEVAL_THREADS)' SEMEVAL_FLOOR='$(SEMEVAL_FLOOR)' \
		SEMEVAL_DATASET='$(abspath $(SEMEVAL_DATASET))' SEMEVAL_MODELS='$(abspath $(SEMEVAL_MODELS))' \
		SEMEVAL_JSON='$(abspath $(SEMEVAL_JSON))' \
		go test ./semsearch/eval/ -run '^TestCompareModels$$' -v -count=1 -timeout 0

#==========================================================================================
##@ Release
#==========================================================================================

.PHONY: check-git-clean
check-git-clean: # check if git repo is clen
	@git diff --quiet

.PHONY: check-branch
check-branch:
	@current_branch=$$(git symbolic-ref --short HEAD) && \
	if [ "$$current_branch" != "main" ]; then \
		echo "Error: You are on branch '$$current_branch'. Please switch to 'main'."; \
		exit 1; \
	fi

check_env: # check for needed envs
	@[ "${version}" ] || ( echo ">> version is not set, usage: make release version=\"v1.2.3\" "; exit 1 )

tag: check_env check-branch check-git-clean verify ## create a tag and push to git (version="v1.2.3")
	@git diff --quiet || ( echo 'git is in dirty state' ; exit 1 )
	@[ "${version}" ] || ( echo ">> version is not set, usage: make release version=\"v1.2.3\" "; exit 1 )
	@git tag -d $(version) || true
	@git tag -a $(version) -m "Release version: $(version)"
	@git push --delete origin $(version) || true
	@git push origin $(version) || true


#==========================================================================================
#  Help
#==========================================================================================
.PHONY: help
help: # Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

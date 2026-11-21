.DEFAULT_GOAL := help
UV ?= uv
NPM := cd frontend && $(UV) run npm

.PHONY: help
help: ## List commands
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Create the environment (Python incl. CLIP, Node via uv), build the UI, install git hooks
	$(UV) sync
	$(NPM) ci
	$(MAKE) ui
	git config core.hooksPath scripts/git-hooks
	@if [ -d .private.git ]; then git --git-dir=.private.git config core.hooksPath scripts/git-hooks; fi

.PHONY: check
check: ui-check ui ## Everything CI runs: Python format/lint/types, UI checks, all tests + coverage
	$(UV) run ruff format --check .
	$(UV) run ruff check .
	$(UV) run mypy
	$(UV) run pytest --cov --cov-report=term

.PHONY: fix
fix: ## Auto-format and apply safe lint fixes (Python and UI)
	$(UV) run ruff format .
	$(UV) run ruff check --fix .
	$(NPM) run format

.PHONY: test
test: ## Python tests only
	$(UV) run pytest -q

.PHONY: ui
ui: ## Build the web interface into src/calico/web/static/dist
	$(NPM) run build

.PHONY: ui-check
ui-check: ## UI lint (Biome), types (tsc) and tests (Vitest)
	$(NPM) run check

.PHONY: ui-dev
ui-dev: ## UI dev server with hot reload (run `make serve` alongside it)
	$(NPM) run dev

.PHONY: api-types
api-types: ## Regenerate the UI's TypeScript API types from the server's OpenAPI schema
	$(UV) run python -m calico.web.openapi > frontend/openapi.json
	$(UV) run python -m calico.web.openapi --fingerprint > frontend/src/api/fingerprint.ts
	$(NPM) run api-types

.PHONY: serve
serve: ## Start Calico (web interface on http://127.0.0.1:8765)
	$(UV) run calico serve

.PHONY: private-status
private-status: ## What changed in the private material (when this checkout has it)
	@scripts/private.sh status

.PHONY: private-save
private-save: ## Commit the private material and push it to calico-private
	@scripts/private.sh save

.PHONY: deploy-local
deploy-local: ## Update this computer's running Calico from GitHub (REF=branch to try one; needs the private deploy script)
	@if [ -x .private/deploy-local.sh ]; then .private/deploy-local.sh $(or $(REF),main); else echo "No deploy script in this checkout."; fi

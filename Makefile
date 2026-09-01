.PHONY: help setup lint format type test test-all test-oscd cov data demo tune eval e2e baselines vlm-report figures controls docker docker-demo clean

PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
endif

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

setup: ## Create the venv and install with dev extras
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

lint: ## Run ruff (lint + format check)
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

format: ## Auto-fix lint and formatting
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/ruff format src tests scripts

type: ## Type-check with mypy
	$(BIN)/mypy

test: ## Run the offline test suite (no network, no API spend)
	$(BIN)/pytest

test-all: ## Include dataset-dependent tests (requires `make data`)
	$(BIN)/pytest -m "not vlm"

test-oscd: ## Only the tests that need the real dataset on disk
	$(BIN)/pytest -m oscd

cov: ## Offline tests with a coverage report
	$(BIN)/pytest --cov=satchangegate --cov-report=term-missing --cov-report=html

data: ## Download the 13-band OSCD dataset (~513 MB, checksum-verified)
	$(BIN)/satchangegate download-oscd

demo: ## Clone-to-result on committed fixtures; no dataset, no API key
	$(BIN)/python scripts/demo.py

tune: ## Fit gate thresholds on the train split (test split held out)
	$(BIN)/satchangegate tune --split train

eval: ## Out-of-sample evaluation on the held-out test split
	$(BIN)/satchangegate eval --split test

e2e: ## Funnel with measured cost, gate only (add VLM=1 to call the API)
ifdef VLM
	$(BIN)/satchangegate e2e --split test --vlm --max-vlm-calls 20
else
	$(BIN)/satchangegate e2e --split test --no-vlm
endif

baselines: ## Compare the rule gate against learned models (needs .[baseline])
	$(BIN)/satchangegate baselines

vlm-report: ## Recompute the second-tier figures from the e2e ledger (no API calls)
	$(BIN)/satchangegate vlm-report --split test

figures: ## Regenerate the README figures
	$(BIN)/python scripts/make_figures.py

controls: ## Offline control battery
	$(BIN)/satchangegate dev-tests

docker: ## Build the container
	docker build -t satchangegate:latest .

docker-demo: ## Check the container's config resolves from the wheel
	docker run --rm satchangegate:latest verify || true

clean: ## Remove caches and generated reports
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

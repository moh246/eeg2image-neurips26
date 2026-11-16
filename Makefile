# Repeatable entry points. `make setup` then `make check` is the clean-clone smoke test.
PYTHON ?= python3.12
VENV   ?= .venv
BIN    := $(VENV)/bin
CONFIG ?= configs/experiment/smoke_ridge.yaml

# Pinned organisers' starter kit, used only by `contract-replay`.
KIT_REPO   := https://github.com/neural-interfaces26/2026-competition.git
KIT_COMMIT := bddaff3dc6bd4b7300d531797685279a22c7bbf9
KIT_DIR    := .cache/starter-kit

.PHONY: setup setup-official lint format test check train results contract-replay parity data clean

setup:  ## create the virtualenv and install the package with dev tools
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev,data]"

setup-official:  ## add the organisers' pinned stack (large; needed for contract-replay)
	$(BIN)/python -m pip install -e ".[dev,data,targets,official]"

lint:  ## static checks, as run in CI
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## apply formatting and safe lint fixes
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:  ## unit and integration tests, as run in CI
	$(BIN)/python -m pytest

check: lint test  ## everything CI runs

train:  ## run one experiment: make train CONFIG=configs/experiment/<name>.yaml
	$(BIN)/python scripts/train.py --config $(CONFIG)

results:  ## regenerate the results tables in README.md and docs/report.md from run records
	$(BIN)/python scripts/summarize_runs.py --write README.md
	$(BIN)/python scripts/summarize_runs.py --write docs/report.md

$(KIT_DIR):
	git clone --quiet $(KIT_REPO) $(KIT_DIR)
	git -C $(KIT_DIR) checkout --quiet $(KIT_COMMIT)

contract-replay: $(KIT_DIR)  ## score parity with, and replay through, the organisers' code
	STARTER_KIT_DIR=$(KIT_DIR) $(BIN)/python -m pytest -m official -v
	$(BIN)/python scripts/replay_in_starter_kit.py --kit $(KIT_DIR)

parity:  ## compare our preprocessing with the organisers' extractor on two real recordings
	$(BIN)/python scripts/prepare_alljoined.py --download-only --no-stimuli \
		"data.subjects=[1]" "data.sessions=[1]" "data.runs=[1, 2]"
	$(BIN)/python -m pytest tests/test_preprocess_parity.py -v

data:  ## download and preprocess all of Alljoined-1.6M (about 12 GB of disk; see data/README.md)
	$(BIN)/python scripts/prepare_alljoined.py --config configs/data/alljoined16m.yaml

clean:
	rm -rf .pytest_cache .ruff_cache outputs

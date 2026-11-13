# Repeatable entry points. `make setup` then `make check` is the clean-clone smoke test.
PYTHON ?= python3.12
VENV   ?= .venv
BIN    := $(VENV)/bin
CONFIG ?= configs/experiment/smoke_ridge.yaml

# Pinned organisers' starter kit, used only by `contract-replay`.
KIT_REPO   := https://github.com/neural-interfaces26/2026-competition.git
KIT_COMMIT := bddaff3dc6bd4b7300d531797685279a22c7bbf9
KIT_DIR    := .cache/starter-kit

.PHONY: setup setup-official lint format test check train contract-replay clean

setup:  ## create the virtualenv and install the package with dev tools
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

setup-official:  ## add the organisers' pinned stack (large; needed for contract-replay)
	$(BIN)/python -m pip install -e ".[dev,data,official]"

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

$(KIT_DIR):
	git clone --quiet $(KIT_REPO) $(KIT_DIR)
	git -C $(KIT_DIR) checkout --quiet $(KIT_COMMIT)

contract-replay: $(KIT_DIR)  ## score parity with, and replay through, the organisers' code
	STARTER_KIT_DIR=$(KIT_DIR) $(BIN)/python -m pytest -m official -v
	$(BIN)/python scripts/replay_in_starter_kit.py --kit $(KIT_DIR)

clean:
	rm -rf .pytest_cache .ruff_cache outputs

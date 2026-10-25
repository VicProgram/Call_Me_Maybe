export STORAGE:=/home/$(shell whoami)/sgoinfre/vic_cache

export UV_CACHE_DIR:=$(STORAGE)/uv-cache
export UV_PROJECT_ENVIRONMENT:=$(STORAGE)/venv
export UV_PYTHON_INSTALL_DIR:=$(STORAGE)/python

export HF_HOME:=$(STORAGE)/huggingface
export HF_HUB_CACHE:=$(HF_HOME)/hub

export TMPDIR:=$(STORAGE)/tmp

export VIRTUAL_ENV:=$(UV_PROJECT_ENVIRONMENT)


MYPY_FLAGS  = --warn-return-any \
              --warn-unused-ignores \
              --ignore-missing-imports \
              --disallow-untyped-defs \
              --check-untyped-defs

FLAKE8_EXCLUDE = --exclude=data,llm_sdk,.venv
MYPY_EXCLUDE   = --exclude data --exclude llm_sdk --exclude .venv --exclude tests

all: install run

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

test:
	@echo "Running tests (no model, no torch)..."
	uv run --no-project --with numpy --with pydantic --with pytest pytest -q

lint:
	@echo "Checking linter..."
	@status=0; \
	echo ""; \
	echo "========== FLAKE8 =========="; \
	uv run flake8 . $(FLAKE8_EXCLUDE) || status=1; \
	echo ""; \
	echo "=========== MYPY ===========" ; \
	uv run mypy . $(MYPY_FLAGS) $(MYPY_EXCLUDE) || status=1; \
	echo ""; \
	exit $$status

lint-strict:
	@echo "Checking linter (strict)..."
	@status=0; \
	echo ""; \
	echo "========== FLAKE8 =========="; \
	uv run flake8 . $(FLAKE8_EXCLUDE) || status=1; \
	echo ""; \
	echo "=========== MYPY ===========" ; \
	uv run mypy . --strict $(MYPY_EXCLUDE) || status=1; \
	echo ""; \
	exit $$status

clean:
	@echo "Cleaning temporary files...\n"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf data/output
	@echo "\nRemoving virtual environment and caches...\n"
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf $(VIRTUAL_ENV)
	@echo "\nCleaning uv cache\n"
	uv cache clear

fclean: clean
	@echo "\nCleaning vic_cache\n"
	rm -rf $(STORAGE)

re: fclean all

.PHONY: all install run debug test lint lint-strict clean fclean re
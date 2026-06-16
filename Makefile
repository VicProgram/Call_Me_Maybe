export STORAGE:=/home/$(shell whoami)/sgoinfre/vic_cache

export UV_CACHE_DIR:=$(STORAGE)/uv-cache
export UV_PROJECT_ENVIRONMENT:=$(STORAGE)/venv
export UV_PYTHON_INSTALL_DIR:=$(STORAGE)/python

export HF_HOME:=$(STORAGE)/huggingface
export HF_HUB_CACHE:=$(HF_HOME)/hub

export TMPDIR=$(STORAGE)/tmp

#directory changed to achieve space

VENV        = .venv
PYTHON      = $(VENV)/bin/python3
PIP         = $(VENV)/bin/pip


MAIN        = src/__main__.py
CONFIG      = config.txt
REQS        = requirements.txt
# OUTPUT_FILE = 


MYPY_FLAGS  = --warn-return-any \
              --warn-unused-ignores \
              --ignore-missing-imports \
              --disallow-untyped-defs \
              --check-untyped-defs


FLAKE8_EXCLUDE = --exclude=$(VENV),data,llm_sdk
MYPY_EXCLUDE   = --exclude $(VENV) --exclude data --exclude llm_sdk


all: install run


venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		uv venv $(VENV); \
	fi


install: venv
	@echo "Installing dependencies from $(REQS)..."
	uv pip install -r $(REQS)


run: venv
	uv run $(MAIN) 
# 	$(CONFIG)


# debug: venv
# 	@echo "Starting debugger (pdb)..."
# 	uv run -m python3 pdb $(MAIN) $(CONFIG)


lint:
	@echo "Comprobando linter..."
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
	@echo "Comprobando linter (estricto)..."
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

	rm -rf .mypy_cache
	rm -f $(OUTPUT_FILE)

	@echo "\nRemoving virtual environment...\n"
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf $(VENV)

	@echo "\n cleaning uv cache\n"
	uv cache clear

re: clean all


.PHONY: all venv install run debug lint lint-strict clean re
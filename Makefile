# ── Configuración del entorno virtual ────────────────────────────
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
	uv run python3 $(MAIN) $(CONFIG)


debug: venv
	@echo "Starting debugger (pdb)..."
	uv run -m python3 pdb $(MAIN) $(CONFIG)


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
	@echo "Cleaning temporary files..."

	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

	rm -rf .mypy_cache
	rm -f $(OUTPUT_FILE)

	@echo "Removing virtual environment..."
	rm -rf $(VENV)

build:
	$(PYTHON) -m build


re: clean all


.PHONY: all venv install run debug lint lint-strict clean re
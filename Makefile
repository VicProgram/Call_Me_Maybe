# ── Configuración del entorno virtual ────────────────────────────
VENV        = venv
PYTHON      = $(VENV)/bin/python3
PIP         = $(VENV)/bin/pip


MAIN        = __main__.py
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
		python3 -m venv $(VENV); \
	fi


install: venv
	@echo "Installing dependencies from $(REQS)..."
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r $(REQS) --quiet


run: venv
	$(PYTHON) $(MAIN) $(CONFIG)

debug: venv
	@echo "Starting debugger (pdb)..."
	$(PYTHON) -m pdb $(MAIN) $(CONFIG)


lint: venv
	@echo "Checking lint dependencies..."
	@$(PIP) show flake8 >/dev/null 2>&1 || $(PIP) install flake8 --quiet
	@$(PIP) show mypy >/dev/null 2>&1 || $(PIP) install mypy --quiet

	@status=0; \
	echo ""; \
	echo "========== FLAKE8 =========="; \
	$(PYTHON) -m flake8 . $(FLAKE8_EXCLUDE) || status=1; \
	echo ""; \
	echo "=========== MYPY ===========" ; \
	$(PYTHON) -m mypy . $(MYPY_FLAGS) $(MYPY_EXCLUDE) || status=1; \
	echo ""; \
	exit $$status


lint-strict: venv
	@echo "Checking lint dependencies..."
	@$(PIP) show flake8 >/dev/null 2>&1 || $(PIP) install flake8 --quiet
	@$(PIP) show mypy >/dev/null 2>&1 || $(PIP) install mypy --quiet

	@status=0; \
	echo ""; \
	echo "========== FLAKE8 =========="; \
	$(PYTHON) -m flake8 . $(FLAKE8_EXCLUDE) || status=1; \
	echo ""; \
	echo "=========== MYPY ===========" ; \
	$(PYTHON) -m mypy . --strict $(MYPY_FLAGS) $(MYPY_EXCLUDE) || status=1; \
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
# Call Me Maybe

**Function calling system with constrained decoding for a local LLM.**

Given a natural language request (e.g. *"What is the sum of 2 and 3?"*), this system uses the **Qwen3-0.6B** language model to output a structured function call:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "fn_name": "fn_add_numbers",
  "args": {"a": 2.0, "b": 3.0}
}
```

Instead of relying on the model to write valid JSON on its own (which fails ~70% of the time with a model this small), the system implements **constrained decoding**: at each token generation step, only tokens that keep the output valid are allowed — everything else is blocked.

---

## Project structure

```
call-me-maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py           Entry point (argparse, main loop)
│   ├── models.py             Pydantic data models
│   ├── vocab.py              Vocabulary index (token ↔ ID mapping)
│   ├── constrained_decoder.py  Constrained decoding engine (JSONGenerator)
│   ├── function_caller.py    Orchestrator (select function, extract args)
│   ├── prompt_builder.py     Prompt construction for the LLM
│   └── tools.py              File I/O helpers (read/write JSON)
├── llm_sdk/                  SDK wrapping Qwen3-0.6B via Hugging Face
├── data/
│   ├── input/
│   │   ├── function_definitions.json    Available function definitions
│   │   └── function_calling_tests.json   Test prompts
│   └── output/
│       └── function_calling_results.json  Generated results
├── pyproject.toml
├── Makefile
└── README.md
```

---

## How it works

1. **Load** the function definitions and test prompts from JSON files
2. **Load** the Qwen3-0.6B model via the SDK
3. For each prompt:
   - **Select function**: build a prompt listing available functions, score each candidate name by its log-probability, pick the best one
   - **Extract arguments**: for each parameter, build an extraction prompt and use token-by-token constrained decoding to generate the value (only numeric tokens for numbers, non-structural tokens for strings, "true"/"false" for booleans)
   - **Assemble** the result into a `FunctionCall` object
4. **Write** the results to `data/output/function_calling_results.json`

### Constrained decoding

At each token generation step:
1. Get the logits (scores) from the model for every token in the vocabulary
2. Determine which tokens are valid for the current state (e.g., only digits when expecting a number)
3. Set all invalid token scores to -infinity (they can never be chosen)
4. Pick the remaining highest-scoring token

This makes it **physically impossible** for the model to produce invalid output.

---

## Usage

```bash
make install      # Install dependencies (uv sync)
make run          # Run the system
make lint         # Run flake8 and mypy
```

Or manually:

```bash
uv run python -m src
uv run python -m src --input data/input --output data/output/function_calling_results.json
```

---

## Requirements

- Python >= 3.10
- `uv` package manager

Dependencies (installed via `uv sync`): `numpy`, `pydantic`, `accelerate`, and the local `llm-sdk` (which requires `torch`, `transformers`, `huggingface-hub`).

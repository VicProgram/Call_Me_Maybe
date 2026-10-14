*This project has been created as part of the 42 curriculum by vabad-ro.*

# Call Me Maybe

**Function calling for LLMs through constrained decoding.**

Call Me Maybe turns a natural-language request into a structured function call
(`{"prompt": ..., "name": ..., "parameters": {...}}`) using a small language model
(`Qwen/Qwen3-0.6B`) whose output is **never** allowed to become malformed JSON.
Instead of trusting the model to produce valid JSON, a finite state machine
forces every structural token and restricts the vocabulary at each generation
step, so the only thing the model actually decides is *which* available function
to call and *what* values its arguments should take.

---

## Table of contents

- [Description](#description)
- [Explanation of the algorithm](#explanation-of-the-algorithm)
- [Design decisions](#design-decisions)
- [Performance analysis](#performance-analysis)
- [Challenges encountered](#challenges-encountered)
- [Testing strategy](#testing-strategy)
- [Requirements](#requirements)
- [Instructions](#instructions)
- [Usage examples](#usage-examples)
- [Data formats](#data-formats)
- [Project layout](#project-layout)
- [Make targets](#make-targets)
- [Resources](#resources)

---

## Description

Large language models are excellent at understanding language but notoriously
unreliable at emitting strict formats. A 0.6B-parameter model asked to "write the
JSON for this function call" frequently invents fields, forgets quotes, or
answers in prose. Small models make this failure mode so common that prompting
alone cannot be trusted.

The goal of this project is to build a **function-calling pipeline that is
correct by construction**. Given a list of function definitions and a set of
user prompts, the program must, for each prompt:

1. **Select** the most appropriate function from the available definitions.
2. **Extract** the argument values required by that function.
3. **Emit** a well-formed JSON object with exactly three keys: `prompt`,
   `name` and `parameters`.

The core idea is to *decouple formatting from reasoning*. The decoder owns the
syntax (braces, quotes, commas, key names, token types); the model only owns the
semantics (which function, which value). Because the model can only ever choose
among tokens that keep the JSON valid, the output is **always parseable**, no
matter how weak the model is. Correctness of the *values* still depends on the
model — that boundary is deliberate and discussed below.

---

## Explanation of the algorithm

The approach is **constrained decoding** (also called *guided* or *grammar-
constrained* generation): at each step the model produces logits over its whole
vocabulary (≈151k tokens for Qwen3-0.6B), and before choosing the next token we
set the logits of all invalid tokens to `-inf`, then take the argmax. A token
that is forbidden can never win, because `-inf` is lower than any real logit
(the choice of `-inf` instead of `0` matters: many valid logits are negative).

Generation is driven by a **finite state machine** (`JSONState`) that walks the
JSON structure from left to right. Every state is either *forced* (its text is
written verbatim) or *model-driven* (the model picks tokens under a mask).

### Step 1 — Build the selection prompt

`fun_select_prompt` (`src/prompt_builder.py`) builds a single text prompt that lists
the available functions with their descriptions and ends with the user request
and the cue `Function to call: `. The model is expected to continue with a
function name.

### Step 2 — Select the function

`JSONGenerator.select_function_name` scores every candidate function name
**token by token**:

- For each token of a candidate name, it computes the logits of the next token
  in the context of the selection prompt *plus the name tokens already
  emitted*.
- Each logit vector is normalized with **log-softmax** and the
  log-probability of that specific name token is added to a running score.
- The candidate with the **highest summed log-probability** wins.

Using log-probabilities (not raw logits) is essential: it removes the length
bias (a 4-token name should not beat a 2-token name just by summing more
numbers) and makes the per-step scales comparable, because summing log-probs
equals the log-probability of the whole name sequence. The context is copied
(`ids = list(prompt_ids)`) for each candidate so one name never contaminates the
scoring of the next.

### Step 3 — Force the known prefix

The user prompt is already known, so it is **written verbatim** into the output
using `json.dumps` instead of being regenerated:

```python
opening = '{"prompt": ' + json.dumps(prompt) + ', "name": "'
```

This is the same principle as the rest of the decoder: force what is known, ask
the model only for what is unknown. `json.dumps` also guarantees correct
escaping when the prompt contains quotes (e.g. the regex cases).

### Step 4 — Walk the state machine

The generator loops while the state is not `FINISHED`. Each call to
`select_state` returns `(tokens, next_state)`:

| State | Source | Emits |
|---|---|---|
| `FN_NAME` | Forced | The selected function name |
| `AFTER_NAME` | Forced | `", "parameters": {` — or `", "parameters": {}` if the function takes no parameters |
| `PARAM_KEY` | Forced | `"param": ` — or `"param": "` when the parameter is a string |
| `PARAM_VALUE` | **Model** | The argument value, under the mask of its type |
| `NEXT_PARAM` | Forced | `, "next": ` — or the closing brace when no parameters remain |
| `END` | Forced | `}` (closes the outer object) |
| `FINISHED` | — | End of generation |

The only state where the model chooses anything is `PARAM_VALUE`. Forced text
is turned into token ids with `model.encode(...)`, so multi-token literals are
handled gracefully.

### Step 5 — Mask by parameter type

The token masks are precomputed once in `JSONGenerator.__init__`, because
scanning the whole vocabulary on every step would be wasteful. Four types are
supported:

| Type | Allowed tokens | Terminator |
|---|---|---|
| `number` | digits, one optional leading `-`, one optional `.` (sub-state machine) | a token made only of ` ,\n\t` |
| `integer` | like `number` but the `.` is never allowed | a token made only of ` ,\n\t` |
| `string` | tokens whose decoded text is valid string content: any char except a bare `"`, a bare `\` or a raw control char (`{ } [ ]` are legal; `\"`, `\\`, `\n`, `\uXXXX`… are accepted) | a token containing an unescaped `"` (the text before the quote is kept) |
| `boolean` | the exact tokens `true` / `false` | none (one token finishes the value) |

Terminators are *added to the allowed set* so the model is able to choose them;
otherwise the mask would forbid the very token that ends the value and the loop
would never terminate. Numbers use `search_characters`, so their terminators are
tokens made *only* of separators: a number must not swallow the comma that
separates parameters. Strings, by contrast, are classified by decoding every
token once and matching its text against `_STRING_CHUNK`, so a string may end
with a quote attached to text; when that happens, the text before the quote is
kept as content and only the quote closes the value.

Numbers are not masked with a single flat set: a small sub-state machine
(`start → sign → int → dot → frac`) changes the mask at every step, so a
syntactically invalid number (`--`, `1.2.3`, `.`) can never be produced.
`integer` is exactly the same machine without the `dot`/`frac` states. Each type
is handled by its own value generator (`value_generators`), so an unknown type
raises an error instead of silently falling back to another one.

### Step 6 — Decode and parse

The emitted tokens are turned back into text with `model.decode`, then parsed
with `json.loads` in `FunctionCaller.resolve`. Since every structural token was
validated by the FSM, this parse is expected to always succeed; the `try/except`
around it is a safety net, not the main mechanism. After parsing, every value is
re-checked against its declared type (`bool` is a subclass of `int`, so a
boolean can never pass as an `integer`), a `number` is coerced to `float` so
`2` is emitted as `2.0`, and an unexpected parameter name or function name is
rejected.

---

## Design decisions

- **Constrained decoding over fine-tuning or few-shot prompting.** The project
  forbids training, and few-shot prompting is unreliable for a 0.6B model.
  Masking the vocabulary is model-agnostic, needs no extra parameters, and gives
  a hard structural guarantee.

- **A finite state machine instead of a formal JSON grammar.** The output schema
  is *fixed* (`prompt`, `name`, `parameters` with known keys), so a small explicit
  FSM is simpler, faster to reason about, and easier to debug than a general
  grammar engine. It also keeps the code free of external parsing libraries.

- **Force the prompt instead of letting the model echo it.** The prompt is known
  input, so regenerating it only adds risk. The model used to recapitalize it
  (`Greet shrek` → `Greet Shrek`), and that echo leaked into string arguments
  (the extracted `name` became `Shrek`). Writing it verbatim with `json.dumps`
  fixed both the capitalization drift and quoting/escaping.

- **Score function names with summed log-probabilities.** This gives a
  principled comparison between names of different lengths and avoids raw-logit
  scale arbitrariness (see Step 2).

- **Precompute masks once in the constructor.** Iterating over ~151k vocabulary
  entries is expensive; doing it per generation would dominate the runtime.

- **Pydantic models at the boundaries.** `FunctionDefinition`, `TestPrompt` and
  `FunctionCall` validate all input/output data, so a malformed definition file
  fails early with a clear message instead of producing a mysterious crash
  later. The internal generation path stays on plain dicts/lists for speed.

- **Separate the vocabulary from the generator.** `VocabIndex` is created
  outside `JSONGenerator` and injected, which keeps the generator easy to test
  and reuse and decouples it from how the vocab file is located.

---

## Performance analysis

**Accuracy.** On the project's own test set (11 prompts covering 5 functions),
function **selection** is correct in all cases and the output is valid JSON in
all cases — those two are guaranteed by construction and visible in the results
that `make run` writes to `data/output/function_calling_results.json`. Argument
**extraction** is correct for the arithmetic, greeting, string-reversal and
square-root cases — including
the string arguments, which now keep whatever text sits in front of their
closing quote. The regex cases are where the model — not the decoder —
struggles: for example `"Replace all vowels in 'Programming is fun' with
asterisks"` yields the regex `"([aeiouAEIOU])"` with a replacement `" *"` (a
stray leading space), and `"Substitute the word 'cat' with 'dog' …"` yields the
greedy pattern `"cat.*cat"` instead of `"cat"`. All of those values are valid
JSON strings and valid strings for the argument type; they are simply the wrong
*choice*. This is the expected boundary: the decoder guarantees form, the model
is responsible for content.

**Speed.** Generation cost is dominated by the number of forward passes through
the model. Function selection costs one forward pass per token of *every*
candidate name (a linear scan over all functions), and value extraction costs
one pass per generated token. In exchange, masking is pure NumPy and negligible
compared to inference. Running on CPU is possible but slow; a CUDA GPU (or Apple
MPS) gives a large speed-up and is selected automatically by `llm_sdk`.
Precomputing the masks means the one-off vocabulary scan does not repeat per
prompt.

**Reliability.** The structural guarantees are deterministic and independent of
the model: valid JSON, the three exact keys, a `name` drawn from the provided
definitions, and argument values respecting their declared type. The number and
integer masks are produced by a sub-state machine, so invalid numeric syntax is
impossible, and every parsed value is re-checked against its declared type in
`FunctionCaller.resolve` (a boolean can never pass as an integer). The residual
risk is model-side only (wrong function or wrong value); a `try/except` around
`json.loads` remains as a final safety net.

**Memory.** `Qwen3-0.6B` is intentionally small; `llm_sdk` uses `float16` on
GPU/MPS and `float32` on CPU, so the whole pipeline runs comfortably on a laptop.

---

## Challenges encountered

- **The model rewrote the prompt.** Letting the model regenerate the `prompt`
  field caused capitalization changes that leaked into argument extraction
  (`shrek` → `Shrek`). **Solved** by forcing the prompt verbatim and escaping it
  with `json.dumps`.

- **Numeric terminators.** A number value must end, but a digit mask alone has no
  way to stop. **Solved** by adding separator-only tokens (` ,\n\t`) to the
  allowed set and treating them as terminators, so the value ends exactly at the
  comma before the next key.

- **String terminators, and strings that need any text.** A string ends on a
  quote, but an early mask simply forbade `{ } [ ] \` inside a value, which made
  Windows paths, `\d+` regexes, character classes and templates impossible to
  emit. **Solved** by classifying every token from its decoded text with
  `_STRING_CHUNK`: content tokens must be valid JSON-string content (any
  character except a bare `"`, a bare `\` or a raw control character; `{ } [ ]`
  are legal and escapes such as `\"`, `\\`, `\n` or `\uXXXX` are accepted),
  while a terminator is any token with an unescaped `"` whose text before the
  quote is valid content. The quote is what closes the string, so text glued in
  front of it (`)"`) is preserved instead of discarded.

- **Length bias in selection.** Summing raw logits favored longer names.
  **Solved** by switching to summed log-softmax log-probabilities and by scoring
  each candidate in its own copied context.

- **Unreliable value choice.** Strings such as the `regex` argument are picked by
  the model and are sometimes semantically wrong. This is not fixable by
  "forcing the format harder" — documented as a model-capacity limitation rather
  than a decoder bug.

- **Numeric syntax.** An early version masked numbers with a single flat set of
  digit/`.`/`-` tokens, which could in principle emit `1.2.3`, `--` or an empty
  value. **Solved** with a sub-state machine that switches the mask at each step
  (see Step 5) and by re-validating the parsed value against its declared type.

- **Vocabulary handling.** The model ships a large `vocab.json`; it is loaded
  once into a `VocabIndex` with bidirectional `token⇄id` maps and specialized
  searches (`search_exact`, `search_characters`) so mask construction is a
  one-time cost.

---

## Testing strategy

Validation combines the provided test set, static analysis and manual
inspection — each layer checks a different property:

1. **End-to-end run.** `make run` processes
   `data/input/function_calling_tests.json` against
   `data/input/functions_definition.json` and writes
   `data/output/function_calling_results.json`.

2. **Format checks (decoder guarantees).** For every result: the file parses
   with `json.loads`; each entry has exactly the keys `prompt`, `name`,
   `parameters`; `name` is one of the available functions; each argument matches
   the declared type (numbers are JSON numbers, not `"2"`).

3. **Correctness checks (model behaviour).** Results are inspected against the
   expected function and arguments, starting from
   the easier cases (arithmetic, greeting, reversal) before the regex cases.
   Comparing them with the expectations is how the model-side errors described in
   *Performance analysis* were identified as value errors, not format errors.

4. **Prompt fidelity.** The `prompt` field of each result must be byte-identical
   to the original request — no recapitalization or rephrasing.

5. **Static analysis.** `make lint` runs `flake8` and `mypy` (with unused-ignore,
   return-any, untyped-def and untyped-body checks); `make lint-strict` runs
   `mypy --strict`. `pyproject.toml` also enables `strict` mode for mypy. Type
   hints and docstrings are used throughout `src/`.

6. **Manual isolation checks.** Individual functions (`log_softmax`,
   `apply_mask`, `select_best_token`, the `VocabIndex` searches) were exercised
   in isolation to confirm the masking logic before wiring it into the FSM.

`make lint` and `make lint-strict` run the static checks above; the rest of the
verification procedure is the end-to-end run plus manual inspection of the
results.

---

## Requirements

- Python **>= 3.10** (the code uses PEP 604 unions and `match` statements)
- [**uv**](https://docs.astral.sh/uv/) for dependency management
- A model served through the bundled `llm_sdk` — default `Qwen/Qwen3-0.6B`
- Optional: an NVIDIA GPU or Apple MPS for faster inference (CPU works too)

---

## Instructions

### Installation

```bash
make install      # equivalent to: uv sync
```

This installs the runtime dependencies (`numpy`, `pydantic`, `flake8`, `mypy`)
and the local `llm_sdk` package declared in `pyproject.toml`. The model weights
are downloaded from the Hugging Face Hub on first run.

#### Virtual environment location

The `Makefile` does not keep the virtual environment in the project root.
Instead it exports `UV_PROJECT_ENVIRONMENT` (together with `UV_CACHE_DIR` and
`HF_HOME`) so that everything lives under a single storage directory:

```make
export UV_PROJECT_ENVIRONMENT:=$(STORAGE)/venv
```

Because of this, `make install`, `make run`, `make lint`, etc. create and use
the environment at `$(STORAGE)/venv`, not at `./.venv`. If you run a bare `uv`
command instead (for example `uv sync` or `uv run ...`), `UV_PROJECT_ENVIRONMENT`
is not set and uv falls back to creating a `./.venv` in the project root. That
`.venv` is not required and is already listed in `.gitignore`; delete it with
`rm -rf .venv` if it appears. To avoid it, either use the `make` targets or export
the variable first:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/sgoinfre/vic_cache/venv"
export UV_CACHE_DIR="$HOME/sgoinfre/vic_cache/uv-cache"
```

### Running

```bash
make run          # equivalent to: uv run python -m src
```

With explicit paths (the form used by the subject):

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

`--input` may be a directory (in which case `functions_definition.json` and
`function_calling_tests.json` are read from it) or a single prompts file (in
which case the definitions are expected next to it). `--functions_definition`
overrides the definitions file explicitly, and `--output` is the path of the
results file; missing parent directories are created automatically.

### Debugging

```bash
make debug        # equivalent to: uv run python -m pdb -m src
```

### Linting

```bash
make lint         # flake8 + mypy (project flags)
make lint-strict  # flake8 + mypy --strict
```

### Cleaning

```bash
make clean        # caches and data/output
make fclean       # clean + local uv/HF storage
make re           # fclean + all
```

---

## Usage examples

### Default run

```bash
make run
```

This reads `data/input/functions_definition.json` (5 functions) and
`data/input/function_calling_tests.json` (11 prompts), loads the model
`Qwen/Qwen3-0.6B`, and writes the results to
`data/output/function_calling_results.json`. While it runs, the progress for
each prompt is printed to the console (selected function and extracted
arguments), followed by a summary with the number of successes and errors.

### Custom input/output

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output /tmp/results.json
```

### Single prompts file

```bash
uv run python -m src --input my_prompts.json --output out.json
# functions_definition.json must sit next to my_prompts.json
```

### Example input and output

Input prompt:

```json
{ "prompt": "What is the sum of 2 and 3?" }
```

Resulting entry in the output file:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

### Using the pipeline as a library

```python
from llm_sdk import Small_LLM_Model
from src.function_caller import FunctionCaller
from src.tools import load_function_def

model = Small_LLM_Model()                     # Qwen/Qwen3-0.6B
caller = FunctionCaller(model)
functions = load_function_def("data/input/functions_definition.json")

call = caller.resolve("Greet shrek", functions)
print(call.name, call.parameters)             # fn_greet {'name': 'shrek'}
```

---

## Data formats

### Function definitions — `data/input/functions_definition.json`

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

Every parameter (and `returns`) declares a `type`. The supported types are
`number`, `integer`, `string` and `boolean`; any other value is rejected when
the definitions file is loaded.

### Prompts — `data/input/function_calling_tests.json`

A list of objects with a `prompt` key (plain strings are also accepted):

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" }
]
```

### Results — `data/output/function_calling_results.json`

Each entry has exactly three keys: `prompt`, `name` and `parameters`.

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  }
]
```

---

## Project layout

```
src/
  __main__.py             CLI entry point (argparse, load, run, export)
  function_caller.py      Orchestrates selection + extraction (FunctionCaller)
  constrained_decoder.py  FSM and token masking (JSONGenerator)
  prompt_builder.py       Builds the function-selection prompt
  vocab.py                Token <-> id index and token searches (VocabIndex)
  models.py               Pydantic models and the JSONState enum
  tools.py                Input loaders and JSON exporter
data/
  input/                  Function definitions and test prompts
  output/                 Generated results
llm_sdk/                  Local inference SDK wrapping Hugging Face models
```

---

## Make targets

| Target | Description |
|---|---|
| `make all` | `install` followed by `run` |
| `make install` | Install dependencies (`uv sync`) |
| `make run` | Run the pipeline (`uv run python -m src`) |
| `make debug` | Run the pipeline under `pdb` (`uv run python -m pdb -m src`) |
| `make lint` | Run flake8 and mypy |
| `make lint-strict` | Run flake8 and mypy in `--strict` mode |
| `make clean` | Remove caches and the output directory |
| `make fclean` | `clean` plus the local uv/HF storage |
| `make re` | `fclean` followed by `all` |

---

## Resources

### Classic references on the topic

- 42 subject — *Call Me Maybe: Introduction to function calling in LLMs*.
- Hugging Face — *Transformers* documentation.
  https://huggingface.co/docs/transformers
- Qwen — *Qwen3-0.6B* model card.
  https://huggingface.co/Qwen/Qwen3-0.6B
- Davis, E. et al. — *Constrained Language Models Yield Few-Shot Semantic
  Parsers* (EMNLP 2021).
- Willard, B. T. & Louf, R. — *Efficient Guided Generation for Large Language
  Models* (Outlines, 2023).
- Geng, S. et al. — *Grammar-Constrained Decoding for Structured NLP Tasks
  without Fine-Tuning* (EMNLP 2023).
- Python documentation — `json`, `argparse`, `pathlib`, `enum`.
  https://docs.python.org/3/library/
- NumPy documentation — `argmax`, `full`, `exp`, `log`.
  https://numpy.org/doc/
- Pydantic documentation. https://docs.pydantic.dev/
- uv documentation. https://docs.astral.sh/uv/

### Use of AI

AI assistants were used as a support tool during development, not as a
replacement for understanding. Concretely:

- **Algorithm design and explanation.** AI was used to reason about constrained
  decoding, log-softmax scoring and the logit-masking strategy, and to turn
  those decisions into the FSM states in `src/models.py` and
  `src/constrained_decoder.py`.
- **Code scaffolding and refactoring.** AI helped draft the Pydantic models
  (`src/models.py`), the input/output helpers (`src/tools.py`), the prompt
  builder (`src/prompt_builder.py`) and the vocabulary index (`src/vocab.py`),
  and to modularize the pipeline into the single-responsibility modules above.
- **Debugging.** AI assisted in diagnosing the prompt-echo bug (forcing the
  `prompt` field with `json.dumps`) and the terminator/mask issues described in
  *Challenges encountered*.
- **Documentation.** AI helped write the docstrings throughout `src/` and the
  original version of this README.
- **This document.** This README was restructured and rewritten with AI
  assistance to meet the format required by the subject.

All AI-generated code was reviewed, tested and adapted before being committed;
final design choices and verification remain the author's responsibility.

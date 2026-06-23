import json
import re
from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model
from src.vocab import VocabIndex
from src.models import FunctionDefinition


my_model = Small_LLM_Model()
vocab = VocabIndex(my_model)
id_to_token = vocab.id_to_token

FUNCTION_NAMES: list[str] = []


def generate_json(
    prompt: str,
    functions: list[FunctionDefinition],
) -> dict[str, Any]:

    input_ids = my_model.encode(
        prompt
    ).flatten().tolist()

    generated: list[int] = []
    max_tokens = 100

    for _ in range(max_tokens):
        logits = np.array(
            my_model.get_logits_from_input_ids(input_ids + generated),
            dtype=np.float32
        )
        next_id = int(np.argmax(logits))
        token_str = id_to_token.get(next_id, "")

        if next_id == my_model._tokenizer.eos_token_id:
            break

        generated.append(next_id)
        full_text = my_model.decode(generated)

        if "\n\n" in full_text and len(full_text) > 5:
            break

    full_output = my_model.decode(generated).strip()
    print(f"  Model output: {full_output[:120]}...")

    fn_name = _find_function_name(full_output, functions)
    fn_def = next((fn for fn in functions if fn.name == fn_name), functions[0])
    fn_params = {k: v.type for k, v in fn_def.parameters.items()}
    param_order = list(fn_def.parameters.keys())

    json_obj = _extract_json(full_output)
    if json_obj is not None:
        args = json_obj.get("args", {})
    else:
        args = _extract_args_from_call(full_output, fn_name, param_order, fn_params)
        if not args:
            args = _extract_args_natural(full_output, fn_params)

    return {
        "fn_name": fn_name,
        "args": args
    }


def _find_function_name(
    text: str, functions: list[FunctionDefinition]
) -> str:
    text_lower = text.lower()

    fn_call = re.search(r"\b(fn_\w+)\s*\(", text_lower)
    if fn_call:
        name = fn_call.group(1)
        if any(fn.name == name for fn in functions):
            return name

    positions = []
    for fn in functions:
        idx = text_lower.find(fn.name.lower())
        if idx != -1:
            positions.append((idx, fn.name))

    if positions:
        positions.sort()
        return positions[0][1]

    for fn in functions:
        desc_keywords = [w for w in fn.description.lower().split() if len(w) > 3][:3]
        if desc_keywords and all(kw in text_lower for kw in desc_keywords):
            return fn.name

    return functions[0].name


def _extract_args_from_call(
    text: str,
    fn_name: str,
    param_order: list[str],
    fn_params: dict[str, str],
) -> dict[str, Any]:

    pattern = re.escape(fn_name) + r"\s*\(\s*(.*?)\s*\)"
    match = re.search(pattern, text)
    if not match:
        return {}

    args_str = match.group(1)
    if not args_str:
        return {}

    raw_args = _split_call_args(args_str)
    args: dict[str, Any] = {}

    for i, raw in enumerate(raw_args):
        if i >= len(param_order):
            break

        raw = raw.strip()
        kw_match = re.match(r'(\w+)\s*=\s*(.+)', raw)
        if kw_match:
            pname = kw_match.group(1)
            val = _parse_arg_value(kw_match.group(2), fn_params.get(pname, "string"))
            if pname in fn_params:
                args[pname] = val
        else:
            pname = param_order[i]
            args[pname] = _parse_arg_value(raw, fn_params.get(pname, "string"))

    return args


def _split_call_args(args_str: str) -> list[str]:
    args = []
    depth = 0
    current = []

    for ch in args_str:
        if ch in ("(", "[", "{"):
            depth += 1
            current.append(ch)
        elif ch in (")", "]", "}"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        args.append("".join(current).strip())

    return args


def _parse_arg_value(raw: str, param_type: str) -> Any:
    raw = raw.strip().strip("\"'")

    if param_type in ("number", "integer", "float"):
        try:
            val = float(raw)
            return int(val) if val == int(val) else val
        except ValueError:
            pass

    return raw


def _extract_json(text: str) -> dict[str, Any] | None:
    brace_depth = 0
    start = -1

    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start != -1:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

    return None


def _extract_args_natural(
    text: str, fn_params: dict[str, str]
) -> dict[str, Any]:
    args: dict[str, Any] = {}

    for pname, ptype in fn_params.items():
        patterns = [
            re.escape(pname) + r'["\s]*[:=]\s*"?([^"\s,}]+)"?',
        ]

        found = False
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                val: Any = match.group(1).strip()
                if ptype in ("number", "integer", "float"):
                    try:
                        val = float(val)
                        if val == int(val):
                            val = int(val)
                    except ValueError:
                        continue
                else:
                    val = val.strip("\"'")
                args[pname] = val
                found = True
                break

        if not found and ptype in ("number", "integer", "float"):
            nums = re.findall(r"-?\d+\.?\d*", text)
            if nums:
                val = float(nums[0])
                if val == int(val):
                    val = int(val)
                args[pname] = val

    return args

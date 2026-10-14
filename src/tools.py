import os
import json
from pathlib import Path
from typing import Any
from .models import FunctionDefinition, TestPrompt, FunctionCall


def json_reader(file_path: str | Path) -> Any:
    """Read a JSON file and return its content as a Python object.

    Args:
        file_path: The path to the JSON file.

    Returns:
        The parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain valid JSON.
    """

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: The file '{file_path}' was not found."
            )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in '{file_path}': {e}"
            )


def load_function_def(fun_def_json: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        fun_def_json: Path to the JSON file containing function definitions.

    Returns:
        A list of validated FunctionDefinition objects.

    Raises:
        ValueError: If the file cannot be read or a definition is invalid.
        TypeError: If the data is not a list.
    """
    try:
        data = json_reader(fun_def_json)

    except Exception as e:
        raise ValueError(
            f"Error al leer el archivo JSON '{fun_def_json}': {e}"
            )

    if not isinstance(data, list):
        raise TypeError(f"Error: the data {data} is not a list")

    functions_def = []
    for fun in data:
        try:
            function_obj = FunctionDefinition(**fun)
            functions_def.append(function_obj)
        except Exception as e:
            raise ValueError(
                f"Error processing function {fun}: {e}")

    return functions_def


def load_prompt(prompt_json: str) -> list[TestPrompt]:
    """Load test prompts from a JSON file.

    Supports both plain string prompts and dictionary prompts
    with a "prompt" key.

    Args:
        prompt_json: Path to the JSON file containing test prompts.

    Returns:
        A list of TestPrompt objects.

    Raises:
        ValueError: If the file cannot be read or a prompt is invalid.
        TypeError: If the data is not a list.
    """
    try:
        data = json_reader(prompt_json)
    except Exception as e:
        raise ValueError(
            f"Error al leer el archivo JSON '{prompt_json}' : {e}"
            )

    if not isinstance(data, list):
        raise TypeError(f"Error: the data {data} is not an array")

    prompt_list = []

    for prompt in data:
        try:
            if isinstance(prompt, str):
                prompt_obj = TestPrompt(prompt=prompt)
                prompt_list.append(prompt_obj)
            elif isinstance(prompt, dict):
                prompt_obj = TestPrompt(**prompt)
                prompt_list.append(prompt_obj)
            else:
                raise ValueError(
                    f"Error: the prompt {prompt!r} is not a "
                    "string or an object"
                )

        except Exception as e:
            raise ValueError(
                f"Error processing the prompt '{prompt}': {e}"
            )
    return prompt_list


def json_exporter(
        func_call_obj: list[FunctionCall], func_call_path: str | Path
        ) -> None:
    """Export function call results to a JSON file.

    Creates the output directory if it does not exist and writes
    the results as a formatted JSON array.

    Args:
        func_call_obj: List of FunctionCall objects to export.
        func_call_path: Path to the output JSON file.

    Raises:
        ValueError: If there is an error writing the output file.
    """
    func_dict = []

    for item in func_call_obj:
        new_item = item.model_dump()
        func_dict.append(new_item)

    try:
        parent_dir = os.path.dirname(func_call_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(func_call_path, "w", encoding="utf-8") as f:
            json.dump(func_dict, f, indent=4)
    except (OSError, TypeError) as e:
        raise ValueError(
            f"Error creating the output '{func_call_path}': {e}"
            )

import os
import json
from typing import Any
from .models import FunctionDefinition, TestPrompt, FunctionCall


def json_reader(file_path: str) -> Any:
    """
    Reads a JSON file and returns its content as a Python object.

    Args:
        file_path (str): The path to the JSON file.
        """

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")


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
        raise ValueError(f"Error al leer el archivo JSON '{fun_def_json}': {e}")

    if not isinstance(data, list):
        raise TypeError(f"Error: the data {data} is not a list")

    functions_def = []
    for fun in data:
        try:
            function_obj = FunctionDefinition(**fun)
            functions_def.append(function_obj)
        except Exception as e:
            raise ValueError(
                f"Error procesing function {fun}: {e}")

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
        raise ValueError(f"Error al leer el archivo JSON '{prompt_json}' : {e}")

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

        except Exception as e:
            raise ValueError(
                f"Error procesing the prompt '{prompt}': {e}"
            )
    return prompt_list


def json_exporter(func_call_obj: list[FunctionCall], func_call_path: str) -> None:
    """Export function call results to a JSON file.

    Creates the output directory if it does not exist and writes
    the results as a formatted JSON array.

    Args:
        func_call_obj: List of FunctionCall objects to export.
        func_call_path: Path to the output JSON file.

    Raises:
        ValueError: If there is an error writing the output file.
    """
    if not os.path.exists(func_call_path):
        os.makedirs(
            os.path.dirname(func_call_path), exist_ok=True
            )

    func_dict = []

    for item in func_call_obj:
        new_item = item.dict()
        func_dict.append(new_item)

    with open(func_call_path, "w", encoding="utf-8") as f:
        try:
            json.dump(func_dict, f, indent=4)
        except Exception as e:
            raise ValueError(
                f"Error creating the output: {e}"
                )

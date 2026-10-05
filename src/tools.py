import os
import json
from models import FunctionDefinition, TestPrompt, FunctionCall


def json_reader(file_path: str):
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


def load_function_def(fun_def_json: str):

    try:
        data = json_reader(fun_def_json)

    except Exception as e:
        raise ValueError(f"Error al leer el archivo JSON '{fun_def_json}': {e}")

    if not isinstance(data, list):
        raise TypeError(f"Error: the data {data} is not a list")  

    functions_def = []
    for fun in data:
        try:
            function_obj = FunctionDefinition(fun)
            functions_def.append(function_obj)
        except Exception as e:
            raise ValueError(
                f"Error procesing function {fun}: {e}")

    return functions_def


def load_prompt(prompt_json: str):

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


def json_exporter(func_call_obj: list[FunctionCall], func_call_path: str):

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
            json.dump(func_dict, f)
        except Exception as e:
            raise ValueError(
                f"Error creating the output: {e}"
                )

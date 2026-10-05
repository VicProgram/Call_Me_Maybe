import json
from pathlib import Path
from typing import Any
from src.models import FunctionDefinition, FunctionCall

def json_reader(path: Path) -> Any:

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        print(f"Error reading JSON: {e}")
        return None



def load_function_def(path: Path) -> list[FunctionDefinition]:

    raw = json_reader(path)

    if not isinstance(raw, list):
        raise ValueError("function_definitions.json must contain an array JSON")
    
    definitions = []

    for i, item in enumerate(raw):
        try:
            definitions.append(FunctionDefinition(**item))
        except Exception as e:
            raise ValueError(f"Invalid definition in {i}: {e}") from e
        
    return definitions




# def load_prompt(path: Path) -> list[str]:
#     raw = json_reader(path)

#     if not isinstance(raw, list):
#         raise ValueError("function_definitions.json must contain an array JSON")
    
#     prompts = []

#     for i, item in enumerate(raw):
#         if not isinstance(item, str):
#             raise ValueError (
#                 f"El prompt en índice {i} debe ser string, "
#                 f"pero es {type(item).__name__}"
#             )
#         prompts.append(item)

#     return prompts


def load_prompt(path: Path) -> list[str]:
    raw = json_reader(path)

    if not isinstance(raw, list):
        raise ValueError("function_definitions.json must contain an array JSON")
    
    prompts = []

    for i, item in enumerate(raw):
        # Si es un diccionario, extraemos el valor de la clave 'prompt'
        if isinstance(item, dict):
            texto = item.get("prompt")
            if texto is None:
                raise ValueError(f"El diccionario en el índice {i} no tiene la clave 'prompt'")
            prompts.append(str(texto))
        # Si ya es un string, lo añadimos directamente
        elif isinstance(item, str):
            prompts.append(item)
        # Si es cualquier otra cosa, lanzamos el error
        else:
            raise ValueError (
                f"El prompt en índice {i} debe ser string o dict, "
                f"pero es {type(item).__name__}"
            )

    return prompts


def json_exporter(results: list[FunctionCall], output_path: Path) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = [result.model_dump() for result in results]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Resultado guardado en: {output_path}")


# def valid_json(json_path: Path) -> bool:
#     try:
#         with open(json_path, 'r', encoding="utf-8") as f:
#             json.load(f)
#             return True

#     except json.JSONDecodeError as e:
#         print("Invalid JSON: Syntax Error")
#         print(f"Detail: {e.msg}")
#         return False

#     except FileNotFoundError:
#         print("File doesn't exists.")
#         return False


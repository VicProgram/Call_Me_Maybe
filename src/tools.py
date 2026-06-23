import json
from pathlib import Path
from typing import Any
from src.models import FunctionDefinition, FunctionCall

def json_reader(path: Path) -> Any:

    if not path.exists():
        print(f"Error: File doesn't exist in path {path.resolve()}")
        return False

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
        except:
             raise ValueError(f"Invalid definition in {i}: {e}") from e
        
    return definitions




def load_prompt(path: Path) -> list[str]:
    raw = json_reader(path)

    if not isinstance(raw, list):
        raise ValueError("function_definitions.json must contain an array JSON")
    
    prompts = []

    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError (
                f"El prompt en índice {i} debe ser string, "
                f"pero es {type(item).__name__}"
            )
        prompts.append(item)

    return prompts


def json_exporter(decode_text: str, output_path) -> None:

    output_file = output_path / "response.txt"
    
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for c in decode_text:
            f.write(c)
            f.write("\n")
            
    print(f"Resultado guardado en: {output_file}")

    # output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # # Convertir objetos Pydantic a diccionarios
    # output_data = [result.model_dump() for result in results]
    
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(output_data, f, indent=2)


# validacion rapida con json para saber si esta bien escrito 
# o si es un json correcto

def valid_json(json_path: Path) -> bool:
    try:
        with open(json_path, 'r', encoding="utf-8") as f:
            json.load(f)
            return True

    except json.JSONDecodeError as e:
        print("Invalid JSON: Syntax Error")
        print(f"Detail: {e.msg}")
        return False

    except FileNotFoundError:
        print("File doesn't exists.")
        return False


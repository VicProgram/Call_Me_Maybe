import json
from pathlib import Path
import sys

def json_reader(path: Path) -> str:

    if not path.exists():
        print(f"Error: El archivo no existe en la ruta {path.resolve()}")
        return False

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            function_list = []

            # PRINT FUNCTION LIST NOT NECESSARY

            for element in data:
                name = element.get('name', "")
                function_list.append(name)

            print("Functions:")
            for fn in function_list:
                print("-", fn)

            # PRINT FUNCTION LIST NOT NECESSARY

        return function_list

    except json.JSONDecodeError as e:
        print(f"Ocurrió un error al leer el JSON: {e}")
        return None


def json_exporter(decode_text: str, output_path) -> None:

    output_file = output_path / "response.txt"
    output_path.mkdir(parents=True, exist_ok=True) 

    with open(output_file, "w", encoding="utf-8") as f:
        for c in decode_text:
            f.write(c)
            f.write("\n")
            
    print(f"Resultado guardado en: {output_file}")


# validacion rapida con json para saber si esta bien escrito 
# o si es un json correcto

def valid_json(json_path: Path) -> bool:
    try:
        with open(json_path, 'r', encoding="utf-8") as f:
            json.load(f)
            return True

    except json.JSONDecodeError as e:
        print(f"JSON Inválido: Error de sintaxis en la línea {e.lineno}, columna {e.colno}.")
        print(f"Detalle: {e.msg}")
        return False

    except FileNotFoundError:
        print("El archivo no existe.")
        return False

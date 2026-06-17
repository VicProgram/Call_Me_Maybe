
import argparse
import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
# from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/input/functions_definition.json'))
    parser.add_argument('--output', type=Path, default=Path('data/output'))
    return parser.parse_args()

def json_reader(path: Path) -> bool:

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

        return True

    except json.JSONDecodeError as e:
        print(f"Ocurrió un error al leer el JSON: {e}")
        return False




def main() -> None:

    args = parse_args()
    my_model = Small_LLM_Model()

    print("Input:", args.input)
    print("Output:", args.output)
    print(type(args.input))

    print("---------------------------------------")
    print("cwd:", Path.cwd())
    print("input:", args.input.resolve())
    print("exists:", args.input.exists())
    print("---------------------------------------")


    print("Current dir:", Path.cwd())
    print("Input:", args.input.resolve())
    print("Exists:", args.input.exists())
    print("---------------------------------------")

    json_reader(args.input)

    print("---------------------------------------")
    print("---------------------------------------")

    # test_raw_inference(my_model)


if __name__ == "__main__":
    main()

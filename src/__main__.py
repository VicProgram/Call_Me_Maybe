import argparse
import json
from pathlib import Path

def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/input/functions_definition.json'))
    parser.add_argument('--output', type=Path, default=Path('data/output'))
    return parser.parse_args()

def json_reader(path: Path) -> None:

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        function_list = []
        for el in data:
            name = el.get('name', "")
            function_list.append(name)

        print("Functions:")
        for fn in function_list:
            print("-", fn)


def main() -> None:

    args = parse_args()

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


if __name__ == "__main__":
    main()

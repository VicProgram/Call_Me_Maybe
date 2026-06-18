
import argparse
import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .tools import json_exporter, json_reader, valid_json
import sys
# from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:

    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="LLM Function Calling System")

    parser.add_argument(
        '--input', type=Path, default=Path('data/input/functions_definition.json')
        )

    parser.add_argument(
        '--output', type=Path, default=Path('data/output')
        )
    
    return parser.parse_args()


def main() -> None:

    args = parse_args()

    try:
        my_model = Small_LLM_Model()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return False



    if args.input.is_dir():
        definitions_path = args.input / "function_definitions.json"
        tests_path = args.input / "function_calling_tests.json"
    else:
        definitions_path = args.input
        tests_path = args.input

    # tools json_reader()????
    # siempre un valid por si acaso??
    # tools json_exporter()????

    if valid_json(args.input):
        texto = json_reader(args.input)

    json_exporter(texto, args.output)


    # test_raw_inference(my_model)

if __name__ == "__main__":
    main()

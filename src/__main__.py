
import argparse
import sys

from llm_sdk import Small_LLM_Model
from pathlib import Path
from .tools import load_prompt, load_function_def, json_exporter
from .function_caller import FunctionCaller


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments as an argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="LLM Function Calling System - call me maybe"
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=None,
        help="Fichero JSON con las definiciones de funciones "
        "(default: <input>/functions_definition.json)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input"),
        help="Directorio o fichero de entrada (default: data/input)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Archivo de salida (default: "
        "data/output/function_calling_results.json)"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the program.

    Loads input files, initializes the model, processes all prompts,
    and writes the results to the output file.

    Returns:
        0 if all prompts were processed successfully, 1 otherwise.
    """
    args = parse_args()

    if args.input.is_dir():
        default_definitions = args.input / "functions_definition.json"
        tests_path = args.input / "function_calling_tests.json"
    else:
        default_definitions = args.input.parent / "functions_definition.json"
        tests_path = args.input

    if args.functions_definition is not None:
        definitions_path = args.functions_definition
    else:
        definitions_path = default_definitions

    print("Cargando archivos de entrada...")
    try:
        functions = load_function_def(definitions_path)
        prompts = load_prompt(tests_path)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"  {len(functions)} funciones cargadas")
    print(f"  {len(prompts)} prompts a procesar")

    print("\nCargando modelo LLM (puede tardar un momento)...")
    try:
        model = Small_LLM_Model()
    except Exception as e:
        print(f"Error cargando el modelo: {e}", file=sys.stderr)
        return 1
    print("  Modelo cargado.")

    caller = FunctionCaller(model)
    results = []
    errors = 0

    print("\nProcesando prompts...")
    for i, prompt_item in enumerate(prompts):
        prompt = prompt_item.prompt

        print(f"\n[{i+1}/{len(prompts)}] {prompt[:70]}...")
        print("  (Ejecutando inferencia, espera un momento...)")

        try:
            result = caller.resolve(prompt, functions)
            results.append(result)
            print(f"  ✓ {result.name}({result.parameters})")
        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            errors += 1

    print(f"\nEscribiendo resultados en {args.output}...")
    try:
        json_exporter(results, args.output)
    except Exception as e:
        print(f"Error escribiendo resultados: {e}", file=sys.stderr)
        return 1

    print(f"\n{'='*50}")
    print(f"Completado: {len(results)} éxitos, {errors} errores")
    print(f"Resultados en: {args.output}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

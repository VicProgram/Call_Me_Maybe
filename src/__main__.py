import argparse
import sys
# Importamos Small_LLM_Model del SDK. La ruta puede cambiar según la estructura del proyecto.
from llm_sdk import Small_LLM_Model
from pathlib import Path
from .tools import load_prompt, load_function_def, json_exporter
from .function_caller import FunctionCaller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Function Calling System - call me maybe"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input"),
        help="Directorio de entrada (default: data/input)"
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
    args = parse_args()

    if args.input.is_dir():
        definitions_path = args.input / "functions_definition.json"
        tests_path = args.input / "function_calling_tests.json"
    else:
        definitions_path = args.input.parent / "functions_definition.json"
        tests_path = args.input

    print("Cargando archivos de entrada...")
    try:
        functions = load_function_def(definitions_path)
        prompts = load_prompt(tests_path)
    except (FileNotFoundError, ValueError) as e:
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
        if isinstance(prompt_item, dict):
            prompt = (
                prompt_item.get("prompt")
                or prompt_item.get("text")
                or str(prompt_item)
            )
        else:
            prompt = str(prompt_item)

        print(f"\n[{i+1}/{len(prompts)}] {prompt[:70]}...")
        print("  (Ejecutando inferencia, espera un momento...)")

        try:
            result = caller.resolve(prompt, functions)
            results.append(result)
            print(f"  ✓ {result.fn_name}({result.args})")
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

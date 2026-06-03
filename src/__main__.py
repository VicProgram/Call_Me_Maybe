import argparse
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/input'))
    parser.add_argument('--output', type=Path, default=Path('data/output'))
    return parser.parse_args()


def main():
    """Función principal, entrada al programa."""
    args = parse_args()
    print(f'Leyendo entrada de: {args.input}')
    print(f'Escribiendo salida en: {args.output}')

if __name__ == "__main__":
    main()

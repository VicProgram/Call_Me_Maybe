# import argparse
# from pathlib import Path


# def parse_args() -> argparse.Namespace:
#     """Parsea los argumentos de línea de comandos."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', type=Path, default=Path('../data/input'))
#     parser.add_argument('--output', type=Path, default=Path('../data/output'))
#     return parser.parse_args()


# def main() -> None:
#     """Función principal, entrada al programa."""
#     args = parse_args()

#     print(args)

# if __name__ == "__main__":
#     main()

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 1. Asegurar ruta absoluta local
    model_path = os.path.abspath("./llm_sdk/Small_LLM_Model")
    
    print(f"Cargando modelo desde: {model_path}")
    
    # 2. Cargar tokenizador y modelo
    # Cargar tokenizador y modelo obligando a que busque SOLO en local
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32, # Usamos float32 temporalmente para asegurar compatibilidad de CPU
        device_map="auto"
    )


    # 3. Crear el prompt para la suma
    prompt = "Pregunta: ¿Cuánto es 2 + 3?\nRespuesta:"
    
    # 4. Tokenizar la entrada
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 5. Generar la respuesta del modelo
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20,     # Pocos tokens porque la respuesta es corta
            temperature=0.1,       # Temperatura baja para que sea determinista
            do_sample=False
        )
    
    # 6. Decodificar y mostrar el resultado
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Resultado del Modelo ---")
    print(respuesta)
    print("----------------------------")

if __name__ == "__main__":
    main()

# TEST PARA PROBAR CONEXION CON LA IA
import torch

def test_raw_inference(Small_LLM_Model) -> None:
    """Prueba básica para comprobar que el modelo responde usando el SDK."""
    print("\n--- Probando Inferencia Libre con el SDK ---")
    
    # Formato de Chat oficial para Qwen. Esto evita respuestas raras y bucles.
    test_prompt = "You are an assistant that selects the best function to call based on the user request. User request: I need to calculate the sum of 2 and 3"
    print(f"Prompt estructurado:\n{test_prompt}")

    # 1. Codificar el texto
    input_ids_raw = Small_LLM_Model.encode(test_prompt)
    
    # Extraer la lista plana de tokens (quitando los corchetes dobles de lote [[...]])
    if isinstance(input_ids_raw, list) and len(input_ids_raw) > 0 and isinstance(input_ids_raw[0], list):
        input_ids = input_ids_raw[0]
    elif hasattr(input_ids_raw, 'tolist'):
        input_ids = input_ids_raw.tolist()
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]
    else:
        input_ids = list(input_ids_raw)

    tokens_generados = []
    
    # Vamos a generar un máximo de 30 tokens
    for _ in range(20):
        # Enviar la secuencia acumulada al SDK
        logits = Small_LLM_Model.get_logits_from_input_ids(input_ids)
        
        # Convertir la salida a un Tensor para extraer la predicción de forma segura
        logits_tensor = torch.tensor(logits)
        
        if logits_tensor.ndim == 3:    # [batch, seq, vocab]
            next_token_logits = logits_tensor[0, -1, :]
        elif logits_tensor.ndim == 2:  # [seq, vocab]
            next_token_logits = logits_tensor[-1, :]
        else:                          # [vocab]
            next_token_logits = logits_tensor
            
        # Tomar el token con mayor probabilidad
        next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())

        # Romper el bucle si el modelo genera el token de fin de texto (Evita basura extra)
        if next_token_id in [151643, 151645, 0]: # IDs comunes de fin de cadena (<|im_end|>)
            break

        tokens_generados.append(next_token_id)
        input_ids.append(next_token_id)

    # Decodificar solo lo que el modelo ha respondido nuevo
    texto_decodificado = Small_LLM_Model.decode(tokens_generados)
    
    print("\nRESPUESTA GENERADA POR LLM:\n", texto_decodificado.strip())

from typing import Any
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCall
from src.constrained_decoder import generate_json, FUNCTION_NAMES
from src.prompt_builder import build_function_selection_prompt


class FunctionCaller:

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model

    def resolve(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> FunctionCall:

        # Rellenar FUNCTION_NAMES para que constrained_decoder sepa
        # qué nombres son válidos al restringir los tokens
        FUNCTION_NAMES.clear()
        FUNCTION_NAMES.extend(fn.name for fn in functions)

        # Construir el mapa de parámetros: {"a": "number", "b": "number"}
        # generate_json necesita saber el tipo de cada parámetro
        # para aplicar la restricción correcta (número vs string)
        #
        # Usamos el prompt de selección para dar contexto al modelo
        selection_prompt = build_function_selection_prompt(prompt, functions)

        # generate_json hace todo:
        # 1. Elige el nombre de función (restringido a FUNCTION_NAMES)
        # 2. Para cada parámetro, extrae el valor con el tipo correcto
        # 3. Devuelve {"fn_name": "...", "args": {...}}
        #
        # Pero necesita saber qué función eligió para conocer sus parámetros.
        # El problema es que generate_json necesita fn_params ANTES de saber
        # qué función eligió. Así que lo hacemos en dos pasos:

        # PASO 1: elegir la función
        chosen_name = self._select_function(selection_prompt, functions)

        # PASO 2: obtener los parámetros de esa función
        fn_def = next(fn for fn in functions if fn.name == chosen_name)
        fn_params = {
            name: param.type.value      # .value porque es un Enum
            for name, param in fn_def.parameters.items()
        }

        # PASO 3: generar el JSON completo con los argumentos
        result = generate_json(selection_prompt, chosen_name, fn_params)

        return FunctionCall(
            prompt=prompt,
            fn_name=result["fn_name"],
            args=result["args"],
        )

    def _select_function(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> str:

        import torch
        import numpy as np

        prompt_ids = self.model.encode(prompt)
        tensor = torch.tensor([prompt_ids])
        logits = self.model.get_logits_from_input_ids(tensor)

        # Última posición → puntuaciones para el siguiente token
        if len(logits.shape) == 3:
            next_logits = logits[0, -1, :].float().numpy()
        else:
            next_logits = logits[-1, :].float().numpy()

        # Para cada función, mirar la probabilidad de su primer token
        best_name = functions[0].name
        best_score = -float("inf")

        for fn in functions:
            fn_ids = self.model.encode(fn.name)
            if fn_ids and fn_ids[0] < len(next_logits):
                score = float(next_logits[fn_ids[0]])
                if score > best_score:
                    best_score = score
                    best_name = fn.name

        return best_name
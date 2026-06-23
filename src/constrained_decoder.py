import json
import math
from enum import Enum, auto
from typing import Any

import numpy as np
import torch

from llm_sdk import Small_LLM_Model
from src.vocab import VocabIndex


my_model = Small_LLM_Model()
vocab = VocabIndex(my_model)
token_to_id = vocab.token_to_ids
id_to_token = vocab.id_to_token

# Se rellena cuando se cargan las definiciones de funciones
FUNCTION_NAMES: list[str] = []


class State(Enum):
    START = auto()
    FN_NAME_KEY = auto()
    FN_NAME_COLON = auto()
    FN_NAME_VALUE = auto()
    COMMA_AFTER_FN = auto()
    ARGS_KEY = auto()
    ARGS_COLON = auto()
    ARGS_OPEN = auto()
    PARAM_KEY = auto()
    PARAM_COLON = auto()
    PARAM_VALUE = auto()
    PARAM_VALUE_NUM = auto()
    COMMA_OR_ARGS_CLOSE = auto()
    ARGS_CLOSE = auto()
    END = auto()


def token_of(text: str) -> int:
    if text not in token_to_id:
        raise ValueError(f"Token no encontrado: '{text}'")
    return token_to_id[text]


def get_valid_tokens(
    state: State,
    param_keys: list[str],
    param_types: dict[str, str],
    current_param: str,
    num_accumulated: str,
) -> set[int]:

    if state == State.START:
        return {token_of("{")}

    elif state == State.FN_NAME_KEY:
        return {token_of('"fn_name"')}

    elif state == State.FN_NAME_COLON:
        return {token_of(":")}

    elif state == State.FN_NAME_VALUE:
        valid = set()
        for fn in FUNCTION_NAMES:
            if fn in token_to_id:
                valid.add(token_of(fn))
        return valid                        

    elif state == State.COMMA_AFTER_FN:
        return {token_of(",")}

    elif state == State.ARGS_KEY:
        return {token_of('"args"')}

    elif state == State.ARGS_COLON:
        return {token_of(":")}

    elif state == State.ARGS_OPEN:
        return {token_of("{")}

    elif state == State.PARAM_KEY:
        valid = set()

        for key in param_keys:
            quoted = f'"{key}"'
            if quoted in token_to_id:
                valid.add(token_of(quoted))

        return valid                        

    elif state == State.PARAM_COLON:
        return {token_of(":")}

    elif state == State.PARAM_VALUE_NUM:
        valid = set()
        for text, tid in token_to_id.items():
            is_digits = text.lstrip("-").replace(".", "", 1).isdigit()
            is_minus = text == "-" and not num_accumulated
            is_dot = text == "." and "." not in num_accumulated

            if is_digits or is_minus or is_dot:
                valid.add(tid)

        for term in [",", "}", " ", "\n"]:
            if term in token_to_id:
                valid.add(token_of(term))
                
        return valid

    elif state == State.PARAM_VALUE:
        blocked = {"{", "}", "[", "]", "\n"}
        return {tid for text, tid in token_to_id.items() if text not in blocked}

    elif state == State.COMMA_OR_ARGS_CLOSE:
        if param_keys:
            return {token_of(",")}
        else:
            return {token_of("}")}

    elif state == State.ARGS_CLOSE:
        return {token_of("}")}

    elif state == State.END:
        return {token_of("}")}

    return set()


def generate_token(
    input_ids: list[int],
    state: State,
    param_keys: list[str],
    param_types: dict[str, str],
    current_param: str,
    num_accumulated: str,
) -> int:

    tensor = torch.tensor([input_ids])
    logits = my_model.get_logits_from_input_ids(tensor)

    if len(logits.shape) == 3:
        logits = logits[0, -1, :]
    elif len(logits.shape) == 2:
        logits = logits[-1, :]

    logits = logits.float().numpy()

    valid_tokens = get_valid_tokens(
        state, param_keys, param_types, current_param, num_accumulated
    )
    
    if not valid_tokens:
        print(f"No hay tokens válidos para el estado {state}. Usando fallback.")
        valid_tokens = set(range(len(logits)))
        
    for token_id in range(len(logits)):
        if token_id not in valid_tokens:
            logits[token_id] = -np.inf

    return int(np.argmax(logits))


def update_state(
    state: State,
    token: int,
    param_keys: list[str],
    num_accumulated: str,
    current_param: str,
    fn_params: dict[str, str],
) -> State:
    token_text = id_to_token.get(token, "")

    if state == State.START:
        return State.FN_NAME_KEY
    
    elif state == State.FN_NAME_KEY:
        return State.FN_NAME_COLON
    
    elif state == State.FN_NAME_COLON:
        return State.FN_NAME_VALUE
    
    elif state == State.FN_NAME_VALUE:
        return State.COMMA_AFTER_FN
    
    elif state == State.COMMA_AFTER_FN:
        return State.ARGS_KEY
    
    elif state == State.ARGS_KEY:
        return State.ARGS_COLON
    
    elif state == State.ARGS_COLON:
        return State.ARGS_OPEN
    
    elif state == State.ARGS_OPEN:
        return State.PARAM_KEY
    
    elif state == State.PARAM_KEY:
        return State.PARAM_COLON
    
    elif state == State.PARAM_COLON:                      
        param_type = fn_params.get(current_param, "string")
        if param_type in ("number", "integer", "float"):
            return State.PARAM_VALUE_NUM
        
        return State.PARAM_VALUE
    
    elif state == State.PARAM_VALUE_NUM:
        if token_text in [",", "}", " ", "\n"]:
            return State.COMMA_OR_ARGS_CLOSE
        
        return State.PARAM_VALUE_NUM
    
    elif state == State.PARAM_VALUE:
        if token_text in ["\n", '","']:
            return State.COMMA_OR_ARGS_CLOSE
        
        return State.PARAM_VALUE
    
    elif state == State.COMMA_OR_ARGS_CLOSE:
        if token_text == ",":
            return State.PARAM_KEY
        
        return State.END
    
    elif state == State.ARGS_CLOSE:
        return State.END

    return State.END


def generate_json(
    prompt: str,
    fn_name: str,
    fn_params: dict[str, str],
) -> dict[str, Any]:

    prompt_ids = my_model.encode(prompt)
    generated: list[int] = []                  

    state = State.START
    param_keys_remaining = list(fn_params.keys())
    curr_param = ""
    num_accumulated = ""
    extracted_args: dict[str, Any] = {}        

    print("Generating JSON:")

    while state != State.END:

        current_ids = prompt_ids + generated

        next_token = generate_token(
            current_ids, state, param_keys_remaining,
            fn_params, curr_param, num_accumulated
        )

        token_text = id_to_token.get(next_token, "")

        if state == State.PARAM_KEY:
            curr_param = token_text.strip('"')

        elif state == State.PARAM_VALUE_NUM:
            if token_text in [",", "}", " ", "\n"]:
                try:
                    extracted_args[curr_param] = float(num_accumulated)
                except ValueError:
                    extracted_args[curr_param] = 0.0
                num_accumulated = ""
                if curr_param in param_keys_remaining:
                    param_keys_remaining.remove(curr_param)
                state = update_state(
                    state, next_token, param_keys_remaining,
                    num_accumulated, curr_param, fn_params
                )
                print(token_text, end="")
                continue
            else:
                num_accumulated += token_text

        generated.append(next_token)
        print(token_text, end="")

        state = update_state(
            state, next_token, param_keys_remaining,
            num_accumulated, curr_param, fn_params
        )

    print()
    return {
        "fn_name": fn_name,
        "args": extracted_args
    }
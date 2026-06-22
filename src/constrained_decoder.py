import json
import numpy as np
from enum import Enum
from models import State
from llm_sdk import Small_LLM_Model


def token_of(text: str):

    if text not in token_to_id:
        raise ValueError(f"Token no encontrado: {text}")

    return token_to_id[text]


# function names(lo tengo en lista)

def get_valid_tokens(state: State):

    if state == State.START:

        return {
            token_of("{")
        }

    elif state == State.FN_NAME_KEY:

        return {
            token_of('"fn_name"')
        }

    elif state == State.FN_NAME_VALUE:

        valid = set()

        for fn in FUNCTION_NAMES:

            if fn in token_to_id:
                valid.add(token_of(fn))

        return valid

    elif state == State.ARGS_KEY:

        return {
            token_of('"args"')
        }

    elif state == State.END:

        return {
            token_of("}")
        }
    return set()


def generate_token(input_ids, state):

    logits = my_model.get_logits_from_input_ids(input_ids)

    # por si devuelve [seq_len, vocab_size]

    if len(logits.shape) == 2:
        logits = logits[-1]

    valid_tokens = get_valid_tokens(state)

    for token_id in range(len(logits)):

        if token_id not in valid_tokens:
            logits[token_id] = -np.inf

    next_token = int(np.argmax(logits))

    return next_token

# pasar de un estado al siguiente

def update_state(state, token):

    token_text = id_to_token[token]

    if state == State.START:
        return State.FN_NAME_KEY

    elif state == State.FN_NAME_KEY:
        return State.FN_NAME_VALUE

    elif state == State.FN_NAME_VALUE:
        return State.ARGS_KEY

    elif state == State.ARGS_KEY:
        return State.END

    return State.END


def generate_json(prompt):

    prompt_ids = my_model.encode(prompt)

    generated = []

    state = State.START

    while state != State.END:

        current_ids = prompt_ids + generated

        next_token = generate_token(
            current_ids,
            state
        )

        generated.append(next_token)

        print(
            my_model.decode([next_token]),
            end=""
        )

        state = update_state(
            state,
            next_token
        )

    return my_model.decode(generated)


# EJEMPLO USO

prompt = "sum 4 plus 3"
result = generate_json(prompt)

print("\n\nRESULTADO:")
print(result)
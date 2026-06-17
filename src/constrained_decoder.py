import numpy as np
from llm_sdk import Small_LLM_Model

my_model = Small_LLM_Model()
prompt = ""

prompt_encoded = my_model.encode(prompt)

state = State.START

generated = []

ids = prompt_ids + generated




def getvalid_tokens(state: State):
    if state == State.START:
        return {token_of("{")}
    
    elif state == State.FN_NAME_KEY:
        return {token_of('"fn_name"')}

    elif state == State.FN_NAME_VALUE:
        return (token_of({list_functions.name}))
    


    elif state == State.
    elif state == State.
    elif state == State.
    elif state == State.
    elif state == State.
    







def generate_token(prompt_ids, state):

    logits = my_model.get_logits_from_input_ids(prompt_ids)

    valid = get_valid_tokens()

    for token_id in range(len(logits)):
        if token_id not in valid:
            logits[token_id] = -np.inf

    return int(np.argmax(logits))


def valid_token(state) -> bool:
    ...

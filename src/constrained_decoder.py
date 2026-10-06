import numpy as np
from llm_sdk import Small_LLM_Model


def get_next_token_logits(model: Small_LLM_Model, input_ids: list) -> list[int]:

    logits_arr = model.get_logits_from_input_ids(input_ids)

    return (np.array(logits_arr))


def apply_mask(logits: list, valid_ids: list) -> list[int]:

    masked = np.full(len(logits), -np.inf)

    for valid_id in valid_ids:
        masked[valid_id] = logits[valid_id]

    return masked


def select_best_token(masked: list) -> int:
    best_token = np.argmax(masked)
    return best_token
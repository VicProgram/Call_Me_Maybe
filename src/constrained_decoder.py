import numpy as np
from llm_sdk import Small_LLM_Model
from vocab import VocabIndex


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


class JSONGenerator:

    def __init__(self, model: Small_LLM_Model, vocab: VocabIndex):
        self.model = model
        self.vocab = vocab

    def select_function_name(self):
        ...

    def extract_number(self):
        ...

    def extract_string(self):
        ...

    def extract_boolean(self):
        ...


def load_generator() -> JSONGenerator:
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file(Small_LLM_Model)
    vocab = VocabIndex(vocab_path)
    return JSONGenerator(model, vocab)

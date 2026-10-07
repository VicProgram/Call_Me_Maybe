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

    def select_function_name(self, prompt_ids: list, function_list: list):

        scores = {}
        for fun_name in function_list:
            tensor_ids = self.model.encode(fun_name).flatten().tolist()
            fun_scores = get_next_token_logits(self.model, prompt_ids)
            scores[fun_name] = sum(fun_scores[token_id] for token_id in tensor_ids)

        return max(scores, key=scores.get)

    def extract_number(self, prompt_ids: list):

        digits = self.vocab.search_characters("0123456789")
        dot = self.vocab.search_characters(".")
        minus = self.vocab.search_characters("-")
        terminators = self.vocab.search_characters(" ,\n\t")
        valid_ids = digits + dot + minus + terminators
        tokens = []

        while True:
            logits = get_next_token_logits(self.model, prompt_ids)
            masked = apply_mask(logits, valid_ids)
            selected = select_best_token(masked)
            if selected in terminators:
                break
            tokens.append(selected)
            prompt_ids.append(selected)
        number_str = self.model.decode(tokens)

        return float(number_str)

    def extract_string(self):
        ...

    def extract_boolean(self):
        ...


def load_generator() -> JSONGenerator:
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    vocab = VocabIndex(vocab_path)
    return JSONGenerator(model, vocab)
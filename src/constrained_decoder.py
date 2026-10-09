import numpy as np
from llm_sdk import Small_LLM_Model
from .vocab import VocabIndex


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

        digits = self.vocab.search_characters("0123456789")
        dot = self.vocab.search_characters(".")
        minus = self.vocab.search_characters("-")
        self.number_valid_ids = digits + dot + minus
        self.number_terminators = self.vocab.search_characters(" ,\n\t")
        self.number_valid_ids = self.number_valid_ids + self.number_terminators

        self.string_valid_ids = []
        # valid_ids = self.number_valid_ids + terminators
        for token, token_id in self.vocab.token_to_id.items():
            if not any(char in token for char in "{}[]"):
                self.string_valid_ids.append(token_id)
        self.string_terminators = self.vocab.search_characters('"\n')
        self.string_valid_ids = self.string_valid_ids + self.string_terminators

        self.true_id = self.vocab.search_exact("true")
        self.false_id = self.vocab.search_exact("false")
        self.bool_valid_ids = self.true_id + self.false_id

    def select_function_name(self, prompt_ids: list, function_list: list):

        scores = {}
        for fun_name in function_list:
            tensor_ids = self.model.encode(fun_name.name).flatten().tolist()
            fun_scores = get_next_token_logits(self.model, prompt_ids)
            scores[fun_name.name] = sum(
                fun_scores[token_id] for token_id in tensor_ids
                )

        best_name = max(scores, key=scores.get)
        for fun in function_list:
            if fun.name == best_name:
                return fun

        return max(scores, key=scores.get)

    def extract_number(self, prompt_ids: list):
        # terminators = self.vocab.search_characters(" ,\n\t")
        tokens = []

        while True:
            logits = get_next_token_logits(self.model, prompt_ids)
            masked = apply_mask(logits, self.number_valid_ids)
            selected = select_best_token(masked)
            if selected in self.number_terminators:
                break
            tokens.append(selected)
            prompt_ids.append(selected)
        number_str = self.model.decode(tokens)

        return float(number_str)

    # region
    # def extract_string(self, prompt_ids: list):
    #     # terminators = self.vocab.search_characters('"\n')
    #     tokens = []

    #     while True:
    #         logits = get_next_token_logits(self.model, prompt_ids)
    #         masked = apply_mask(logits, self.string_valid_ids)
    #         selected = select_best_token(masked)

    #         if selected in self.string_terminators:
    #             break
    #         tokens.append(selected)
    #         prompt_ids.append(selected)
    #     valid_str = self.model.decode(tokens)
    #     return valid_str
    # endregion

    def extract_string(self, prompt_ids: list):
        print(f"string_terminators: {self.string_terminators}")
        print(f"string_valid_ids count: {len(self.string_valid_ids)}")
        tokens = []
    
        while True:
            logits = get_next_token_logits(self.model, prompt_ids)
            masked = apply_mask(logits, self.string_valid_ids)
            selected = select_best_token(masked)
            print(f"selected: {selected} ({self.model.decode([selected])}), in terminators: {selected in self.string_terminators}")
            
            if selected in self.string_terminators:
                break
            tokens.append(selected)
            prompt_ids.append(selected)

        valid_str = self.model.decode(tokens)
        return valid_str

    def extract_boolean(self, prompt_ids: list):

        bool_logits = get_next_token_logits(self.model, prompt_ids)
        masked = apply_mask(bool_logits, self.bool_valid_ids)
        selected = select_best_token(masked)
        return selected in self.true_id


def load_generator() -> JSONGenerator:
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    vocab = VocabIndex(vocab_path)
    return JSONGenerator(model, vocab)

import numpy as np
from llm_sdk import Small_LLM_Model
from .vocab import VocabIndex


def get_next_token_logits(model: Small_LLM_Model, input_ids: list) -> list[int]:
    """Get logits from the model for the next token.

    Args:
        model: The language model to use.
        input_ids: List of token IDs representing the current context.

    Returns:
        A NumPy array of logits for all tokens in the vocabulary.
    """
    logits_arr = model.get_logits_from_input_ids(input_ids)

    return (np.array(logits_arr))


def apply_mask(logits: list, valid_ids: list) -> list[int]:
    """Apply a mask to logits, setting invalid tokens to -inf.

    Args:
        logits: The original logits array.
        valid_ids: List of token IDs that are allowed.

    Returns:
        A masked logits array where invalid tokens have -inf.
    """
    masked = np.full(len(logits), -np.inf)

    for valid_id in valid_ids:
        masked[valid_id] = logits[valid_id]

    return masked


def select_best_token(masked: list) -> int:
    """Select the token with the highest logit.

    Args:
        masked: The masked logits array.

    Returns:
        The ID of the token with the highest logit.
    """
    best_token = np.argmax(masked)
    return best_token


class JSONGenerator:
    """Generator for JSON output using constrained decoding.

    Uses a finite state machine approach to generate valid JSON
    by restricting token selection at each step.

    Attributes:
        model: The language model to use.
        vocab: The vocabulary index for token lookups.
        number_valid_ids: Token IDs valid for number generation.
        number_terminators: Token IDs that terminate a number.
        string_valid_ids: Token IDs valid for string generation.
        string_terminators: Token IDs that terminate a string.
        bool_valid_ids: Token IDs valid for boolean generation.
    """

    def __init__(self, model: Small_LLM_Model, vocab: VocabIndex):
        """Initialize the generator with a model and vocabulary.

        Args:
            model: The language model to use.
            vocab: The vocabulary index for token lookups.
        """
        self.model = model
        self.vocab = vocab

        #numbers
        digits = self.vocab.search_characters("0123456789")
        dot = self.vocab.search_characters(".")
        minus = self.vocab.search_characters("-")
        self.number_valid_ids = digits + dot + minus
        self.number_terminators = self.vocab.search_characters(" ,\n\t")
        self.number_valid_ids = self.number_valid_ids + self.number_terminators

        #strings
        self.string_valid_ids = []
        for token, token_id in self.vocab.token_to_id.items():
            if not any(char in token for char in "{}[]"):
                self.string_valid_ids.append(token_id)
        self.string_terminators = self.vocab.search_characters('"\n')
        self.string_valid_ids = self.string_valid_ids + self.string_terminators

        #booleans
        self.true_id = self.vocab.search_exact("true")
        self.false_id = self.vocab.search_exact("false")
        self.bool_valid_ids = self.true_id + self.false_id

    def select_function_name(self, prompt_ids: list, function_list: list):
        """Select the most appropriate function using constrained decoding.

        Scores each function based on the sum of logits for its tokens
        and returns the function with the highest score.

        Args:
            prompt_ids: Token IDs of the prompt.
            function_list: List of FunctionDefinition objects.

        Returns:
            The FunctionDefinition with the highest score.
        """
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
        """Extract a number using constrained decoding.

        Generates digits, decimal point, and minus sign until
        a terminator token is found.

        Args:
            prompt_ids: Token IDs of the prompt.

        Returns:
            The extracted number as a float.
        """
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


    def extract_string(self, prompt_ids: list):
        """Extract a string using constrained decoding.

        Generates characters until a terminator token is found
        or the generated string ends with a quote or newline.

        Args:
            prompt_ids: Token IDs of the prompt.

        Returns:
            The extracted string.
        """
        tokens = []

        while True:
            logits = get_next_token_logits(self.model, prompt_ids)
            masked = apply_mask(logits, self.string_valid_ids)
            selected = select_best_token(masked)

            if selected in self.string_terminators:
                break
            tokens.append(selected)
            prompt_ids.append(selected)
            valid_str = self.model.decode(tokens)
            if valid_str.endswith('"') or valid_str.endswith('\n'):
                break        
        return valid_str

    def extract_boolean(self, prompt_ids: list):
        """Extract a boolean using constrained decoding.

        Generates either 'true' or 'false' based on the model's choice.

        Args:
            prompt_ids: Token IDs of the prompt.

        Returns:
            True if 'true' was selected, False otherwise.
        """

        bool_logits = get_next_token_logits(self.model, prompt_ids)
        masked = apply_mask(bool_logits, self.bool_valid_ids)
        selected = select_best_token(masked)
        return selected in self.true_id


def load_generator() -> JSONGenerator:
    """Load and initialize a JSONGenerator with a model and vocabulary.

    Returns:
        An initialized JSONGenerator instance.
    """
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    vocab = VocabIndex(vocab_path)
    return JSONGenerator(model, vocab)

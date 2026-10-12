import numpy as np
from llm_sdk import Small_LLM_Model
from .vocab import VocabIndex
from .models import JSONState
from .prompt_builder import fun_selection


def get_next_token_logits(
    model: Small_LLM_Model, input_ids: list
        ) -> list[int]:
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

        # numbers
        digits = self.vocab.search_characters("0123456789")
        dot = self.vocab.search_characters(".")
        minus = self.vocab.search_characters("-")
        self.number_valid_ids = digits + dot + minus
        self.number_terminators = self.vocab.search_characters(" ,\n\t")
        self.number_valid_ids = self.number_valid_ids + self.number_terminators

        # strings
        self.string_valid_ids = []
        for token, token_id in self.vocab.token_to_id.items():
            if not any(char in token for char in "{}[]"):
                self.string_valid_ids.append(token_id)
        self.string_terminators = self.vocab.search_characters('"\n')
        self.string_valid_ids = self.string_valid_ids + self.string_terminators

        # booleans
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
        logits = get_next_token_logits(self.model, prompt_ids)

        scores = {}
        for fun in function_list:
            name_ids = self.model.encode(fun.name).flatten().tolist()
            scores[fun.name] = sum(logits[token_id] for token_id in name_ids)

        best_name = max(scores, key=scores.get)
        for fun in function_list:
            if fun.name == best_name:
                return fun

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

    def generate_json(
            self,  prompt: str, fun_list: list
            ) -> dict:
        """Generate a JSON object using the finite state machine.

        Args:
            model: The language model.
            prompt: The user's prompt.
            function_list: List of available functions.

        Returns:
            A dictionary with the generated JSON structure.
        """

        state = JSONState.START
        prompt_ids = self.model.encode(
            fun_selection(prompt, fun_list)
            ).flatten().tolist()

        output_tokens = []
        selected_fun = self.select_function_name(prompt_ids, fun_list)
        param_names = list(selected_fun.parameters.keys())
        param_count = 0

        while state != JSONState.FINISHED:
            tokens, new_state = self.select_state(
                state, prompt_ids, selected_fun, param_names, param_count
                )

            for token in tokens:
                prompt_ids.append(token)
                output_tokens.append(token)
            if new_state == JSONState.NEXT_PARAM:
                param_count += 1
            state = new_state

        res_text = self.model.decode(output_tokens)
        return res_text

    def check_termination(self, buffer: str, fixed_text: str) -> bool:
        """Verifica si el buffer contiene el
        texto fijo que indica terminación."""
        return fixed_text in buffer

    def get_param_type(self, param_name: str, function_list: list) -> str:
        """Obtiene el tipo de un parámetro de la función seleccionada."""
        for fun in function_list:
            if param_name in fun.parameters:
                return fun.parameters[param_name].type
        return None

    def get_next_param_name(self, param_names: list, used_params: list) -> str:
        """Obtiene el nombre del siguiente parámetro a procesar."""
        for param in param_names:
            if param not in used_params:
                return param

        return None

    def select_state(
        self, state: JSONState, prompt_ids: list, selected_fun,
            param_names: list, param_count: int) -> tuple[int, JSONState]:

        """Select the next token based on the current state.

        Args:
            model: The language model.
            state: Current state of the FSM.
            prompt_ids: Current token IDs.
            function_list: Available functions (for FN_NAME state).
            param_names: Parameter names (for PARAM_KEY state).

        Returns:
            A tuple of (selected_token_id, new_state).
        """

        match state:
            case JSONState.START:
                fixed_txt = self.model.encode(
                    '{"prompt": "').flatten().tolist()
                selected = fixed_txt
                return selected, JSONState.PROMPT

            case JSONState.PROMPT:
                logits = get_next_token_logits(self.model, prompt_ids)
                masked = apply_mask(logits, self.string_valid_ids)
                selected = select_best_token(masked)

                if selected in self.string_terminators:
                    return [], JSONState.AFTER_PROMPT

                return [selected], JSONState.PROMPT

            case JSONState.AFTER_PROMPT:
                fixed_txt = self.model.encode(
                    '", "fn_name": "').flatten().tolist()
                selected = fixed_txt
                return selected, JSONState.FN_NAME

            case JSONState.FN_NAME:
                tokens = self.model.encode(
                    selected_fun.name
                    ).flatten().tolist()
                return tokens, JSONState.AFTER_NAME

            case JSONState.AFTER_NAME:
                if not param_names:
                    fixed_txt = self.model.encode(
                        '", "args": {}').flatten().tolist()
                    return fixed_txt, JSONState.END
                fixed_txt = self.model.encode(
                  '", "args": {').flatten().tolist()

                selected = fixed_txt

                return selected, JSONState.PARAM_KEY

            case JSONState.PARAM_KEY:
                param_name = param_names[param_count]
                param_type = selected_fun.parameters[param_name].type

                opening = '": "' if param_type == "string" else '": '

                fixed_txt = self.model.encode(
                    f'"{param_name}{opening}'
                ).flatten().tolist()
                return fixed_txt, JSONState.PARAM_VALUE

            case JSONState.PARAM_VALUE:
                param_name = param_names[param_count]
                param_type = selected_fun.parameters[param_name].type

                if param_type == "number":
                    valid_ids = self.number_valid_ids
                    terminators = self.number_terminators
                elif param_type == "string":
                    valid_ids = self.string_valid_ids
                    terminators = self.string_terminators
                else:
                    valid_ids = self.bool_valid_ids
                    terminators = []

                logits = get_next_token_logits(self.model, prompt_ids)
                masked = apply_mask(logits, valid_ids)
                selected = select_best_token(masked)

                if selected in terminators:
                    return [], JSONState.NEXT_PARAM

                if param_type not in ("number", "string"):
                    return [selected], JSONState.NEXT_PARAM

                return [selected], JSONState.PARAM_VALUE

            case JSONState.NEXT_PARAM:
                close = ""
                if param_count > 0:
                    prev = param_names[param_count - 1]
                    if selected_fun.parameters[prev].type == "string":
                        close = '"'

                if param_count < len(param_names):
                    nxt = param_names[param_count]
                    nxt_type = selected_fun.parameters[nxt].type
                    opening = '": "' if nxt_type == "string" else '": '
                    fixed_txt = self.model.encode(
                        f'{close}, "{nxt}{opening}'
                    ).flatten().tolist()
                    return fixed_txt, JSONState.PARAM_VALUE

                fixed_txt = self.model.encode(
                    f'{close}}}').flatten().tolist()
                return fixed_txt, JSONState.END

            case JSONState.END:
                fixed_txt = self.model.encode('}').flatten().tolist()
                selected = fixed_txt
                return selected, JSONState.FINISHED

            case JSONState.FINISHED:
                return None, JSONState.FINISHED


def force_tokens(model: Small_LLM_Model, fixed_text: str, logits: list):

    fixed_tokens = model.encode(fixed_text)
    masked = apply_mask(logits, fixed_tokens)
    selected = select_best_token(masked)

    return selected


def load_generator() -> JSONGenerator:
    """Load and initialize a JSONGenerator with a model and vocabulary.

    Returns:
        An initialized JSONGenerator instance.
    """
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    vocab = VocabIndex(vocab_path)
    return JSONGenerator(model, vocab)

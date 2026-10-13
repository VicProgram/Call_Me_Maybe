import json

import numpy as np
from llm_sdk import Small_LLM_Model
from .vocab import VocabIndex
from .models import JSONState, FunctionDefinition
from .prompt_builder import fun_selection


def get_next_token_logits(
    model: Small_LLM_Model, input_ids: list[int]
        ) -> list[float]:
    """Get logits from the model for the next token.

    Args:
        model: The language model to use.
        input_ids: List of token IDs representing the current context.

    Returns:
        A NumPy array of logits for all tokens in the vocabulary.
    """
    logits_arr = model.get_logits_from_input_ids(input_ids)

    return np.array(logits_arr)  # type: ignore[return-value]


def apply_mask(
    logits: list[float] | list[int], valid_ids: list[int]
) -> list[float]:
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

    return masked  # type: ignore[return-value]


def select_best_token(masked: list[float]) -> int:
    """Select the token with the highest logit.

    Args:
        masked: The masked logits array.

    Returns:
        The ID of the token with the highest logit.
    """
    best_token = np.argmax(masked)
    return best_token  # type: ignore[return-value]


def log_softmax(logits: list[float] | list[int]) -> list[float]:
    """Normalize raw logits into log-probabilities.

    Args:
        logits: The raw logits for the next token.

    Returns:
        An array with the log-probability of each token.
    """

    logits_arr = np.asarray(logits, dtype=np.float64)
    max_logit = np.max(logits_arr)
    shifted = logits_arr - max_logit
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    result = shifted - log_sum_exp
    probs: list[float] = result.tolist()
    return probs


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
            if not any(char in token for char in "{}[]\\"):
                self.string_valid_ids.append(token_id)
        self.string_terminators = self.vocab.search_any_character('"\n')
        self.string_valid_ids = self.string_valid_ids + self.string_terminators

        # booleans
        self.true_id = self.vocab.search_exact("true")
        self.false_id = self.vocab.search_exact("false")
        self.bool_valid_ids = self.true_id + self.false_id

    def select_function_name(
        self, prompt_ids: list[int], function_list: list[FunctionDefinition]
    ) -> FunctionDefinition | None:
        """Select the function whose name is the most probable.

        Every candidate name is scored token by token: each token
        is evaluated in the context of the prompt plus the name
        tokens generated so far. The function with the highest
        summed log-probability wins.

        Args:
            prompt_ids: Token IDs representing the prompt context.
            function_list: Available function definitions.

        Returns:
            The function with the highest score.
        """
        best_fun = None
        best_score = None

        for fun in function_list:
            ids = list(prompt_ids)
            score = 0.0

            for token_id in self.model.encode(fun.name).flatten().tolist():
                logits = get_next_token_logits(self.model, ids)
                score += float(log_softmax(logits)[token_id])
                ids.append(token_id)

            if best_score is None or score > best_score:
                best_fun = fun
                best_score = score

        return best_fun

    def generate_json(
            self, prompt: str, fun_list: list[FunctionDefinition]
            ) -> str:
        """Generate a JSON object using the finite state machine.

        Args:
            prompt: The user's natural language request.
            fun_list: List of available function definitions.

        Returns:
            The generated JSON text as a string.
        """

        prompt_ids = self.model.encode(
            fun_selection(prompt, fun_list)
            ).flatten().tolist()

        selected_fun = self.select_function_name(prompt_ids, fun_list)
        if selected_fun is None:
            raise ValueError("No valid function found")
        param_names = list(selected_fun.parameters.keys())
        param_count = 0
        opening = '{"prompt": ' + json.dumps(prompt) + ', "name": "'
        output_tokens = self.model.encode(opening).flatten().tolist()
        prompt_ids += output_tokens

        state = JSONState.FN_NAME

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
        return res_text  # type: ignore[no-any-return]

    def select_state(
        self, state: JSONState, prompt_ids: list[int],
            selected_fun: FunctionDefinition, param_names: list[str],
            param_count: int
            ) -> tuple[list[int], JSONState]:
        """Select the tokens to emit for the current state.

        Args:
            state: Current state of the FSM.
            prompt_ids: Current token IDs of the context.
            selected_fun: The function chosen for the call.
            param_names: Names of the parameters to extract.
            param_count: Index of the parameter being extracted.

        Returns:
            A tuple with the tokens to emit and the next state.
        """

        match state:
            case JSONState.FN_NAME:
                tokens = self.model.encode(
                    selected_fun.name
                    ).flatten().tolist()
                return tokens, JSONState.AFTER_NAME

            case JSONState.AFTER_NAME:
                if not param_names:
                    fixed_txt = self.model.encode(
                        '", "parameters": {}').flatten().tolist()
                    return fixed_txt, JSONState.END
                fixed_txt = self.model.encode(
                  '", "parameters": {').flatten().tolist()

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
                return [], JSONState.FINISHED

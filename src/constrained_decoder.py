import json
import re
from typing import Callable

import numpy as np
from llm_sdk import Small_LLM_Model
from .vocab import VocabIndex
from .models import JSONState, FunctionDefinition
from .prompt_builder import fun_select_prompt


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

    Raises:
        ValueError: If the mask does not allow any token.
    """
    if not np.isfinite(masked).any():
        raise ValueError("No valid token available for the current state")

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


# --- Arreglo 5: strings que respetan el JSON --------------------------------
# Un valor de tipo string se construye con "trozos". Un trozo es una secuencia
# que ya es válida dentro de un string JSON: caracteres normales (incluidas
# las llaves y corchetes, que no hay que escapar) o una secuencia de escape
# completa ("\"", "\\", "\n", "\u1234", ...). Todo lo demás (una comilla sin
# escapar, un salto de línea crudo o una barra invertida suelta) rompería el
# JSON y queda prohibido.
_STRING_CHUNK = re.compile(
    r'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*'
)


def _first_unescaped_quote(text: str) -> int:
    """Devuelve el índice de la primera comilla SIN escapar, o -1.

    Sirve para saber si el texto de un token cierra el string (comilla sin
    escapar) o si es contenido: por ejemplo ``\\"`` es una comilla escapada y
    por tanto NO cierra nada.

    Args:
        text: El texto que produce un token al decodificarlo.

    Returns:
        El índice de la primera comilla sin escapar, o -1 si no hay.
    """
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return index
    return -1


class JSONGenerator:
    """Generator for JSON output using constrained decoding.

    Uses a finite state machine approach to generate valid JSON
    by restricting token selection at each step.

    Attributes:
        model: The language model to use.
        vocab: The vocabulary index for token lookups.
        digit_ids: Token IDs composed only of digits.
        dot_id: Token ID of a single "." (decimal separator).
        minus_id: Token ID of a single "-" (sign).
        terminator_ids: Token IDs that terminate a numeric value.
        string_valid_ids: Token IDs valid for string generation.
        string_terminators: Token IDs that terminate a string.
        bool_valid_ids: Token IDs valid for boolean generation.
        value_generators: One value generator per supported parameter type.
    """

    def __init__(self, model: Small_LLM_Model, vocab: VocabIndex):
        """Initialize the generator with a model and vocabulary.

        Args:
            model: The language model to use.
            vocab: The vocabulary index for token lookups.
        """
        self.model = model
        self.vocab = vocab

        # atomic token groups shared by the numeric masks
        self.digit_ids = self.vocab.search_characters("0123456789")
        self.dot_id = self.vocab.search_exact(".")
        self.minus_id = self.vocab.search_exact("-")
        self.terminator_ids = self.vocab.search_characters(" ,\n\t")
        self.max_value_len = 512

        # strings: clasificamos cada token por el texto que produce al
        # decodificarlo (una sola vez, cuesta ~1 s sobre 151k tokens). Se
        # separan los tokens que son contenido válido de los que cierran el
        # string. (Arreglo 5: antes se prohibía por completo " \ { } [ ]".)
        content_ids: list[int] = []
        terminator_ids: list[int] = []
        for token_id in self.vocab.token_to_id.values():
            text = self.model.decode([token_id])
            cut = _first_unescaped_quote(text)
            if cut == -1:
                if _STRING_CHUNK.fullmatch(text):
                    content_ids.append(token_id)
            elif _STRING_CHUNK.fullmatch(text[:cut]):
                terminator_ids.append(token_id)
        self.string_terminators = terminator_ids
        # Los terminadores también deben poder elegirse; si no, el modelo no
        # tendría forma de cerrar el string.
        self.string_valid_ids = content_ids + terminator_ids

        # booleans
        self.bool_valid_ids = (
            self.vocab.search_exact("true") + self.vocab.search_exact("false")
        )

        # one value generator per supported parameter type
        self.value_generators: dict[
            str, Callable[[list[int]], list[int]]
        ] = {
            "string": self.generate_string,
            "boolean": self.generate_boolean,
            "number": lambda ids: self.generate_numeric(ids, integer=False),
            "integer": lambda ids: self.generate_numeric(ids, integer=True),
        }

        # fail fast if the vocabulary cannot build a non-empty mask
        if not self.digit_ids:
            raise ValueError("Empty mask: no valid digit tokens")
        if not self.dot_id or not self.minus_id:
            raise ValueError("Vocabulary lacks single '.'/'-' tokens")
        if not self.string_valid_ids:
            raise ValueError("Empty mask: no valid tokens for 'string'")
        if not self.bool_valid_ids:
            raise ValueError(
                "Vocabulary does not contain 'true'/'false' as single tokens"
            )

    def generate_string(self, prompt_ids: list[int]) -> list[int]:
        """Generate the tokens that make up a string value.

        Emits tokens until a string terminator is chosen. The terminator
        itself is not emitted, but the text glued in front of its closing
        quote is kept (arreglo 4). The closing quote is added later by the
        NEXT_PARAM state.

        Args:
            prompt_ids: Token IDs of the context before the value.

        Returns:
            The token IDs that make up the string content.

        Raises:
            ValueError: If the value exceeds the maximum length.
        """
        value_tokens: list[int] = []
        context = list(prompt_ids)

        while True:
            logits = get_next_token_logits(self.model, context)
            masked = apply_mask(logits, self.string_valid_ids)
            selected = select_best_token(masked)

            if selected in self.string_terminators:
                # Arreglo 4: el token que cierra el string puede traer texto
                # pegado delante (p. ej. ")" en el token ')"'). Antes se
                # tiraba el token entero y se perdía ese texto. Ahora nos
                # quedamos con lo anterior a la comilla y lo añadimos al
                # valor. Ese prefijo ya es contenido JSON válido (lo garantiza
                # la clasificación hecha en __init__).
                text = self.model.decode([selected])
                prefix = text[:_first_unescaped_quote(text)]
                if prefix:
                    value_tokens.extend(
                        self.model.encode(prefix).flatten().tolist()
                    )
                return value_tokens

            value_tokens.append(selected)
            context.append(selected)

            if len(value_tokens) > self.max_value_len:
                raise ValueError("String value exceeded the maximum length")

    def generate_boolean(self, prompt_ids: list[int]) -> list[int]:
        """Generate the single token of a boolean value.

        Args:
            prompt_ids: Token IDs of the context before the value.

        Returns:
            A one-element list with the chosen `true`/`false` token.
        """
        logits = get_next_token_logits(self.model, prompt_ids)
        masked = apply_mask(logits, self.bool_valid_ids)
        return [select_best_token(masked)]

    def generate_numeric(
        self, prompt_ids: list[int], integer: bool
    ) -> list[int]:
        """Generate the tokens that make up a numeric value.

        Runs a small sub-state machine so that only syntactically valid
        numbers can be produced: an optional leading minus, at least one
        digit and, for `number` only, a single dot followed by digits.
        The terminator itself is not emitted.

        Args:
            prompt_ids: Token IDs of the context before the value.
            integer: If True, no decimal point is allowed.

        Returns:
            The token IDs that make up the numeric value.

        Raises:
            ValueError: If the value exceeds the maximum length.
        """
        value_tokens: list[int] = []
        context = list(prompt_ids)
        state = "start"

        while True:
            allowed = self.numeric_allowed(state, integer)
            logits = get_next_token_logits(self.model, context)
            masked = apply_mask(logits, allowed)
            selected = select_best_token(masked)

            if selected in self.terminator_ids:
                return value_tokens

            value_tokens.append(selected)
            context.append(selected)
            state = self.numeric_next(state, selected)

            if len(value_tokens) > self.max_value_len:
                raise ValueError("Numeric value exceeded the maximum length")

    def numeric_allowed(self, state: str, integer: bool) -> list[int]:
        """Return the token IDs allowed by the numeric sub-state machine.

        Args:
            state: Current numeric state: "start", "sign", "int",
                "dot" or "frac".
            integer: If True, the decimal states are never reached.

        Returns:
            The list of token IDs allowed in this state.
        """
        if state == "start":
            return self.minus_id + self.digit_ids
        if state == "sign":
            return self.digit_ids
        if state == "int":
            allowed = self.digit_ids + self.terminator_ids
            return allowed if integer else allowed + self.dot_id
        if state == "dot":
            return self.digit_ids
        return self.digit_ids + self.terminator_ids

    def numeric_next(self, state: str, token_id: int) -> str:
        """Return the next numeric state after emitting a token.

        Args:
            state: Current numeric state.
            token_id: The token that was just emitted.

        Returns:
            The next state.
        """
        if state == "start":
            return "sign" if token_id in self.minus_id else "int"
        if state == "sign":
            return "int"
        if state == "int":
            return "dot" if token_id in self.dot_id else "int"
        if state == "dot":
            return "frac"
        return "frac"

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
            fun_select_prompt(prompt, fun_list)
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

                generator = self.value_generators.get(param_type)
                if generator is None:
                    raise ValueError(
                        f"Unsupported parameter type: {param_type!r}"
                    )

                return generator(prompt_ids), JSONState.NEXT_PARAM

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

import json
from llm_sdk import Small_LLM_Model
from .vocab import VocabIndex
from .constrained_decoder import JSONGenerator
from .models import FunctionDefinition, FunctionCall, ValueType


def value_matches(value: object, declared: ValueType) -> bool:
    """Check that a parsed value matches its declared type.

    ``bool`` is a subclass of ``int`` in Python, so it is explicitly
    excluded from the numeric types: a boolean must never pass as an
    integer or a number.

    Args:
        value: The value parsed from the generated JSON.
        declared: The type declared for the parameter.

    Returns:
        True if the value is valid for the declared type.
    """
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class FunctionCaller:
    """Orchestrator for the function calling process.

    Coordinates function selection and argument extraction
    using the constrained decoder.

    Attributes:
        model: The language model to use.
        vocab: The vocabulary index for token lookups.
        generator: The JSONGenerator for constrained decoding.
    """

    def __init__(self, model: Small_LLM_Model):
        """Initialize the FunctionCaller with a model.

        Args:
            model: The language model to use.
        """
        self.model = model
        vocab_path = model.get_path_to_vocab_file()
        self.vocab = VocabIndex(vocab_path)
        self.generator = JSONGenerator(model, self.vocab)

    def resolve(
            self, prompt: str, fun_list: list[FunctionDefinition]
            ) -> FunctionCall:
        """Resolve a prompt into a function call.

        Selects the most appropriate function and extracts all
        required arguments using constrained decoding.

        Args:
            prompt: The user's natural language request.
            fun_list: List of available FunctionDefinition objects.

        Returns:
            A FunctionCall object with the selected
            function and extracted arguments.

        Raises:
            ValueError: If the generated text is not valid JSON.
        """

        raw_json = self.generator.generate_json(prompt, fun_list)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido generado: {e} ({raw_json!r})")

        if not isinstance(data, dict):
            raise ValueError(
                f"El JSON generado no es un objeto: {raw_json!r}"
            )

        name = data.get("name")
        parameters = data.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            raise ValueError(
                f"El JSON generado tiene una forma inesperada: {raw_json!r}"
            )

        selected = next((f for f in fun_list if f.name == name), None)
        if selected is None:
            raise ValueError(f"Función generada no definida: {name!r}")

        # Arreglo 1: el subject (V.4.1) muestra {"a": 2.0, "b": 3.0} y su
        # corrector hace `assert isinstance(a, float)`. El decoder puede emitir
        # el literal entero (2), así que aquí coaccionamos "number" a float.
        # "integer" y "boolean" NO se tocan: `bool` es subclase de `int`, y el
        # tipo declarado manda.
        resolved: dict[str, object] = {}
        for param_name, value in parameters.items():
            definition = selected.parameters.get(param_name)
            if definition is None:
                raise ValueError(
                    f"Parámetro inesperado {param_name!r} en {name!r}"
                )
            if not value_matches(value, definition.type):
                raise ValueError(
                    f"El parámetro {param_name!r} de {name!r} no es un "
                    f"{definition.type} válido: {value!r}"
                )
            if definition.type == "number":
                value = float(value)
            resolved[param_name] = value

        return FunctionCall(
            prompt=prompt, name=name, parameters=resolved
            )

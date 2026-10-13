import json
from llm_sdk import Small_LLM_Model
from .vocab import VocabIndex
from .constrained_decoder import JSONGenerator
from .models import FunctionDefinition
from .models import FunctionCall


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

        return FunctionCall(
            prompt=prompt, name=data["name"],
            parameters=data["parameters"]
            )

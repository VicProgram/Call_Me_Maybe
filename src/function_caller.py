from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCall
from src.constrained_decoder import generate_json, FUNCTION_NAMES
from src.prompt_builder import build_function_selection_prompt


class FunctionCaller:

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model

    def resolve(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> FunctionCall:

        FUNCTION_NAMES.clear()
        FUNCTION_NAMES.extend(fn.name for fn in functions)

        selection_prompt = build_function_selection_prompt(prompt, functions)

        result = generate_json(selection_prompt, functions)

        return FunctionCall(
            prompt=prompt,
            fn_name=result["fn_name"],
            args=result["args"],
        )

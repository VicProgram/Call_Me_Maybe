from typing import Any
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCall
from src.vocabulary import VocabularyIndex
from src.constrained_decoder import JSONGenerator
from src.prompt_builder import (
    build_function_selection_prompt,
    build_argument_prompt
)


class FunctionCaller:
    def __init__(self, model: Small_LLM_Model)-> None:
        self.model = model
        self.vocab = VocabularyIndex(model)
        self.generator = JSONGenerator(model, self.vocab)

    def resolve(
            self, prompt: str, functions: list[FunctionDefinition]
    ) -> FunctionCall:
    
        selection_prompt = build_function_selection_prompt(prompt, functions)
        selection_ids = self.model.encode(selection_prompt)

        fn_names = [fn.name for fn in functions]

        chosen_name = self.generator.select_function_name(
            selection_ids, fn_names
        )

        fn_def = next(fn for fn in functions if fn.name == chosen_name)

        args: dict[str, Any] = {}

        for param_name, param_def in fn_def.parameters.items():
            arg_prompt = build_argument_prompt(prompt, fn_def, param_name, args)

            arg_ids = self.model.encode(arg_prompt)

            if param_def.type in ("number", "float"):
                args[param_name] = self.generator.extract_number(
                    arg_ids, param_name
                )
            elif param_def.type == "integer":
                value = self.generator.extract_number(arg_ids, param_name)
                args[param_name] = int(value)
            elif param_def.type == "boolean":
                args[param_name] = self.generator.extract_boolean(arg_ids)
            else:
                # Por defecto, string
                args[param_name] = self.generator.extract_string(
                    arg_ids, param_name
                )

            return FunctionCall(
                prompt=prompt,
                fn_name=chosen_name,
                args=args
            )


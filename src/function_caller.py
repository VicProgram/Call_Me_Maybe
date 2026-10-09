from llm_sdk import Small_LLM_Model
from .vocab import VocabIndex
from .constrained_decoder import JSONGenerator
from .models import FunctionDefinition
from .models import FunctionCall
from .prompt_builder import function_selection, arg_extract


class FunctionCaller:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        vocab_path = model.get_path_to_vocab_file()
        self.vocab = VocabIndex(vocab_path)
        self.generator = JSONGenerator(model, self.vocab)

    def resolve(self, prompt: str, functions_list: list[FunctionDefinition]) -> FunctionCall:

        prompt_txt = function_selection(prompt, functions_list)

        prompt_ids = self.model.encode(prompt_txt).flatten().tolist()

        sel_funct = self.generator.select_function_name(prompt_ids, functions_list)

        extracted_args = {}
        for param_name, param_def in sel_funct.parameters.items():

            arg_prompt = arg_extract(
                sel_funct, prompt, param_name, extracted_args)

            arg_prompt_ids = self.model.encode(arg_prompt).flatten().tolist()

            if param_def.type == "number":
                value = self.generator.extract_number(arg_prompt_ids)
            elif param_def.type == "string":
                value = self.generator.extract_string(arg_prompt_ids)
            elif param_def.type == "bool":
                value = self.generator.extract_boolean(arg_prompt_ids)
            else:
                value = None

            extracted_args[param_name] = value

        return FunctionCall(
            prompt=prompt,
            fn_name=sel_funct.name,
            args=extracted_args
        )
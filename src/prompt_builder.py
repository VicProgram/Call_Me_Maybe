# Importamos FunctionDefinition del módulo models dentro de src.
from .models import FunctionDefinition
from typing import Any


def function_selection(
        user_prompt: str, function_list: list[FunctionDefinition]
        ) -> str:

    formatted_functions = []

    for function in function_list:
        line = f"- {function.name}: {function.description}"
        formatted_functions.append(line)

    functions_text = "\n".join(formatted_functions)

    main_prompt = (
        "You are a function caller assistant\n"
        "Select the most appropriate function for the user's request.\n"
        " The available functions are: "
        f"{functions_text}\n"
        f"{user_prompt}\n"
        "Function to call: "
    )

    return main_prompt


# def arg_extract(
#         selected_function: FunctionDefinition, user_prompt: str,
#         target_param: str, extracted_args: dict
#                 ) -> dict[str, Any]:

#     param_def = selected_function.parameters[target_param]

#     main_prompt = (
#         f"Function '{selected_function.name}' - '{selected_function.description}'\n"
#         f"Extract parameter '{target_param}'- (type: {param_def.type})\n"
#         f"Already extracted arguments: {extracted_args}\n"
#         f"User request: {user_prompt}\n"
#         "Value: "
#     )

#     return main_prompt



def arg_extract(selected_function: FunctionDefinition, target_param: str, user_prompt, extracted_args) -> str:

    # param_def = selected_function.parameters[target_param]

    main_prompt = (
        f"Function: {selected_function.name} - {selected_function.description}\n"
        f"Extract parameter '{target_param}':\n"
        f"Value: "
    )
    return main_prompt



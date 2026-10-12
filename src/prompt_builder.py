# Importamos FunctionDefinition del módulo models dentro de src.
from .models import FunctionDefinition
from typing import Any


def function_selection(
        user_prompt: str, function_list: list[FunctionDefinition]
        ) -> str:
    """Build a prompt for function selection.

    Creates a formatted prompt that instructs the LLM to select
    the most appropriate function for the user's request.

    Args:
        user_prompt: The user's natural language request.
        function_list: List of available FunctionDefinition objects.

    Returns:
        A formatted prompt string for function selection.
    """
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


def arg_extract(
        selected_function: FunctionDefinition, user_prompt: str,
        target_param: str, extracted_args: dict
                ) -> dict[str, Any]:
    """Build a prompt for argument extraction.

    Creates a formatted prompt that instructs the LLM to extract
    the value of a specific parameter from the user's request.

    Args:
        selected_function: The function chosen for extraction.
        user_prompt: The user's natural language request.
        target_param: The name of the parameter to extract.
        extracted_args: Dictionary of already extracted arguments.

    Returns:
        A formatted prompt string for argument extraction.
    """
    param_def = selected_function.parameters[target_param]

    main_prompt = (
        f"Function '{selected_function.name}' - '{selected_function.description}'\n"
        f"Extract parameter '{target_param}'- (type: {param_def.type})\n"
        f"Already extracted arguments: {extracted_args}\n"
        f"User request: {user_prompt}\n"
        "Value: "
    )

    return main_prompt



# def arg_extract(selected_function: FunctionDefinition, target_param: str, user_prompt, extracted_args) -> str:

#     # param_def = selected_function.parameters[target_param]

#     main_prompt = (
#         f"Function: {selected_function.name} - {selected_function.description}\n"
#         f"Extract parameter '{target_param}':\n"
#         f"Value: "
#     )
#     return main_prompt


# probar estos :  f"Extract the value of '{target_param}' from: {user_prompt}\n"
#        f"Value: "
    

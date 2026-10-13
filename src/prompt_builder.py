from .models import FunctionDefinition


def fun_selection(
        user_prompt: str, fun_list: list[FunctionDefinition]
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

    for function in fun_list:
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

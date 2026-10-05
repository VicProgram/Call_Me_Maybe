from models import FunctionDefinition


def function_selection(
        user_prompt: str, function_list: list[FunctionDefinition]
        ):

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
        selected_function: FunctionDefinition, user_prompt : str,

                ):

    params_text = ""

    for param_name, param_def in selected_function.parameters.items():
        params_text += f"- {param_name} ({param_def.type}): {param_def.description}\n"

    main_prompt = (
        f"Extract the arguments for the function '{selected_function.name}' "
        f"from the user request.\n"
        f"Required parameters:\n{params_text}\n"
        f"User request: {user_prompt}\n"
        "Value of params: "
    )

    return main_prompt
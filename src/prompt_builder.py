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
        selected_function: FunctionDefinition, user_prompt: str,
        target_param: str, extracted_args: dict
                ):

    param_def = selected_function.parameters[target_param]

    main_prompt = (
        f"Function '{selected_function.name}' - "
        f"'{selected_function.description}'\n"
        f"Extract parameter '{target_param}' (type: {param_def.type})\n"
        f"Already extracted arguments: {extracted_args}\n"
        f"User request: {user_prompt}\n"
        "Value: "
    )

    return main_prompt

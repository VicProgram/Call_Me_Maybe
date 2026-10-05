from models import FunctionDefinition


def function_selection(
        user_prompt: str, function_list: list[FunctionDefinition]
        ):

    formated_function = []

    for function in function_list:
        line = f"- {function.name}: {function.description}"
        formated_function.appen(line)

    functions_text = "\n".join(formated_function)

    main_prompt = (
        "You are a function caller assistant\n"
        "Select the most appropriate function for the user's request.\n"
        " The available functions are: "
        f"{functions_text}\n"
        f"{user_prompt}\n"
                  )

    return main_prompt


def arg_extract():
    ...

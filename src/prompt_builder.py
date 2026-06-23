from src.models import FunctionDefinition


def build_function_selection_prompt(
        user_prompt: str, functions: list[FunctionDefinition]
) -> str:

    fn_list = "\n".join(
        f"- {fn.name}: {fn.description}"
        for fn in functions
    )

    return(
        f"Available functions:\n"
        f"{fn_list}\n\n"
        f"User request: {user_prompt}\n\n"
        f"Output ONLY the function call in the format: function_name(arg1, arg2, ...)\n"
        f"Function call: "
    )

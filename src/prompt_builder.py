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
        f"Function to call: "
    )


def build_arg_prompt(
    user_prompt: str,
    fn_def: FunctionDefinition,
    param_name: str,
    already_extracted: dict[str, object]    
) -> str:
    
    param_type = fn_def.params[param_name].type

    lines = [
        f"Function: {fn_def.name} - {fn_def.description}",
        f"User request: {user_prompt}"
    ]

    for prev_name, prev_val in already_extracted.items():
        prev_type = fn_def.params[prev_name].type
        lines.append(f"Parameter '{prev_name}' ({prev_type}): {prev_val}")

    lines.append(f"\nExtract parameter '{param_name}' (type: {param_type}):")
    lines.append("Value: ")

    return "\n".join(lines)


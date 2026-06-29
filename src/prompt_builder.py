from src.models import FunctionDefinition


def build_function_selection_prompt(
        user_prompt: str, functions: list[FunctionDefinition]
) -> str:

    fn_list = "\n".join(
        f"- {fn.name}: {fn.description}"
        for fn in functions
    )

    return (
        f"You are a function calling assistant. "
        f"Select the most appropriate function.\n\n"
        f"Available functions:\n{fn_list}\n\n"
        f"User request: {user_prompt}\n\n"
        f"Function to call: "
    )


def build_argument_extraction_prompt(
    user_prompt: str,
    fn_def: FunctionDefinition,
    param_name: str,
    already_extracted: dict[str, object]
) -> str:

    param_type = fn_def.parameters[param_name].type

    context_parts = [
        f"Function: {fn_def.name} - {fn_def.description}",
        f"User request: {user_prompt}",
    ]

    for prev_name, prev_val in already_extracted.items():
        context_parts.append(
            f"Parameter '{prev_name}' ({fn_def.parameters[prev_name].type}): "
            f"already extracted = {prev_val}"
        )

    context_parts.append(
        f"\nExtract parameter '{param_name}' (type: {param_type}):\n"
        f"Value: "
    )

    return "\n".join(context_parts)

import re
from typing import Literal


def infer_template_format(template: str) -> Literal["f-string", "jinja"]:
    """
    Infer the format of string template used in the given string.

    Args:
        template: The string template to analyze.

    Returns:
        A literal type annotation indicating the format of the template: either 'f-string' or 'jinja'.
    """
    if "{" in template and "}" in template:
        # The template contains curly braces, indicating an f-string
        return "f-string"
    elif "{%" in template or "{{" in template:
        # The template contains Jinja-specific syntax, indicating a Jinja template
        return "jinja"
    else:
        # The template does not match either type, default to f-string
        return "f-string"


def extract_input_variables(template: str, template_format: str) -> list[str]:
    """
    Extract the input variables used in the given string template.

    Args:
        template: The string template to analyze.
        template_type: The type of template used, either 'f-string' or 'jinja'.

    Returns:
        A list of unique input variable names used in the template, excluding local variables created inside loops.
    """

    if template_format == "f-string":
        pattern = r"{(.*?)}"
        matches = re.findall(pattern, template)
    elif template_format == "jinja":
        # Extract variables enclosed in double curly braces
        pattern = r"{{(.*?)}}"
        matches = re.findall(pattern, template)

        # Extract variables used in for loops (excluding loop variables)
        for_loop_pattern = r"{% for (.*?) in (.*?) %}"
        for_loop_matches = re.findall(for_loop_pattern, template)

        for loop_var, iterable_var in for_loop_matches:
            if iterable_var not in matches:
                matches.append(iterable_var)

    else:
        raise ValueError(f"Unsupported template type: {template_format}")

    # Remove loop variables
    loop_variables = (
        {match[0] for match in for_loop_matches}
        if template_format == "jinja"
        else set()
    )

    # Extract variable names, excluding those created in for loops
    input_variables = set()
    for match in matches:
        match = match.strip()
        if match not in loop_variables:
            input_variables.add(match)

    return sorted(list(input_variables))

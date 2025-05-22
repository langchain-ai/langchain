"""A tool for validating inputs to model chats."""

import re

# Raw delimiter patterns (without safe-escape support)
delimiters = [
    r"\[INST\]", r"\[/INST\]", r"\<\<SYS\>\>", r"\<\<\/SYS\>\>",
    r"\<\!\-\-.*?\-\-\>", r"\<\!\-\-.*?\-\-\>"
]

# Escape-aware regex (allows [%INST%] and [/%INST%], and <%<%SYS%>%> and <%<%/SYS%>%>)
escape_safe_delimiters = [
    r"\[\%?INST\%?\]", r"\[\%?/INST\%?\]", r"\<\%?\<\%?SYS\%?\>\%?\>", r"\<\%?\<\%?/SYS\%?\>\%?\>",
    r"\<\%?\!\-\-.*?\-\-\%?\>", r"\<\%?\!\-\-.*?\-\-\%?\>"
]

# Strict patterns that do *not* allow any escape sequences
strict_delimiters = [
    r"(?<!%)\[INST\](?!%)", r"(?<!%)\[/INST\](?!%)", r"(?<!%)\<\<SYS\>\>(?!%)", r"(?<!%)\<\<\/SYS\>\>(?!%)",
    r"(?<!%)\<\!\-\-.*?\-\-\>(?!%)", r"(?<!%)\<\!\-\-.*?\-\-\>(?!%)"
]

def sanitize_input(input_text: str) -> str:
    """Sanitize input for chat by removing any delimiters to prevent escape of context."""
    # Create a regex pattern that matches any of the delimiters
    pattern = re.compile("|".join(strict_delimiters), re.DOTALL)
    # Remove the delimiters from the input text
    sanitized_text = re.sub(pattern, "", input_text)
    return sanitized_text

def validate_input(input_text: str) -> bool:
    """Validate input for chat by checking for delimiters."""
    # Create a regex pattern that matches any of the delimiters
    pattern = re.compile("|".join(strict_delimiters), re.DOTALL)
    return not bool(pattern.search(input_text))

def normalize_escaped_delimiters(input_text: str) -> str:
    """
    Conver safe-escaped delimiters back to usable format.
    For example: [%INST%] -> [INST]
    """
    escape_clean_delimiters = re.compile(r"\[\%?(\/?INST)\%?\]|\<\%?\<\%?(\/?SYS)\%?\>\%?\>|\<\%?\!\-\-(.*?)\-\-\%?\>")
    # Replace the escape sequences with their normalized versions
    return re.sub(escape_clean_delimiters, replacer, input_text)

def replacer(match):
        if match.group(1) is not None:
            return f"[{match.group(1)}]"
        elif match.group(2) is not None:
            return f"<<{match.group(2)}>>"
        elif match.group(3) is not None:
            return f"<!--{match.group(3)}-->"
        return match.group(0)  # Return the original match if no group is found
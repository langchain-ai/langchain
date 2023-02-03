import re


def normalize_boolean_output(input_string, true_values=["1"], false_values=["0"]):
    """Outputs a boolean from a string. Allows a LLM's response to be parsed into a boolean. For example, if a LLM returns "1", this function will return True. Likewise if an LLM returns "The answer is: \n1\n", this function will also return True.

    If value errors are common try changing the true and false values to rare characters so that it is unlikely the response could contain the character unless that was the 'intention' (insofar as that makes epistemological sense to say for a non-agential program) of the LLM.

    Args:
        input_string (str): The string to be parsed into a boolean.
        true_values (list, optional): A list of strings that should be parsed as True. Defaults to ["1"].
        false_values (list, optional): A list of strings that should be parsed as False. Defaults to ["0"].

    Raises:
        ValueError: If the input string is not a valid boolean.

    Returns:
        bool: The boolean value of the input string.
    """
    if any([true_value in false_values for true_value in true_values]):
        raise ValueError(
            "The true values and false values lists contain the same value."
        )
    input_string = re.sub(
        r"[^" + "".join(true_values + false_values) + "]", "", input_string
    )
    if input_string == "":
        raise ValueError(
            "The input string contains neither true nor false characters and is therefore not a valid boolean."
        )
    # if the string has both true and false values, raise a value error
    if any([true_value in input_string for true_value in true_values]) and any(
        [false_value in input_string for false_value in false_values]
    ):
        raise ValueError(
            "The input string contains both true and false characters and therefore is not a valid boolean."
        )
    return input_string in true_values

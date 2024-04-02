import os
from enum import Enum

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


class ExpectationType(str, Enum):
    prompts = "prompts"
    completions = "completions"


def read_expectation_from_file(
    expectation_type: ExpectationType, scenario_name: str
) -> str:
    """
    Returns an expected prompt or completion from a given scenario name.
    Expectations are stored as .txt files make it as easy as possible to read.
    """
    with open(
        os.path.join(DATA_DIR, expectation_type.value, f"{scenario_name}.txt"), "r"
    ) as f:
        content = f.read()

    # Remove a single trailing new line, if present, to aid authoring the txt file.
    if content.endswith("\n"):
        content = content[: -len("\n")]
    return content

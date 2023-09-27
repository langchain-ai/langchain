from langchain.prompts.prompt import PromptTemplate

DEFAULT_INPUT_KEY = "example"
DEFAULT_PROMPT = PromptTemplate(
    input_variables=[DEFAULT_INPUT_KEY], template="{example}"
)

SYNTHETIC_FEW_SHOT_PREFIX = (
    "This is a test about generating synthetic data about {subject}. Examples below:"
)
SYNTHETIC_FEW_SHOT_SUFFIX = (
    """Now you generate synthetic data about {subject}. Make sure to {extra}:"""
)

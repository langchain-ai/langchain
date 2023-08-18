from libs.langchain.langchain.prompts.prompt import PromptTemplate

DEFAULT_PROMPT = PromptTemplate(
    input_variables=["example"], template="{example}"
)

SYNTHETIC_FEW_SHOT_PREFIX = "This is a test about generating synthetic data about {subject}. Examples below:"
SYNTHETIC_FEW_SHOT_SUFFIX = """Now you generate synthetic data about {subject}. Make sure that {extra}:"""

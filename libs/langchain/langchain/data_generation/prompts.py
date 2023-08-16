from libs.langchain.langchain.prompts.prompt import PromptTemplate

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["example"], template="example: {example}"
)

SYNTHETIC_FEW_SHOT_PREFIX = (
    "This is a test about generating synthetic data about {subject}. Examples below:"
)
SYNTHETIC_FEW_SHOT_SUFFIX = "Now you try to generate synthetic data about {subject}:"

import os

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField

# For a description of each inference parameter, see
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
_model_kwargs = {
    "temperature": float(os.getenv("BEDROCK_JCVD_TEMPERATURE", "0.1")),
    "top_p": float(os.getenv("BEDROCK_JCVD_TOP_P", "1")),
    "top_k": int(os.getenv("BEDROCK_JCVD_TOP_K", "250")),
    "max_tokens_to_sample": int(os.getenv("BEDROCK_JCVD_MAX_TOKENS_TO_SAMPLE", "300")),
}

# Full list of base model IDs is available at
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
_model_alts = {
    "claude_2_1": ChatBedrock(
        model_id="anthropic.claude-v2:1", model_kwargs=_model_kwargs
    ),
    "claude_1": ChatBedrock(model_id="anthropic.claude-v1", model_kwargs=_model_kwargs),
    "claude_instant_1": ChatBedrock(
        model_id="anthropic.claude-instant-v1", model_kwargs=_model_kwargs
    ),
}

# For some tips on how to construct effective prompts for Claude,
# check out Anthropic's Claude Prompt Engineering deck (Bedrock edition)
# https://docs.google.com/presentation/d/1tjvAebcEyR8la3EmVwvjC7PHR8gfSrcsGKfTPAaManw
_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "You are JCVD. {input}"),
    ]
)

_model = ChatBedrock(
    model_id="anthropic.claude-v2", model_kwargs=_model_kwargs
).configurable_alternatives(
    which=ConfigurableField(
        id="model", name="Model", description="The model that will be used"
    ),
    default_key="claude_2",
    **_model_alts,
)

chain = _prompt | _model

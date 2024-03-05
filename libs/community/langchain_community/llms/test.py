from bedrock import Bedrock

llm = Bedrock(
    credentials_profile_name="unabened+sqlagent-Admin",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"max_tokens": 4096},
)

test=llm.invoke(
    "What are some theories about the relationship between unemployment and inflation?"
)

print(test)


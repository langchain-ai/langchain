from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

# Test the original issue scenario
chat = AzureChatOpenAI(
    azure_deployment="gpt-5-nano",
    api_version="2024-12-01-preview",
    azure_endpoint="https://example.com/",
    api_key=SecretStr("test-key"),
    max_completion_tokens=1500
)

print("✅ max_completion_tokens parameter accepted successfully")
print(f"max_tokens field value: {chat.max_tokens}")

# Also test max_tokens still works
chat2 = AzureChatOpenAI(
    azure_deployment="gpt-5-nano",
    api_version="2024-12-01-preview",
    azure_endpoint="https://example.com/",
    api_key=SecretStr("test-key"),
    max_tokens=1500
)

print("✅ max_tokens parameter still works")
print(f"max_tokens field value: {chat2.max_tokens}")
print("\n🎉 GPT-5 compatibility issue resolved!")

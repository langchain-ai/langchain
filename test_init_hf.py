import sys, os
sys.path.append(os.path.abspath("."))  # ensures local package can be imported

from libs.langchain.langchain_classic.chat_models import init_chat_model

# Test initializing a Hugging Face chat model
llm = init_chat_model(
    model="microsoft/Phi-3-mini-4k-instruct",  # <- model name (string)
    model_provider="huggingface",              # <- provider name
    temperature=0,                             # <- not a string
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)

print(llm)

from langchain.chains import create_tagging_chain
from langchain_experimental.llms.anthropic_functions import AnthropicFunctions

model = AnthropicFunctions(model='claude-2')

schema = {
    "properties": {
        "sentiment": {"type": "string"},
        "aggressiveness": {"type": "integer"},
        "language": {"type": "string"},
    }
}

# This is LLMChain, which implements invoke 
chain = create_tagging_chain(schema, model)
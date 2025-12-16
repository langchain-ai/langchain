from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
def test_token_counter_handles_function_call():


    llm = ChatOpenAI(model="gpt-4o", use_responses_api=True)

    messages = [
        HumanMessage(
            content=[
                {
                    "type": "function_call",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}',
                    },
                }
            ]
        )
    ]

    tokens = llm.get_num_tokens_from_messages(messages)
    assert tokens > 0

from langchain import hub


def test_agent_prompt():
    prompt = hub.pull("zhipuai-all-tools-chat/zhipuai-all-tools-agent")
    assert prompt is not None


def test_chat_prompt():
    prompt = hub.pull("zhipuai-all-tools-chat/zhipuai-all-tools-chat")
    assert prompt is not None

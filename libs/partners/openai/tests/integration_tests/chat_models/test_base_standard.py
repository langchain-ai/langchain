"""Standard LangChain interface tests"""

from pathlib import Path
from typing import Dict, List, Literal, Type, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_openai import ChatOpenAI

REPO_ROOT_DIR = Path(__file__).parents[6]


class TestOpenAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4o-mini", "stream_usage": True}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supported_usage_metadata_details(
        self,
    ) -> Dict[
        Literal["invoke", "stream"],
        List[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        return {"invoke": ["reasoning_output", "cache_read_input"], "stream": []}

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        with open(REPO_ROOT_DIR / "README.md", "r") as f:
            readme = f.read()

        input_ = f"""What's langchain? Here's the langchain README:
        
        {readme}
        """
        llm = ChatOpenAI(model="gpt-4o-mini", stream_usage=True)
        _invoke(llm, input_, stream)
        # invoke twice so first invocation is cached
        return _invoke(llm, input_, stream)

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        llm = ChatOpenAI(model="o1-mini", stream_usage=True, temperature=1)
        input_ = (
            "explain  the relationship between the 2008/9 economic crisis and the "
            "startup ecosystem in the early 2010s"
        )
        return _invoke(llm, input_, stream)


def _invoke(llm: ChatOpenAI, input_: str, stream: bool) -> AIMessage:
    if stream:
        full = None
        for chunk in llm.stream(input_):
            full = full + chunk if full else chunk  # type: ignore[operator]
        return cast(AIMessage, full)
    else:
        return cast(AIMessage, llm.invoke(input_))

"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


class TestHuggingFaceEndpoint(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatHuggingFace

    @property
    def chat_model_params(self) -> dict:
        return {}

    @pytest.fixture
    def model(self) -> BaseChatModel:
        llm = HuggingFaceEndpoint(  # type: ignore[call-arg]
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        return self.chat_model_class(llm=llm)  # type: ignore[call-arg]

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_stream(self, model: BaseChatModel) -> None:
        super().test_stream(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    async def test_astream(self, model: BaseChatModel) -> None:
        await super().test_astream(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_usage_metadata(self, model: BaseChatModel) -> None:
        super().test_usage_metadata(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_tool_calling(self, model: BaseChatModel) -> None:
        super().test_tool_calling(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        await super().test_tool_calling_async(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
        super().test_bind_runnables_as_tools(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_structured_output_async(self, model: BaseChatModel) -> None:  # type: ignore[override]
        super().test_structured_output(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(
            model, my_adder_tool=my_adder_tool
        )

    @pytest.mark.xfail(reason=("Not implemented"))
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_structured_few_shot_examples(model, my_adder_tool=my_adder_tool)

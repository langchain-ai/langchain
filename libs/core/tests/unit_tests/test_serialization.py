from typing import Any

from langchain_core.documents.base import Document
from langchain_core.load.dump import dumps
from langchain_core.load.serializable import Serializable
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
from langchain_core.output_parsers.list import (
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
)
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.outputs.chat_generation import ChatGeneration, ChatGenerationChunk
from langchain_core.outputs.generation import Generation, GenerationChunk
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.few_shot import (
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.prompts.few_shot_with_templates import FewShotPromptWithTemplates
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.runnables.base import (
    RunnableBinding,
    RunnableBindingBase,
    RunnableEach,
    RunnableEachBase,
    RunnableMap,
    RunnableParallel,
    RunnableSequence,
)
from langchain_core.runnables.branch import RunnableBranch
from langchain_core.runnables.configurable import (
    RunnableConfigurableAlternatives,
    RunnableConfigurableFields,
)
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.passthrough import RunnableAssign, RunnablePassthrough
from langchain_core.runnables.retry import RunnableRetry
from langchain_core.runnables.router import RouterRunnable
from tests.unit_tests.fake.memory import ChatMessageHistory


def test_serialize_system_message(snapshot: Any) -> None:
    obj: Any = SystemMessage(content="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_system_message_chunk(snapshot: Any) -> None:
    obj: Any = SystemMessageChunk(content="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_ai_message(snapshot: Any) -> None:
    obj: Any = AIMessage(content="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_ai_message_chunk(snapshot: Any) -> None:
    obj: Any = AIMessageChunk(content="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_human_message(snapshot: Any) -> None:
    obj: Any = HumanMessage(content="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_human_message_chunk(snapshot: Any) -> None:
    obj: Any = HumanMessageChunk(content="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_chat_message(snapshot: Any) -> None:
    obj: Any = ChatMessage(content="", role="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_chat_message_chunk(snapshot: Any) -> None:
    obj: Any = ChatMessageChunk(content="", role="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_tool_message(snapshot: Any) -> None:
    obj: Any = ToolMessage(content="", tool_call_id="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_tool_message_chunk(snapshot: Any) -> None:
    obj: Any = ToolMessageChunk(content="", tool_call_id="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_base_message(snapshot: Any) -> None:
    obj: Any = BaseMessage(content="", type="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_base_message_chunk(snapshot: Any) -> None:
    obj: Any = BaseMessageChunk(content="", type="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_function_message(snapshot: Any) -> None:
    obj: Any = FunctionMessage(content="", name="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_function_message_chunk(snapshot: Any) -> None:
    obj: Any = FunctionMessageChunk(content="", name="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_configurable_alternatives(snapshot: Any) -> None:
    obj: Any = RunnableConfigurableAlternatives(
        default=RunnablePassthrough(), which=ConfigurableField(id=""), alternatives={}
    )
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_configurable_fields(snapshot: Any) -> None:
    obj: Any = RunnableConfigurableFields(default=RunnablePassthrough(), fields={})
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_branch(snapshot: Any) -> None:
    obj: Any = RunnableBranch(
        (RunnablePassthrough(), RunnablePassthrough()), RunnablePassthrough()
    )
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_retry(snapshot: Any) -> None:
    obj: Any = RunnableRetry(bound=RunnablePassthrough())
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_with_fallbacks(snapshot: Any) -> None:
    obj: Any = RunnableWithFallbacks(
        runnable=RunnablePassthrough(), fallbacks=(RunnablePassthrough(),)
    )
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_router_runnable(snapshot: Any) -> None:
    obj: Any = RouterRunnable({"": RunnablePassthrough()})
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_assign(snapshot: Any) -> None:
    obj: Any = RunnableAssign(mapper=RunnableParallel({}))
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_passthrough(snapshot: Any) -> None:
    obj: Any = RunnablePassthrough()
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_binding(snapshot: Any) -> None:
    obj: Any = RunnableBinding(bound=RunnablePassthrough())
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_binding_base(snapshot: Any) -> None:
    obj: Any = RunnableBindingBase(bound=RunnablePassthrough())
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_each(snapshot: Any) -> None:
    obj: Any = RunnableEach(bound=RunnablePassthrough())
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_each_base(snapshot: Any) -> None:
    obj: Any = RunnableEachBase(bound=RunnablePassthrough())
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_map(snapshot: Any) -> None:
    obj: Any = RunnableMap()
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_parallel(snapshot: Any) -> None:
    obj: Any = RunnableParallel()
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_sequence(snapshot: Any) -> None:
    obj: Any = RunnableSequence(first=RunnablePassthrough(), last=RunnablePassthrough())
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_runnable_with_message_history(snapshot: Any) -> None:
    def get_chat_history(session_id: str) -> ChatMessageHistory:
        return ChatMessageHistory()

    obj: Any = RunnableWithMessageHistory(RunnablePassthrough(), get_chat_history)
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_serializable(snapshot: Any) -> None:
    obj: Any = Serializable()
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_comma_separated_list_output_parser(snapshot: Any) -> None:
    obj: Any = CommaSeparatedListOutputParser()
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_markdown_list_output_parser(snapshot: Any) -> None:
    obj: Any = MarkdownListOutputParser()
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_numbered_list_output_parser(snapshot: Any) -> None:
    obj: Any = NumberedListOutputParser()
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_str_output_parser(snapshot: Any) -> None:
    obj: Any = StrOutputParser()
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_few_shot_prompt_with_templates(snapshot: Any) -> None:
    obj: Any = FewShotPromptWithTemplates(
        example_prompt=PromptTemplate.from_template(""),
        suffix=PromptTemplate.from_template(""),
        examples=[],
        input_variables=[],
    )
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_few_shot_chat_message_prompt_template(snapshot: Any) -> None:
    obj: Any = FewShotChatMessagePromptTemplate(
        example_prompt=HumanMessagePromptTemplate.from_template(""), examples=[]
    )
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_few_shot_prompt_template(snapshot: Any) -> None:
    obj: Any = FewShotPromptTemplate(
        example_prompt=PromptTemplate.from_template(""),
        suffix="",
        examples=[],
        input_variables=[],
    )
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_ai_message_prompt_template(snapshot: Any) -> None:
    obj: Any = AIMessagePromptTemplate.from_template("")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_chat_message_prompt_template(snapshot: Any) -> None:
    obj: Any = ChatMessagePromptTemplate.from_template("", role="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_chat_prompt_template(snapshot: Any) -> None:
    obj: Any = ChatPromptTemplate.from_template("", role="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_human_message_prompt_template(snapshot: Any) -> None:
    obj: Any = HumanMessagePromptTemplate.from_template("")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_messages_placeholder(snapshot: Any) -> None:
    obj: Any = MessagesPlaceholder(variable_name="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_system_message_prompt_template(snapshot: Any) -> None:
    obj: Any = SystemMessagePromptTemplate.from_template("")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_pipeline_prompt_template(snapshot: Any) -> None:
    obj: Any = PipelinePromptTemplate(
        pipeline_prompts=[], final_prompt=PromptTemplate.from_template("")
    )
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_prompt_template(snapshot: Any) -> None:
    obj: Any = PromptTemplate.from_template("")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_document(snapshot: Any) -> None:
    obj: Any = Document(page_content="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_generation(snapshot: Any) -> None:
    obj: Any = Generation(text="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_generation_chunk(snapshot: Any) -> None:
    obj: Any = GenerationChunk(text="")
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_chat_generation(snapshot: Any) -> None:
    obj: Any = ChatGeneration(message=AIMessage(content=""))
    assert dumps(obj, pretty=True) == snapshot


def test_serialize_chat_generation_chunk(snapshot: Any) -> None:
    obj: Any = ChatGenerationChunk(message=AIMessage(content=""))
    assert dumps(obj, pretty=True) == snapshot

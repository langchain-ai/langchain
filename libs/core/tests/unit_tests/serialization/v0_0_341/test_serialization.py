import json
from typing import Any

from langchain_core.documents.base import Document
from langchain_core.load.load import load
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

with open("tests/unit_tests/serialization/v0_0_341/snapshot.ambr") as f:
    SNAPSHOTS = f.read()
SNAPSHOT_MAP = {
    x.split("\n")[0][15:]: json.loads(x.split("'''")[1])
    for x in SNAPSHOTS.split("# name: ")
    if not x.startswith("#")
}


def load_snapshot(snake_case_class: str) -> str:
    return SNAPSHOT_MAP[snake_case_class]


def test_deserialize_system_message() -> None:
    snapshot = load_snapshot("system_message")
    obj: Any = SystemMessage(content="")
    assert load(snapshot) == obj


def test_deserialize_system_message_chunk() -> None:
    snapshot = load_snapshot("system_message_chunk")
    obj: Any = SystemMessageChunk(content="")
    assert load(snapshot) == obj


def test_deserialize_ai_message() -> None:
    snapshot = load_snapshot("ai_message")
    obj: Any = AIMessage(content="")
    assert load(snapshot) == obj


def test_deserialize_ai_message_chunk() -> None:
    snapshot = load_snapshot("ai_message_chunk")
    obj: Any = AIMessageChunk(content="")
    assert load(snapshot) == obj


def test_deserialize_human_message() -> None:
    snapshot = load_snapshot("human_message")
    obj: Any = HumanMessage(content="")
    assert load(snapshot) == obj


def test_deserialize_human_message_chunk() -> None:
    snapshot = load_snapshot("human_message_chunk")
    obj: Any = HumanMessageChunk(content="")
    assert load(snapshot) == obj


def test_deserialize_chat_message() -> None:
    snapshot = load_snapshot("chat_message")
    obj: Any = ChatMessage(content="", role="")
    assert load(snapshot) == obj


def test_deserialize_chat_message_chunk() -> None:
    snapshot = load_snapshot("chat_message_chunk")
    obj: Any = ChatMessageChunk(content="", role="")
    assert load(snapshot) == obj


def test_deserialize_tool_message() -> None:
    snapshot = load_snapshot("tool_message")
    obj: Any = ToolMessage(content="", tool_call_id="")
    assert load(snapshot) == obj


def test_deserialize_tool_message_chunk() -> None:
    snapshot = load_snapshot("tool_message_chunk")
    obj: Any = ToolMessageChunk(content="", tool_call_id="")
    assert load(snapshot) == obj


def test_deserialize_base_message() -> None:
    snapshot = load_snapshot("base_message")
    obj: Any = BaseMessage(content="", type="")
    assert load(snapshot) == obj


def test_deserialize_base_message_chunk() -> None:
    snapshot = load_snapshot("base_message_chunk")
    obj: Any = BaseMessageChunk(content="", type="")
    assert load(snapshot) == obj


def test_deserialize_function_message() -> None:
    snapshot = load_snapshot("function_message")
    obj: Any = FunctionMessage(content="", name="")
    assert load(snapshot) == obj


def test_deserialize_function_message_chunk() -> None:
    snapshot = load_snapshot("function_message_chunk")
    obj: Any = FunctionMessageChunk(content="", name="")
    assert load(snapshot) == obj


def test_deserialize_runnable_configurable_alternatives() -> None:
    snapshot = load_snapshot("runnable_configurable_alternatives")
    obj: Any = RunnableConfigurableAlternatives(
        default=RunnablePassthrough(), which=ConfigurableField(id=""), alternatives={}
    )
    assert load(snapshot) == obj


def test_deserialize_runnable_configurable_fields() -> None:
    snapshot = load_snapshot("runnable_configurable_fields")
    obj: Any = RunnableConfigurableFields(default=RunnablePassthrough(), fields={})
    assert load(snapshot) == obj


def test_deserialize_runnable_branch() -> None:
    snapshot = load_snapshot("runnable_branch")
    obj: Any = RunnableBranch(
        (RunnablePassthrough(), RunnablePassthrough()), RunnablePassthrough()
    )
    assert load(snapshot) == obj


def test_deserialize_runnable_retry() -> None:
    snapshot = load_snapshot("runnable_retry")
    obj: Any = RunnableRetry(bound=RunnablePassthrough())
    assert load(snapshot) == obj


def test_deserialize_runnable_with_fallbacks() -> None:
    snapshot = load_snapshot("runnable_with_fallbacks")
    obj: Any = RunnableWithFallbacks(
        runnable=RunnablePassthrough(), fallbacks=(RunnablePassthrough(),)
    )
    assert load(snapshot) == obj


def test_deserialize_router_runnable() -> None:
    snapshot = load_snapshot("router_runnable")
    obj: Any = RouterRunnable({"": RunnablePassthrough()})
    assert load(snapshot) == obj


def test_deserialize_runnable_assign() -> None:
    snapshot = load_snapshot("runnable_assign")
    obj: Any = RunnableAssign(mapper=RunnableParallel({}))
    assert load(snapshot) == obj


def test_deserialize_runnable_passthrough() -> None:
    snapshot = load_snapshot("runnable_passthrough")
    obj: Any = RunnablePassthrough()
    assert load(snapshot) == obj


def test_deserialize_runnable_binding() -> None:
    snapshot = load_snapshot("runnable_binding")
    obj: Any = RunnableBinding(bound=RunnablePassthrough())
    assert load(snapshot) == obj


def test_deserialize_runnable_binding_base() -> None:
    snapshot = load_snapshot("runnable_binding_base")
    obj: Any = RunnableBindingBase(bound=RunnablePassthrough())
    assert load(snapshot) == obj


def test_deserialize_runnable_each() -> None:
    snapshot = load_snapshot("runnable_each")
    obj: Any = RunnableEach(bound=RunnablePassthrough())
    assert load(snapshot) == obj


def test_deserialize_runnable_each_base() -> None:
    snapshot = load_snapshot("runnable_each_base")
    obj: Any = RunnableEachBase(bound=RunnablePassthrough())
    assert load(snapshot) == obj


def test_deserialize_runnable_map() -> None:
    snapshot = load_snapshot("runnable_map")
    obj: Any = RunnableMap()
    assert load(snapshot) == obj


def test_deserialize_runnable_parallel() -> None:
    snapshot = load_snapshot("runnable_parallel")
    obj: Any = RunnableParallel()
    assert load(snapshot) == obj


def test_deserialize_runnable_sequence() -> None:
    snapshot = load_snapshot("runnable_sequence")
    obj: Any = RunnableSequence(first=RunnablePassthrough(), last=RunnablePassthrough())
    assert load(snapshot) == obj


def test_deserialize_runnable_with_message_history() -> None:
    snapshot = load_snapshot("runnable_with_message_history")

    def get_chat_history(session_id: str) -> ChatMessageHistory:
        return ChatMessageHistory()

    obj: Any = RunnableWithMessageHistory(RunnablePassthrough(), get_chat_history)

    assert load(snapshot) == obj


def test_deserialize_serializable() -> None:
    snapshot = load_snapshot("serializable")
    obj = Serializable()
    assert load(snapshot) == obj


def test_deserialize_comma_separated_list_output_parser() -> None:
    snapshot = load_snapshot("comma_separated_list_output_parser")
    obj = CommaSeparatedListOutputParser()
    assert load(snapshot) == obj


def test_deserialize_markdown_list_output_parser() -> None:
    snapshot = load_snapshot("markdown_list_output_parser")
    obj = MarkdownListOutputParser()
    assert load(snapshot) == obj


def test_deserialize_numbered_list_output_parser() -> None:
    snapshot = load_snapshot("numbered_list_output_parser")
    obj = NumberedListOutputParser()
    assert load(snapshot) == obj


def test_deserialize_str_output_parser() -> None:
    snapshot = load_snapshot("str_output_parser")
    obj = StrOutputParser()
    assert load(snapshot) == obj


def test_deserialize_few_shot_prompt_with_templates() -> None:
    snapshot = load_snapshot("few_shot_prompt_with_templates")
    obj: Any = FewShotPromptWithTemplates(
        example_prompt=PromptTemplate.from_template(""),
        suffix=PromptTemplate.from_template(""),
        examples=[],
        input_variables=[],
    )
    assert load(snapshot) == obj


def test_deserialize_few_shot_chat_message_prompt_template() -> None:
    snapshot = load_snapshot("few_shot_chat_message_prompt_template")
    obj: Any = FewShotChatMessagePromptTemplate(
        example_prompt=HumanMessagePromptTemplate.from_template(""), examples=[]
    )
    assert load(snapshot) == obj


def test_deserialize_few_shot_prompt_template() -> None:
    snapshot = load_snapshot("few_shot_prompt_template")
    obj: Any = FewShotPromptTemplate(
        example_prompt=PromptTemplate.from_template(""),
        suffix="",
        examples=[],
        input_variables=[],
    )
    assert load(snapshot) == obj


def test_deserialize_ai_message_prompt_template() -> None:
    snapshot = load_snapshot("ai_message_prompt_template")
    obj: Any = AIMessagePromptTemplate.from_template("")
    assert load(snapshot) == obj


def test_deserialize_chat_message_prompt_template() -> None:
    snapshot = load_snapshot("chat_message_prompt_template")
    obj: Any = ChatMessagePromptTemplate.from_template("", role="")
    assert load(snapshot) == obj


def test_deserialize_chat_prompt_template() -> None:
    snapshot = load_snapshot("chat_prompt_template")
    obj: Any = ChatPromptTemplate.from_template("", role="")
    assert load(snapshot) == obj


def test_deserialize_human_message_prompt_template() -> None:
    snapshot = load_snapshot("human_message_prompt_template")
    obj: Any = HumanMessagePromptTemplate.from_template("")
    assert load(snapshot) == obj


def test_deserialize_messages_placeholder() -> None:
    snapshot = load_snapshot("messages_placeholder")
    obj: Any = MessagesPlaceholder(variable_name="")
    assert load(snapshot) == obj


def test_deserialize_system_message_prompt_template() -> None:
    snapshot = load_snapshot("system_message_prompt_template")
    obj: Any = SystemMessagePromptTemplate.from_template("")
    assert load(snapshot) == obj


def test_deserialize_pipeline_prompt_template() -> None:
    snapshot = load_snapshot("pipeline_prompt_template")
    obj: Any = PipelinePromptTemplate(
        pipeline_prompts=[], final_prompt=PromptTemplate.from_template("")
    )
    assert load(snapshot) == obj


def test_deserialize_prompt_template() -> None:
    snapshot = load_snapshot("prompt_template")
    obj: Any = PromptTemplate.from_template("")
    assert load(snapshot) == obj


def test_deserialize_document() -> None:
    snapshot = load_snapshot("document")
    obj: Any = Document(page_content="")
    assert load(snapshot) == obj


def test_deserialize_generation() -> None:
    snapshot = load_snapshot("generation")
    obj: Any = Generation(text="")
    assert load(snapshot) == obj


def test_deserialize_generation_chunk() -> None:
    snapshot = load_snapshot("generation_chunk")
    obj: Any = GenerationChunk(text="")
    assert load(snapshot) == obj


def test_deserialize_chat_generation() -> None:
    snapshot = load_snapshot("chat_generation")
    obj: Any = ChatGeneration(message=AIMessage(content=""))
    assert load(snapshot) == obj


def test_deserialize_chat_generation_chunk() -> None:
    snapshot = load_snapshot("chat_generation_chunk")
    obj: Any = ChatGenerationChunk(message=AIMessage(content=""))
    assert load(snapshot) == obj

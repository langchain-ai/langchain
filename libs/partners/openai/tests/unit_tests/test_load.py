from langchain_core.load import dumpd, dumps, load, loads
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence

from langchain_openai import ChatOpenAI, OpenAI

OPENAI_CHAT_TEST_MODEL = "gpt-5.5"


def test_loads_openai_llm() -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello", top_p=0.8)  # type: ignore[call-arg]
    llm_string = dumps(llm)
    llm2 = loads(
        llm_string,
        secrets_map={"OPENAI_API_KEY": "hello"},
        allowed_objects=[OpenAI],
    )

    assert llm2.dict() == llm.dict()
    llm_string_2 = dumps(llm2)
    assert llm_string_2 == llm_string
    assert isinstance(llm2, OpenAI)


def test_load_openai_llm() -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")  # type: ignore[call-arg]
    llm_obj = dumpd(llm)
    llm2 = load(
        llm_obj,
        secrets_map={"OPENAI_API_KEY": "hello"},
        allowed_objects=[OpenAI],
    )

    assert llm2.dict() == llm.dict()
    assert dumpd(llm2) == llm_obj
    assert isinstance(llm2, OpenAI)


def test_loads_openai_chat() -> None:
    llm = ChatOpenAI(  # type: ignore[call-arg]
        model=OPENAI_CHAT_TEST_MODEL,
        temperature=0.5,
        openai_api_key="hello",
    )
    llm_string = dumps(llm)
    llm2 = loads(
        llm_string,
        secrets_map={"OPENAI_API_KEY": "hello"},
        allowed_objects=[ChatOpenAI],
    )

    assert llm2.dict() == llm.dict()
    llm_string_2 = dumps(llm2)
    assert llm_string_2 == llm_string
    assert isinstance(llm2, ChatOpenAI)


def test_load_openai_chat() -> None:
    llm = ChatOpenAI(  # type: ignore[call-arg]
        model=OPENAI_CHAT_TEST_MODEL,
        temperature=0.5,
        openai_api_key="hello",
    )
    llm_obj = dumpd(llm)
    llm2 = load(
        llm_obj,
        secrets_map={"OPENAI_API_KEY": "hello"},
        allowed_objects=[ChatOpenAI],
    )

    assert llm2.dict() == llm.dict()
    assert dumpd(llm2) == llm_obj
    assert isinstance(llm2, ChatOpenAI)


def test_load_openai_chat_openai_compatible_endpoint_config() -> None:
    """Test serialization of OpenAI-compatible endpoint runtime config."""
    llm = ChatOpenAI(  # type: ignore[call-arg]
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key="hello",
        api_key_secret_id="OPENROUTER_API_KEY",
        use_responses_api=True,
    )

    llm_obj = dumpd(llm)
    kwargs = llm_obj["kwargs"]
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert "openai_api_base" not in kwargs
    assert kwargs["openai_api_key"]["id"] == ["OPENROUTER_API_KEY"]
    assert "api_key_secret_id" not in kwargs

    llm2 = load(
        llm_obj,
        secrets_map={"OPENROUTER_API_KEY": "hello"},
        allowed_objects=[ChatOpenAI],
    )

    assert isinstance(llm2, ChatOpenAI)
    assert llm2.openai_api_base == "https://openrouter.ai/api/v1"


def test_loads_runnable_sequence_prompt_model() -> None:
    """Test serialization/deserialization of a chain:

    `prompt | model (RunnableSequence)`
    """
    prompt = ChatPromptTemplate.from_messages([("user", "Hello, {name}!")])
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key="hello")  # type: ignore[call-arg]
    chain = prompt | model

    # Verify the chain is a RunnableSequence
    assert isinstance(chain, RunnableSequence)

    # Serialize
    chain_string = dumps(chain)

    # Deserialize
    # (ChatPromptTemplate contains HumanMessagePromptTemplate and PromptTemplate)
    chain2 = loads(
        chain_string,
        secrets_map={"OPENAI_API_KEY": "hello"},
        allowed_objects=[
            RunnableSequence,
            ChatPromptTemplate,
            HumanMessagePromptTemplate,
            PromptTemplate,
            ChatOpenAI,
        ],
    )

    # Verify structure
    assert isinstance(chain2, RunnableSequence)
    assert isinstance(chain2.first, ChatPromptTemplate)
    assert isinstance(chain2.last, ChatOpenAI)

    # Verify round-trip serialization
    assert dumps(chain2) == chain_string


def test_load_runnable_sequence_prompt_model() -> None:
    """Test load() with a chain:

    `prompt | model (RunnableSequence)`.
    """
    prompt = ChatPromptTemplate.from_messages([("user", "Tell me about {topic}")])
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key="hello")  # type: ignore[call-arg]
    chain = prompt | model

    # Serialize
    chain_obj = dumpd(chain)

    # Deserialize
    # (ChatPromptTemplate contains HumanMessagePromptTemplate and PromptTemplate)
    chain2 = load(
        chain_obj,
        secrets_map={"OPENAI_API_KEY": "hello"},
        allowed_objects=[
            RunnableSequence,
            ChatPromptTemplate,
            HumanMessagePromptTemplate,
            PromptTemplate,
            ChatOpenAI,
        ],
    )

    # Verify structure
    assert isinstance(chain2, RunnableSequence)
    assert isinstance(chain2.first, ChatPromptTemplate)
    assert isinstance(chain2.last, ChatOpenAI)

    # Verify round-trip serialization
    assert dumpd(chain2) == chain_obj

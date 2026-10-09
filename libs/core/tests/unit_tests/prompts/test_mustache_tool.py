from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)


def test_mustache_schema_as_tool_chat_prompt_template() -> None:
    # 1. The issue repro for ChatPromptTemplate
    template = "Write a greeting for {{user.name}} from {{user.company}}."
    prompt = ChatPromptTemplate.from_messages(
        [("human", template)], template_format="mustache"
    )

    # Schema should have nested fields
    schema = prompt.get_input_jsonschema()
    assert "user" in schema["properties"]
    assert "$defs" in schema or "defs" in schema
    defs = schema.get("$defs", schema.get("defs", {}))
    assert "user" in defs
    assert "name" in defs["user"]["properties"]
    assert "company" in defs["user"]["properties"]

    tool = prompt.as_tool(name="greet", description="Greet a user.")

    # Passing the dict input -> as_tool should validate and convert it
    result = tool.invoke({"user": {"name": "Ada", "company": "Acme"}})

    # Assuming tool returns the PromptValue
    assert isinstance(result, list) or hasattr(result, "messages")
    messages = result.messages if hasattr(result, "messages") else result
    assert len(messages) == 1
    assert messages[0].content == "Write a greeting for Ada from Acme."


def test_mustache_schema_as_tool_prompt_template() -> None:
    # 1. The issue repro for PromptTemplate
    template = "Write a greeting for {{user.name}} from {{user.company}}."
    prompt = PromptTemplate.from_template(template, template_format="mustache")

    tool = prompt.as_tool(name="greet", description="Greet a user.")

    result = tool.invoke({"user": {"name": "Ada", "company": "Acme"}})
    # For PromptTemplate, result is a StringPromptValue
    assert result.text == "Write a greeting for Ada from Acme."


def test_mustache_schema_nested_lists() -> None:
    # 2. Nested values in lists (mustache sections)
    template = "Greetings to: {{#users}}{{name}} from {{company}}, {{/users}}"
    prompt = ChatPromptTemplate.from_messages(
        [("human", template)], template_format="mustache"
    )

    tool = prompt.as_tool(name="greet_many", description="Greet users.")

    result = tool.invoke({"users": {"name": "Ada", "company": "Acme"}})

    messages = result.messages if hasattr(result, "messages") else result
    assert messages[0].content == "Greetings to: Ada from Acme, "


def test_mustache_messages_placeholder_as_tool() -> None:
    # 3. MessagesPlaceholder chain through as_tool() to prove nothing regresses
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a bot."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{{question}}"),
        ],
        template_format="mustache",
    )

    tool = prompt.as_tool(name="chat", description="Chat tool")

    result = tool.invoke(
        {
            "history": [HumanMessage(content="hi"), AIMessage(content="hello")],
            "question": "how are you?",
        }
    )

    messages = result.messages if hasattr(result, "messages") else result
    assert len(messages) == 4
    assert messages[0].content == "You are a bot."
    assert messages[1].content == "hi"
    assert messages[2].content == "hello"
    assert messages[3].content == "how are you?"


def test_mustache_schema_mixed_fstring() -> None:
    # 4. Mixed f-string plus mustache chat prompt for schema merge
    # NOTE: Since ChatPromptTemplate only has a single template_format property,
    # "mixed" means using MessagesPlaceholder and other standard vars.
    # Actually, ChatPromptTemplate currently forces all string messages
    # to use its template_format.
    # But let's check a placeholder and optional variables.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Hello {{admin_name}}"),
            MessagesPlaceholder("chat_history"),
            ("human", "My name is {{user.name}} and I work at {{user.company}}."),
        ],
        template_format="mustache",
    )

    schema = prompt.get_input_jsonschema()

    assert "admin_name" in schema["properties"]
    assert "chat_history" in schema["properties"]
    assert "user" in schema["properties"]

    defs = schema.get("$defs", schema.get("defs", {}))
    assert "user" in defs
    assert "name" in defs["user"]["properties"]

"""Tests for jinja2_unrestricted template format.

This test file verifies that jinja2_unrestricted allows safe attribute access
while still providing basic sandboxing against dangerous patterns.
"""

import pytest

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)
from langchain_core.prompts.few_shot_with_templates import FewShotPromptWithTemplates
from langchain_core.prompts.image import ImagePromptTemplate

jinja2 = pytest.importorskip("jinja2")


@pytest.mark.requires("jinja2")
class TestPromptTemplateJinja2Unrestricted:
    """Tests for PromptTemplate with jinja2_unrestricted format."""

    def test_basic_functionality(self) -> None:
        """Test basic jinja2_unrestricted template formatting."""
        template = "Hello {{ name }}!"
        prompt = PromptTemplate.from_template(
            template, template_format="jinja2_unrestricted"
        )
        result = prompt.format(name="World")
        assert result == "Hello World!"

    def test_attribute_access_allowed(self) -> None:
        """Test that jinja2_unrestricted allows safe attribute access."""
        template = "Length: {{ text.upper() }}"
        prompt = PromptTemplate.from_template(
            template, template_format="jinja2_unrestricted"
        )
        result = prompt.format(text="hello")
        assert result == "Length: HELLO"

    def test_object_attribute_access(self) -> None:
        """Test accessing attributes of custom objects."""

        class SimpleObject:
            def __init__(self, value: str) -> None:
                self.value = value

        template = "Value: {{ obj.value }}"
        prompt = PromptTemplate.from_template(
            template, template_format="jinja2_unrestricted"
        )
        result = prompt.format(obj=SimpleObject("test"))
        assert result == "Value: test"

    def test_dangerous_patterns_blocked(self) -> None:
        """Test that dangerous patterns are still blocked by SandboxedEnvironment."""
        # Test blocking access to __class__.__bases__
        template = "{{ ''.__class__.__bases__[0] }}"
        prompt = PromptTemplate.from_template(
            template, template_format="jinja2_unrestricted"
        )
        with pytest.raises(jinja2.exceptions.SecurityError):
            prompt.format()

    def test_multiple_variables(self) -> None:
        """Test with multiple input variables."""
        template = "Hello {{ name }}, you are {{ age }} years old."
        prompt = PromptTemplate.from_template(
            template, template_format="jinja2_unrestricted"
        )
        result = prompt.format(name="Alice", age=30)
        assert result == "Hello Alice, you are 30 years old."

    def test_complex_jinja2_features(self) -> None:
        """Test complex Jinja2 features like loops and conditionals."""
        template = """\
{% for item in items %}
- {{ item.upper() }}
{% endfor %}
"""
        prompt = PromptTemplate.from_template(
            template, template_format="jinja2_unrestricted"
        )
        result = prompt.format(items=["apple", "banana", "cherry"])
        # Jinja2 preserves the newlines in the template,
        # including the one after each item
        expected = "\n- APPLE\n\n- BANANA\n\n- CHERRY\n"
        assert result == expected

    async def test_async_format(self) -> None:
        """Test async formatting with jinja2_unrestricted."""
        template = "Hello {{ name }}!"
        prompt = PromptTemplate.from_template(
            template, template_format="jinja2_unrestricted"
        )
        result = await prompt.aformat(name="Async World")
        assert result == "Hello Async World!"

    def test_input_variables_extraction(self) -> None:
        """Test that input variables are correctly extracted."""
        template = "{{ foo }} and {{ bar }}"
        prompt = PromptTemplate.from_template(
            template, template_format="jinja2_unrestricted"
        )
        assert sorted(prompt.input_variables) == ["bar", "foo"]

    def test_partial_variables(self) -> None:
        """Test partial variables with jinja2_unrestricted."""
        template = "{{ greeting }} {{ name }}"
        prompt = PromptTemplate.from_template(
            template,
            template_format="jinja2_unrestricted",
            partial_variables={"greeting": "Hello"},
        )
        result = prompt.format(name="World")
        assert result == "Hello World"


@pytest.mark.requires("jinja2")
class TestFewShotPromptTemplateJinja2Unrestricted:
    """Tests for FewShotPromptTemplate with jinja2_unrestricted format."""

    def test_basic_functionality(self) -> None:
        """Test basic few-shot prompt with jinja2_unrestricted."""
        example_prompt = PromptTemplate.from_template(
            "Q: {{ question }}\nA: {{ answer }}", template_format="jinja2_unrestricted"
        )

        examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]

        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Answer the following questions:",
            suffix="Q: {{ input }}\nA:",
            input_variables=["input"],
            template_format="jinja2_unrestricted",
        )

        result = prompt.format(input="What is 4+4?")
        assert "What is 2+2?" in result
        assert "4" in result
        assert "What is 4+4?" in result

    def test_with_attribute_access(self) -> None:
        """Test few-shot with attribute access in prefix/suffix."""
        example_prompt = PromptTemplate.from_template(
            "Input: {{ text }}", template_format="jinja2_unrestricted"
        )

        examples = [{"text": "example1"}, {"text": "example2"}]

        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="",
            suffix="User query: {{ query.strip() }}",
            input_variables=["query"],
            template_format="jinja2_unrestricted",
        )

        result = prompt.format(query="  test query  ")
        assert "test query" in result
        assert "  test query  " not in result

    async def test_async_format(self) -> None:
        """Test async formatting for few-shot prompts."""
        example_prompt = PromptTemplate.from_template(
            "Q: {{ question }}", template_format="jinja2_unrestricted"
        )

        examples = [{"question": "What is AI?"}]

        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="",
            suffix="Q: {{ input }}",
            input_variables=["input"],
            template_format="jinja2_unrestricted",
        )

        result = await prompt.aformat(input="What is ML?")
        assert "What is AI?" in result
        assert "What is ML?" in result


@pytest.mark.requires("jinja2")
class TestFewShotPromptWithTemplatesJinja2Unrestricted:
    """Tests for FewShotPromptWithTemplates with jinja2_unrestricted format."""

    def test_basic_functionality(self) -> None:
        """Test FewShotPromptWithTemplates with jinja2_unrestricted."""
        example_prompt = PromptTemplate.from_template(
            "Example: {{ text }}", template_format="jinja2_unrestricted"
        )

        examples = [{"text": "first"}, {"text": "second"}]

        suffix = PromptTemplate.from_template(
            "Query: {{ query }}", template_format="jinja2_unrestricted"
        )

        prefix = PromptTemplate.from_template(
            "Instructions: {{ instruction }}", template_format="jinja2_unrestricted"
        )

        prompt = FewShotPromptWithTemplates(
            examples=examples,
            example_prompt=example_prompt,
            suffix=suffix,
            prefix=prefix,
            input_variables=["query", "instruction"],
            template_format="jinja2_unrestricted",
        )

        result = prompt.format(query="test", instruction="Be helpful")
        assert "first" in result
        assert "second" in result
        assert "test" in result
        assert "Be helpful" in result

    def test_with_attribute_access(self) -> None:
        """Test with string methods in templates."""
        example_prompt = PromptTemplate.from_template(
            "{{ text.upper() }}", template_format="jinja2_unrestricted"
        )

        examples = [{"text": "hello"}, {"text": "world"}]

        suffix = PromptTemplate.from_template(
            "{{ query }}", template_format="jinja2_unrestricted"
        )

        prompt = FewShotPromptWithTemplates(
            examples=examples,
            example_prompt=example_prompt,
            suffix=suffix,
            input_variables=["query"],
            template_format="jinja2_unrestricted",
        )

        result = prompt.format(query="test")
        assert "HELLO" in result
        assert "WORLD" in result

    async def test_async_format(self) -> None:
        """Test async formatting for FewShotPromptWithTemplates."""
        example_prompt = PromptTemplate.from_template(
            "Example: {{ text }}", template_format="jinja2_unrestricted"
        )

        examples = [{"text": "async test"}]

        suffix = PromptTemplate.from_template(
            "Query: {{ query }}", template_format="jinja2_unrestricted"
        )

        prompt = FewShotPromptWithTemplates(
            examples=examples,
            example_prompt=example_prompt,
            suffix=suffix,
            input_variables=["query"],
            template_format="jinja2_unrestricted",
        )

        result = await prompt.aformat(query="async query")
        assert "async test" in result
        assert "async query" in result


@pytest.mark.requires("jinja2")
class TestChatPromptTemplateJinja2Unrestricted:
    """Tests for ChatPromptTemplate with jinja2_unrestricted format."""

    def test_basic_functionality(self) -> None:
        """Test basic chat prompt with jinja2_unrestricted."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a {{ role }} assistant."),
                ("human", "{{ question }}"),
            ],
            template_format="jinja2_unrestricted",
        )

        messages = prompt.format_messages(role="helpful", question="Hello!")
        assert len(messages) == 2
        assert "helpful" in messages[0].content
        assert "Hello!" in messages[1].content

    def test_attribute_access_allowed(self) -> None:
        """Test that attribute access works in chat templates."""
        prompt = ChatPromptTemplate.from_messages(
            [("human", "{{ text.upper() }}")],
            template_format="jinja2_unrestricted",
        )

        messages = prompt.format_messages(text="hello world")
        assert messages[0].content == "HELLO WORLD"

    def test_object_attribute_access(self) -> None:
        """Test accessing attributes of objects in chat templates."""

        class Message:
            def __init__(self, content: str) -> None:
                self.content = content

        prompt = ChatPromptTemplate.from_messages(
            [("human", "Content: {{ msg.content }}")],
            template_format="jinja2_unrestricted",
        )

        messages = prompt.format_messages(msg=Message("test message"))
        assert messages[0].content == "Content: test message"

    def test_dangerous_patterns_blocked(self) -> None:
        """Test that dangerous patterns are blocked in chat templates."""
        prompt = ChatPromptTemplate.from_messages(
            [("human", "{{ ''.__class__.__bases__ }}")],
            template_format="jinja2_unrestricted",
        )

        with pytest.raises(jinja2.exceptions.SecurityError):
            prompt.format_messages()

    async def test_async_format_messages(self) -> None:
        """Test async message formatting."""
        prompt = ChatPromptTemplate.from_messages(
            [("human", "{{ question }}")],
            template_format="jinja2_unrestricted",
        )

        messages = await prompt.aformat_messages(question="Async question")
        assert messages[0].content == "Async question"


@pytest.mark.requires("jinja2")
class TestFewShotChatMessagePromptTemplateJinja2Unrestricted:
    """Tests for FewShotChatMessagePromptTemplate with jinja2_unrestricted."""

    def test_basic_functionality(self) -> None:
        """Test basic few-shot chat prompt with jinja2_unrestricted."""
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{{ input }}"),
                ("ai", "{{ output }}"),
            ],
            template_format="jinja2_unrestricted",
        )

        examples = [
            {"input": "Hi", "output": "Hello!"},
            {"input": "Bye", "output": "Goodbye!"},
        ]

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
        )

        messages = few_shot_prompt.format_messages()
        assert len(messages) == 4  # 2 examples * 2 messages each
        assert any("Hi" in str(msg.content) for msg in messages)
        assert any("Hello!" in str(msg.content) for msg in messages)

    def test_with_attribute_access(self) -> None:
        """Test few-shot chat with attribute access."""
        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{{ text.strip() }}")],
            template_format="jinja2_unrestricted",
        )

        examples = [{"text": "  example  "}]

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
        )

        messages = few_shot_prompt.format_messages()
        assert messages[0].content == "example"

    async def test_async_format_messages(self) -> None:
        """Test async message formatting for few-shot chat."""
        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{{ input }}")],
            template_format="jinja2_unrestricted",
        )

        examples = [{"input": "async test"}]

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
        )

        messages = await few_shot_prompt.aformat_messages()
        assert messages[0].content == "async test"


@pytest.mark.requires("jinja2")
class TestImagePromptTemplateJinja2Unrestricted:
    """Tests for ImagePromptTemplate with jinja2_unrestricted format."""

    def test_basic_functionality(self) -> None:
        """Test basic image prompt with jinja2_unrestricted."""
        template = {"url": "https://example.com/{{ image_id }}.jpg"}
        prompt = ImagePromptTemplate(
            template=template,
            template_format="jinja2_unrestricted",
        )

        result = prompt.format(url=None, image_id="12345")
        assert result["url"] == "https://example.com/12345.jpg"

    def test_with_detail_parameter(self) -> None:
        """Test image prompt with detail parameter."""
        template = {
            "url": "https://example.com/image.jpg",
            "detail": "{{ quality }}",
        }
        prompt = ImagePromptTemplate(
            template=template,
            template_format="jinja2_unrestricted",
        )

        result = prompt.format(url=None, quality="high")
        assert result["url"] == "https://example.com/image.jpg"
        assert result["detail"] == "high"

    def test_attribute_access_in_url(self) -> None:
        """Test attribute access in image URL template."""
        template = {"url": "{{ base_url.strip() }}/image.jpg"}
        prompt = ImagePromptTemplate(
            template=template,
            template_format="jinja2_unrestricted",
        )

        result = prompt.format(url=None, base_url="  https://example.com  ")
        assert result["url"] == "https://example.com/image.jpg"

    async def test_async_format(self) -> None:
        """Test async formatting for image prompts."""
        template = {"url": "https://example.com/{{ image_id }}.jpg"}
        prompt = ImagePromptTemplate(
            template=template,
            template_format="jinja2_unrestricted",
        )

        result = await prompt.aformat(url=None, image_id="async")
        assert result["url"] == "https://example.com/async.jpg"


@pytest.mark.requires("jinja2")
class TestJinja2UnrestrictedComparison:
    """Tests comparing jinja2 vs jinja2_unrestricted behavior."""

    def test_jinja2_blocks_attributes_unrestricted_allows(self) -> None:
        """Test that jinja2 blocks attributes but jinja2_unrestricted allows them."""
        # jinja2 (restricted) blocks attribute access
        restricted_prompt = PromptTemplate.from_template(
            "{{ text.upper() }}", template_format="jinja2"
        )
        with pytest.raises(jinja2.exceptions.SecurityError):
            restricted_prompt.format(text="hello")

        # jinja2_unrestricted allows attribute access
        unrestricted_prompt = PromptTemplate.from_template(
            "{{ text.upper() }}", template_format="jinja2_unrestricted"
        )
        result = unrestricted_prompt.format(text="hello")
        assert result == "HELLO"

    def test_both_block_dangerous_patterns(self) -> None:
        """Test that both formats block dangerous patterns."""
        dangerous_template = "{{ ''.__class__.__bases__ }}"

        # jinja2 blocks it
        restricted_prompt = PromptTemplate.from_template(
            dangerous_template, template_format="jinja2"
        )
        with pytest.raises(jinja2.exceptions.SecurityError):
            restricted_prompt.format()

        # jinja2_unrestricted also blocks it
        unrestricted_prompt = PromptTemplate.from_template(
            dangerous_template, template_format="jinja2_unrestricted"
        )
        with pytest.raises(jinja2.exceptions.SecurityError):
            unrestricted_prompt.format()


@pytest.mark.requires("jinja2")
class TestJinja2UnrestrictedValidation:
    """Tests for template validation with jinja2_unrestricted."""

    def test_missing_input_variables_warning(self) -> None:
        """Test warning when input variables are missing."""
        template = "{{ foo }} and {{ bar }}"
        with pytest.warns(UserWarning, match="Missing variables"):
            PromptTemplate(
                input_variables=["foo"],
                template=template,
                template_format="jinja2_unrestricted",
                validate_template=True,
            )

    def test_extra_input_variables_warning(self) -> None:
        """Test warning when extra input variables are provided."""
        template = "{{ foo }}"
        with pytest.warns(UserWarning, match="Extra variables"):
            PromptTemplate(
                input_variables=["foo", "bar", "baz"],
                template=template,
                template_format="jinja2_unrestricted",
                validate_template=True,
            )

    def test_correct_variables_no_warning(self) -> None:
        """Test no warning when variables are correct."""
        template = "{{ foo }}"
        # Should not raise any warnings
        prompt = PromptTemplate(
            input_variables=["foo"],
            template=template,
            template_format="jinja2_unrestricted",
            validate_template=True,
        )
        assert prompt.input_variables == ["foo"]

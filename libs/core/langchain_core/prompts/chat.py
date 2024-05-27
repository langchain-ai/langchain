"""Chat prompt template."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

from langchain_core._api import deprecated
from langchain_core.load import Serializable
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompt_values import ChatPromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import StringPromptTemplate, get_template_variables
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env


class BaseMessagePromptTemplate(Serializable, ABC):
    """Base class for message prompt templates."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs. Should return a list of BaseMessages.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs. Should return a list of BaseMessages.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return self.format_messages(**kwargs)

    @property
    @abstractmethod
    def input_variables(self) -> List[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variables.
        """

    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation."""
        raise NotImplementedError

    def pretty_print(self) -> None:
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Combine two prompt templates.

        Args:
            other: Another prompt template.

        Returns:
            Combined prompt template.
        """
        prompt = ChatPromptTemplate(messages=[self])  # type: ignore[call-arg]
        return prompt + other


class MessagesPlaceholder(BaseMessagePromptTemplate):
    """Prompt template that assumes variable is already list of messages.

    A placeholder which can be used to pass in a list of messages.

    Direct usage:

        .. code-block:: python

            from langchain_core.prompts import MessagesPlaceholder

            prompt = MessagesPlaceholder("history")
            prompt.format_messages() # raises KeyError

            prompt = MessagesPlaceholder("history", optional=True)
            prompt.format_messages() # returns empty list []

            prompt.format_messages(
                history=[
                    ("system", "You are an AI assistant."),
                    ("human", "Hello!"),
                ]
            )
            # -> [
            #     SystemMessage(content="You are an AI assistant."),
            #     HumanMessage(content="Hello!"),
            # ]

    Building a prompt with chat history:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    MessagesPlaceholder("history"),
                    ("human", "{question}")
                ]
            )
            prompt.invoke(
               {
                   "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
                   "question": "now multiply that by 4"
               }
            )
            # -> ChatPromptValue(messages=[
            #     SystemMessage(content="You are a helpful assistant."),
            #     HumanMessage(content="what's 5 + 2"),
            #     AIMessage(content="5 + 2 is 7"),
            #     HumanMessage(content="now multiply that by 4"),
            # ])
    """

    variable_name: str
    """Name of variable to use as messages."""

    optional: bool = False
    """If True format_messages can be called with no arguments and will return an empty 
        list. If False then a named argument with name `variable_name` must be passed 
        in, even if the value is an empty list."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    def __init__(self, variable_name: str, *, optional: bool = False, **kwargs: Any):
        super().__init__(variable_name=variable_name, optional=optional, **kwargs)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessage.
        """
        value = (
            kwargs.get(self.variable_name, [])
            if self.optional
            else kwargs[self.variable_name]
        )
        if not isinstance(value, list):
            raise ValueError(
                f"variable {self.variable_name} should be a list of base messages, "
                f"got {value}"
            )
        return convert_to_messages(value)

    @property
    def input_variables(self) -> List[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        return [self.variable_name] if not self.optional else []

    def pretty_repr(self, html: bool = False) -> str:
        var = "{" + self.variable_name + "}"
        if html:
            title = get_msg_title_repr("Messages Placeholder", bold=True)
            var = get_colored_text(var, "yellow")
        else:
            title = get_msg_title_repr("Messages Placeholder")
        return f"{title}\n\n{var}"


MessagePromptTemplateT = TypeVar(
    "MessagePromptTemplateT", bound="BaseStringMessagePromptTemplate"
)
"""Type variable for message prompt templates."""


class BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC):
    """Base class for message prompt templates that use a string prompt template."""

    prompt: StringPromptTemplate
    """String prompt template."""
    additional_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the prompt template."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    @classmethod
    def from_template(
        cls: Type[MessagePromptTemplateT],
        template: str,
        template_format: str = "f-string",
        partial_variables: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> MessagePromptTemplateT:
        """Create a class from a string template.

        Args:
            template: a template.
            template_format: format of the template.
            partial_variables: A dictionary of variables that can be used to partially
                               fill in the template. For example, if the template is
                              `"{variable1} {variable2}"`, and `partial_variables` is
                              `{"variable1": "foo"}`, then the final prompt will be
                              `"foo {variable2}"`.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        prompt = PromptTemplate.from_template(
            template,
            template_format=template_format,
            partial_variables=partial_variables,
        )
        return cls(prompt=prompt, **kwargs)

    @classmethod
    def from_template_file(
        cls: Type[MessagePromptTemplateT],
        template_file: Union[str, Path],
        input_variables: List[str],
        **kwargs: Any,
    ) -> MessagePromptTemplateT:
        """Create a class from a template file.

        Args:
            template_file: path to a template file. String or Path.
            input_variables: list of input variables.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        prompt = PromptTemplate.from_file(template_file, input_variables)
        return cls(prompt=prompt, **kwargs)

    @abstractmethod
    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        return self.format(**kwargs)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> List[str]:
        """
        Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        return self.prompt.input_variables

    def pretty_repr(self, html: bool = False) -> str:
        # TODO: Handle partials
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        return f"{title}\n\n{self.prompt.pretty_repr(html=html)}"


class ChatMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """Chat message prompt template."""

    role: str
    """Role of the message."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        text = self.prompt.format(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        text = await self.prompt.aformat(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )


_StringImageMessagePromptTemplateT = TypeVar(
    "_StringImageMessagePromptTemplateT", bound="_StringImageMessagePromptTemplate"
)


class _TextTemplateParam(TypedDict, total=False):
    text: Union[str, Dict]


class _ImageTemplateParam(TypedDict, total=False):
    image_url: Union[str, Dict]


class _StringImageMessagePromptTemplate(BaseMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""

    prompt: Union[
        StringPromptTemplate, List[Union[StringPromptTemplate, ImagePromptTemplate]]
    ]
    """Prompt template."""
    additional_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the prompt template."""

    _msg_class: Type[BaseMessage]

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    @classmethod
    def from_template(
        cls: Type[_StringImageMessagePromptTemplateT],
        template: Union[str, List[Union[str, _TextTemplateParam, _ImageTemplateParam]]],
        template_format: str = "f-string",
        **kwargs: Any,
    ) -> _StringImageMessagePromptTemplateT:
        """Create a class from a string template.

        Args:
            template: a template.
            template_format: format of the template.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        if isinstance(template, str):
            prompt: Union[StringPromptTemplate, List] = PromptTemplate.from_template(
                template, template_format=template_format
            )
            return cls(prompt=prompt, **kwargs)
        elif isinstance(template, list):
            prompt = []
            for tmpl in template:
                if isinstance(tmpl, str) or isinstance(tmpl, dict) and "text" in tmpl:
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast(_TextTemplateParam, tmpl)["text"]  # type: ignore[assignment]
                    prompt.append(
                        PromptTemplate.from_template(
                            text, template_format=template_format
                        )
                    )
                elif isinstance(tmpl, dict) and "image_url" in tmpl:
                    img_template = cast(_ImageTemplateParam, tmpl)["image_url"]
                    if isinstance(img_template, str):
                        vars = get_template_variables(img_template, "f-string")
                        if vars:
                            if len(vars) > 1:
                                raise ValueError(
                                    "Only one format variable allowed per image"
                                    f" template.\nGot: {vars}"
                                    f"\nFrom: {tmpl}"
                                )
                            input_variables = [vars[0]]
                        else:
                            input_variables = None
                        img_template = {"url": img_template}
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables, template=img_template
                        )
                    elif isinstance(img_template, dict):
                        img_template = dict(img_template)
                        if "url" in img_template:
                            input_variables = get_template_variables(
                                img_template["url"], "f-string"
                            )
                        else:
                            input_variables = None
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables, template=img_template
                        )
                    else:
                        raise ValueError()
                    prompt.append(img_template_obj)
                else:
                    raise ValueError()
            return cls(prompt=prompt, **kwargs)
        else:
            raise ValueError()

    @classmethod
    def from_template_file(
        cls: Type[_StringImageMessagePromptTemplateT],
        template_file: Union[str, Path],
        input_variables: List[str],
        **kwargs: Any,
    ) -> _StringImageMessagePromptTemplateT:
        """Create a class from a template file.

        Args:
            template_file: path to a template file. String or Path.
            input_variables: list of input variables.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        with open(str(template_file), "r") as f:
            template = f.read()
        return cls.from_template(template, input_variables=input_variables, **kwargs)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> List[str]:
        """
        Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        input_variables = [iv for prompt in prompts for iv in prompt.input_variables]
        return input_variables

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = self.prompt.format(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )
        else:
            content: List = []
            for prompt in self.prompt:
                inputs = {var: kwargs[var] for var in prompt.input_variables}
                if isinstance(prompt, StringPromptTemplate):
                    formatted: Union[str, ImageURL] = prompt.format(**inputs)
                    content.append({"type": "text", "text": formatted})
                elif isinstance(prompt, ImagePromptTemplate):
                    formatted = prompt.format(**inputs)
                    content.append({"type": "image_url", "image_url": formatted})
            return self._msg_class(
                content=content, additional_kwargs=self.additional_kwargs
            )

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = await self.prompt.aformat(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )
        else:
            content: List = []
            for prompt in self.prompt:
                inputs = {var: kwargs[var] for var in prompt.input_variables}
                if isinstance(prompt, StringPromptTemplate):
                    formatted: Union[str, ImageURL] = await prompt.aformat(**inputs)
                    content.append({"type": "text", "text": formatted})
                elif isinstance(prompt, ImagePromptTemplate):
                    formatted = await prompt.aformat(**inputs)
                    content.append({"type": "image_url", "image_url": formatted})
            return self._msg_class(
                content=content, additional_kwargs=self.additional_kwargs
            )

    def pretty_repr(self, html: bool = False) -> str:
        # TODO: Handle partials
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        prompt_reprs = "\n\n".join(prompt.pretty_repr(html=html) for prompt in prompts)
        return f"{title}\n\n{prompt_reprs}"


class HumanMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""

    _msg_class: Type[BaseMessage] = HumanMessage


class AIMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """AI message prompt template. This is a message sent from the AI."""

    _msg_class: Type[BaseMessage] = AIMessage

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]


class SystemMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """System message prompt template.
    This is a message that is not sent to the user.
    """

    _msg_class: Type[BaseMessage] = SystemMessage

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]


class BaseChatPromptTemplate(BasePromptTemplate, ABC):
    """Base class for chat prompt templates."""

    @property
    def lc_attributes(self) -> Dict:
        """
        Return a list of attribute names that should be included in the
        serialized kwargs. These attributes must be accepted by the
        constructor.
        """
        return {"input_variables": self.input_variables}

    def format(self, **kwargs: Any) -> str:
        """Format the chat template into a string.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            formatted string
        """
        return self.format_prompt(**kwargs).to_string()

    async def aformat(self, **kwargs: Any) -> str:
        """Format the chat template into a string.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            formatted string
        """
        return (await self.aformat_prompt(**kwargs)).to_string()

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """
        Format prompt. Should return a PromptValue.
        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            PromptValue.
        """
        messages = self.format_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        messages = await self.aformat_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format kwargs into a list of messages."""

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format kwargs into a list of messages."""
        return self.format_messages(**kwargs)

    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation."""
        raise NotImplementedError

    def pretty_print(self) -> None:
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


MessageLike = Union[BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate]

MessageLikeRepresentation = Union[
    MessageLike,
    Tuple[
        Union[str, Type],
        Union[str, List[dict], List[object]],
    ],
    str,
]


class ChatPromptTemplate(BaseChatPromptTemplate):
    """Prompt template for chat models.

    Use to create flexible templated prompts for chat models.

    Examples:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI bot. Your name is {name}."),
                ("human", "Hello, how are you doing?"),
                ("ai", "I'm doing well, thanks!"),
                ("human", "{user_input}"),
            ])

            prompt_value = template.invoke(
                {
                    "name": "Bob",
                    "user_input": "What is your name?"
                }
            )
            # Output:
            # ChatPromptValue(
            #    messages=[
            #        SystemMessage(content='You are a helpful AI bot. Your name is Bob.'),
            #        HumanMessage(content='Hello, how are you doing?'),
            #        AIMessage(content="I'm doing well, thanks!"),
            #        HumanMessage(content='What is your name?')
            #    ]
            #)

    Messages Placeholder:

        .. code-block:: python

            # In addition to Human/AI/Tool/Function messages,
            # you can initialize the template with a MessagesPlaceholder
            # either using the class directly or with the shorthand tuple syntax:

            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI bot."),
                # Means the template will receive an optional list of messages under
                # the "conversation" key
                ("placeholder", "{conversation}")
                # Equivalently:
                # MessagesPlaceholder(variable_name="conversation", optional=True)
            ])

            prompt_value = template.invoke(
                {
                    "conversation": [
                        ("human", "Hi!"),
                        ("ai", "How can I assist you today?"),
                        ("human", "Can you make me an ice cream sundae?"),
                        ("ai", "No.")
                    ]
                }
            )

            # Output:
            # ChatPromptValue(
            #    messages=[
            #        SystemMessage(content='You are a helpful AI bot.'),
            #        HumanMessage(content='Hi!'),
            #        AIMessage(content='How can I assist you today?'),
            #        HumanMessage(content='Can you make me an ice cream sundae?'),
            #        AIMessage(content='No.'),
            #    ]
            #)

    Single-variable template:

        If your prompt has only a single input variable (i.e., 1 instance of "{variable_nams}"),
        and you invoke the template with a non-dict object, the prompt template will
        inject the provided argument into that variable location.


        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI bot. Your name is Carl."),
                ("human", "{user_input}"),
            ])

            prompt_value = template.invoke("Hello, there!")
            # Equivalent to
            # prompt_value = template.invoke({"user_input": "Hello, there!"})

            # Output:
            #  ChatPromptValue(
            #     messages=[
            #         SystemMessage(content='You are a helpful AI bot. Your name is Carl.'),
            #         HumanMessage(content='Hello, there!'),
            #     ]
            # )

    """  # noqa: E501

    input_variables: List[str]
    """List of input variables in template messages. Used for validation."""
    messages: List[MessageLike]
    """List of messages consisting of either message prompt templates or messages."""
    validate_template: bool = False
    """Whether or not to try validating the template."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Combine two prompt templates.

        Args:
            other: Another prompt template.

        Returns:
            Combined prompt template.
        """
        # Allow for easy combining
        if isinstance(other, ChatPromptTemplate):
            return ChatPromptTemplate(messages=self.messages + other.messages)  # type: ignore[call-arg]
        elif isinstance(
            other, (BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate)
        ):
            return ChatPromptTemplate(messages=self.messages + [other])  # type: ignore[call-arg]
        elif isinstance(other, (list, tuple)):
            _other = ChatPromptTemplate.from_messages(other)
            return ChatPromptTemplate(messages=self.messages + _other.messages)  # type: ignore[call-arg]
        elif isinstance(other, str):
            prompt = HumanMessagePromptTemplate.from_template(other)
            return ChatPromptTemplate(messages=self.messages + [prompt])  # type: ignore[call-arg]
        else:
            raise NotImplementedError(f"Unsupported operand type for +: {type(other)}")

    @root_validator(pre=True)
    def validate_input_variables(cls, values: dict) -> dict:
        """Validate input variables.

        If input_variables is not set, it will be set to the union of
        all input variables in the messages.

        Args:
            values: values to validate.

        Returns:
            Validated values.
        """
        messages = values["messages"]
        input_vars = set()
        input_types: Dict[str, Any] = values.get("input_types", {})
        for message in messages:
            if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
                input_vars.update(message.input_variables)
            if isinstance(message, MessagesPlaceholder):
                if message.variable_name not in input_types:
                    input_types[message.variable_name] = List[AnyMessage]
        if "partial_variables" in values:
            input_vars = input_vars - set(values["partial_variables"])
        if "input_variables" in values and values.get("validate_template"):
            if input_vars != set(values["input_variables"]):
                raise ValueError(
                    "Got mismatched input_variables. "
                    f"Expected: {input_vars}. "
                    f"Got: {values['input_variables']}"
                )
        else:
            values["input_variables"] = sorted(input_vars)
        values["input_types"] = input_types
        return values

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> ChatPromptTemplate:
        """Create a chat prompt template from a template string.

        Creates a chat template consisting of a single message assumed to be from
        the human.

        Args:
            template: template string
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        prompt_template = PromptTemplate.from_template(template, **kwargs)
        message = HumanMessagePromptTemplate(prompt=prompt_template)
        return cls.from_messages([message])

    @classmethod
    @deprecated("0.0.1", alternative="from_messages classmethod", pending=True)
    def from_role_strings(
        cls, string_messages: List[Tuple[str, str]]
    ) -> ChatPromptTemplate:
        """Create a chat prompt template from a list of (role, template) tuples.

        Args:
            string_messages: list of (role, template) tuples.

        Returns:
            a chat prompt template
        """
        return cls(  # type: ignore[call-arg]
            messages=[
                ChatMessagePromptTemplate.from_template(template, role=role)
                for role, template in string_messages
            ]
        )

    @classmethod
    @deprecated("0.0.1", alternative="from_messages classmethod", pending=True)
    def from_strings(
        cls, string_messages: List[Tuple[Type[BaseMessagePromptTemplate], str]]
    ) -> ChatPromptTemplate:
        """Create a chat prompt template from a list of (role class, template) tuples.

        Args:
            string_messages: list of (role class, template) tuples.

        Returns:
            a chat prompt template
        """
        return cls.from_messages(string_messages)

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[MessageLikeRepresentation],
        template_format: Literal["f-string", "mustache"] = "f-string",
    ) -> ChatPromptTemplate:
        """Create a chat prompt template from a variety of message formats.

        Examples:

            Instantiation from a list of message templates:

            .. code-block:: python

                template = ChatPromptTemplate.from_messages([
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "That's good to hear."),
                ])

            Instantiation from mixed message formats:

            .. code-block:: python

                template = ChatPromptTemplate.from_messages([
                    SystemMessage(content="hello"),
                    ("human", "Hello, how are you?"),
                ])

        Args:
            messages: sequence of message representations.
                  A message can be represented using the following formats:
                  (1) BaseMessagePromptTemplate, (2) BaseMessage, (3) 2-tuple of
                  (message type, template); e.g., ("human", "{user_input}"),
                  (4) 2-tuple of (message class, template), (4) a string which is
                  shorthand for ("human", template); e.g., "{user_input}"

        Returns:
            a chat prompt template
        """
        _messages = [
            _convert_to_message(message, template_format) for message in messages
        ]

        # Automatically infer input variables from messages
        input_vars: Set[str] = set()
        partial_vars: Dict[str, Any] = {}
        for _message in _messages:
            if isinstance(_message, MessagesPlaceholder) and _message.optional:
                partial_vars[_message.variable_name] = []
            elif isinstance(
                _message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)
            ):
                input_vars.update(_message.input_variables)

        return cls(
            input_variables=sorted(input_vars),
            messages=_messages,
            partial_variables=partial_vars,
        )

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the chat template into a list of finalized messages.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            list of formatted messages
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(
                message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
            ):
                message = message_template.format_messages(**kwargs)
                result.extend(message)
            else:
                raise ValueError(f"Unexpected input: {message_template}")
        return result

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the chat template into a list of finalized messages.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            list of formatted messages
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(
                message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
            ):
                message = await message_template.aformat_messages(**kwargs)
                result.extend(message)
            else:
                raise ValueError(f"Unexpected input: {message_template}")
        return result

    def partial(self, **kwargs: Any) -> ChatPromptTemplate:
        """Get a new ChatPromptTemplate with some input variables already filled in.

        Args:
            **kwargs: keyword arguments to use for filling in template variables. Ought
                        to be a subset of the input variables.

        Returns:
            A new ChatPromptTemplate.


        Example:

            .. code-block:: python

                from langchain_core.prompts import ChatPromptTemplate

                template = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are an AI assistant named {name}."),
                        ("human", "Hi I'm {user}"),
                        ("ai", "Hi there, {user}, I'm {name}."),
                        ("human", "{input}"),
                    ]
                )
                template2 = template.partial(user="Lucy", name="R2D2")

                template2.format_messages(input="hello")
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def append(self, message: MessageLikeRepresentation) -> None:
        """Append message to the end of the chat template.

        Args:
            message: representation of a message to append.
        """
        self.messages.append(_convert_to_message(message))

    def extend(self, messages: Sequence[MessageLikeRepresentation]) -> None:
        """Extend the chat template with a sequence of messages."""
        self.messages.extend([_convert_to_message(message) for message in messages])

    @overload
    def __getitem__(self, index: int) -> MessageLike:
        ...

    @overload
    def __getitem__(self, index: slice) -> ChatPromptTemplate:
        ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[MessageLike, ChatPromptTemplate]:
        """Use to index into the chat template."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.messages))
            messages = self.messages[start:stop:step]
            return ChatPromptTemplate.from_messages(messages)
        else:
            return self.messages[index]

    def __len__(self) -> int:
        """Get the length of the chat template."""
        return len(self.messages)

    @property
    def _prompt_type(self) -> str:
        """Name of prompt type."""
        return "chat"

    def save(self, file_path: Union[Path, str]) -> None:
        """Save prompt to file.

        Args:
            file_path: path to file.
        """
        raise NotImplementedError()

    def pretty_repr(self, html: bool = False) -> str:
        # TODO: handle partials
        return "\n\n".join(msg.pretty_repr(html=html) for msg in self.messages)


def _create_template_from_message_type(
    message_type: str,
    template: Union[str, list],
    template_format: Literal["f-string", "mustache"] = "f-string",
) -> BaseMessagePromptTemplate:
    """Create a message prompt template from a message type and template string.

    Args:
        message_type: str the type of the message template (e.g., "human", "ai", etc.)
        template: str the template string.

    Returns:
        a message prompt template of the appropriate type.
    """
    if message_type in ("human", "user"):
        message: BaseMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
            template, template_format=template_format
        )
    elif message_type in ("ai", "assistant"):
        message = AIMessagePromptTemplate.from_template(
            cast(str, template), template_format=template_format
        )
    elif message_type == "system":
        message = SystemMessagePromptTemplate.from_template(
            cast(str, template), template_format=template_format
        )
    elif message_type == "placeholder":
        if isinstance(template, str):
            if template[0] != "{" or template[-1] != "}":
                raise ValueError(
                    f"Invalid placeholder template: {template}."
                    " Expected a variable name surrounded by curly braces."
                )
            var_name = template[1:-1]
            message = MessagesPlaceholder(variable_name=var_name, optional=True)
        elif len(template) == 2 and isinstance(template[1], bool):
            var_name_wrapped, is_optional = template
            if not isinstance(var_name_wrapped, str):
                raise ValueError(
                    "Expected variable name to be a string." f" Got: {var_name_wrapped}"
                )
            if var_name_wrapped[0] != "{" or var_name_wrapped[-1] != "}":
                raise ValueError(
                    f"Invalid placeholder template: {var_name_wrapped}."
                    " Expected a variable name surrounded by curly braces."
                )
            var_name = var_name_wrapped[1:-1]

            message = MessagesPlaceholder(variable_name=var_name, optional=is_optional)
        else:
            raise ValueError(
                "Unexpected arguments for placeholder message type."
                " Expected either a single string variable name"
                " or a list of [variable_name: str, is_optional: bool]."
                f" Got: {template}"
            )
    else:
        raise ValueError(
            f"Unexpected message type: {message_type}. Use one of 'human',"
            f" 'user', 'ai', 'assistant', or 'system'."
        )
    return message


def _convert_to_message(
    message: MessageLikeRepresentation,
    template_format: Literal["f-string", "mustache"] = "f-string",
) -> Union[BaseMessage, BaseMessagePromptTemplate, BaseChatPromptTemplate]:
    """Instantiate a message from a variety of message formats.

    The message format can be one of the following:

    - BaseMessagePromptTemplate
    - BaseMessage
    - 2-tuple of (role string, template); e.g., ("human", "{user_input}")
    - 2-tuple of (message class, template)
    - string: shorthand for ("human", template); e.g., "{user_input}"

    Args:
        message: a representation of a message in one of the supported formats

    Returns:
        an instance of a message or a message template
    """
    if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
        _message: Union[
            BaseMessage, BaseMessagePromptTemplate, BaseChatPromptTemplate
        ] = message
    elif isinstance(message, BaseMessage):
        _message = message
    elif isinstance(message, str):
        _message = _create_template_from_message_type(
            "human", message, template_format=template_format
        )
    elif isinstance(message, tuple):
        if len(message) != 2:
            raise ValueError(f"Expected 2-tuple of (role, template), got {message}")
        message_type_str, template = message
        if isinstance(message_type_str, str):
            _message = _create_template_from_message_type(
                message_type_str, template, template_format=template_format
            )
        else:
            _message = message_type_str(
                prompt=PromptTemplate.from_template(
                    cast(str, template), template_format=template_format
                )
            )
    else:
        raise NotImplementedError(f"Unsupported message type: {type(message)}")

    return _message

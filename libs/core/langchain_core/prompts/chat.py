"""Chat prompt template."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

from pydantic import (
    Field,
    PositiveInt,
    SkipValidation,
    model_validator,
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
from langchain_core.prompts.string import (
    PromptTemplateFormat,
    StringPromptTemplate,
    get_template_variables,
)
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env


class BaseMessagePromptTemplate(Serializable, ABC):
    """Base class for message prompt templates."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable.
        Returns: True.
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs. Should return a list of BaseMessages.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format messages from kwargs.
        Should return a list of BaseMessages.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return self.format_messages(**kwargs)

    @property
    @abstractmethod
    def input_variables(self) -> list[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variables.
        """

    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML. Defaults to False.

        Returns:
            Human-readable representation.
        """
        raise NotImplementedError

    def pretty_print(self) -> None:
        """Print a human-readable representation."""
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

    Limiting the number of messages:

        .. code-block:: python

            from langchain_core.prompts import MessagesPlaceholder

            prompt = MessagesPlaceholder("history", n_messages=1)

            prompt.format_messages(
                history=[
                    ("system", "You are an AI assistant."),
                    ("human", "Hello!"),
                ]
            )
            # -> [
            #     HumanMessage(content="Hello!"),
            # ]
    """

    variable_name: str
    """Name of variable to use as messages."""

    optional: bool = False
    """If True format_messages can be called with no arguments and will return an empty
        list. If False then a named argument with name `variable_name` must be passed
        in, even if the value is an empty list."""

    n_messages: Optional[PositiveInt] = None
    """Maximum number of messages to include. If None, then will include all.
    Defaults to None."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    def __init__(
        self, variable_name: str, *, optional: bool = False, **kwargs: Any
    ) -> None:
        # mypy can't detect the init which is defined in the parent class
        # b/c these are BaseModel classes.
        super().__init__(  # type: ignore
            variable_name=variable_name, optional=optional, **kwargs
        )

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessage.

        Raises:
            ValueError: If variable is not a list of messages.
        """
        value = (
            kwargs.get(self.variable_name, [])
            if self.optional
            else kwargs[self.variable_name]
        )
        if not isinstance(value, list):
            msg = (
                f"variable {self.variable_name} should be a list of base messages, "
                f"got {value} of type {type(value)}"
            )
            raise ValueError(msg)  # noqa: TRY004
        value = convert_to_messages(value)
        if self.n_messages:
            value = value[-self.n_messages :]
        return value

    @property
    def input_variables(self) -> list[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        return [self.variable_name] if not self.optional else []

    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML. Defaults to False.

        Returns:
            Human-readable representation.
        """
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
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    @classmethod
    def from_template(
        cls: type[MessagePromptTemplateT],
        template: str,
        template_format: PromptTemplateFormat = "f-string",
        partial_variables: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> MessagePromptTemplateT:
        """Create a class from a string template.

        Args:
            template: a template.
            template_format: format of the template. Defaults to "f-string".
            partial_variables: A dictionary of variables that can be used to partially
                               fill in the template. For example, if the template is
                              `"{variable1} {variable2}"`, and `partial_variables` is
                              `{"variable1": "foo"}`, then the final prompt will be
                              `"foo {variable2}"`.
                              Defaults to None.
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
        cls: type[MessagePromptTemplateT],
        template_file: Union[str, Path],
        input_variables: list[str],
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
        """Async format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        return self.format(**kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> list[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        return self.prompt.input_variables

    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML. Defaults to False.

        Returns:
            Human-readable representation.
        """
        # TODO: Handle partials
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        return f"{title}\n\n{self.prompt.pretty_repr(html=html)}"


class ChatMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """Chat message prompt template."""

    role: str
    """Role of the message."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
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
        """Async format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        text = await self.prompt.aformat(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )


_StringImageMessagePromptTemplateT = TypeVar(
    "_StringImageMessagePromptTemplateT", bound="_StringImageMessagePromptTemplate"
)


class _TextTemplateParam(TypedDict, total=False):
    text: Union[str, dict]


class _ImageTemplateParam(TypedDict, total=False):
    image_url: Union[str, dict]


class _StringImageMessagePromptTemplate(BaseMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""

    prompt: Union[
        StringPromptTemplate, list[Union[StringPromptTemplate, ImagePromptTemplate]]
    ]
    """Prompt template."""
    additional_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the prompt template."""

    _msg_class: type[BaseMessage]

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    @classmethod
    def from_template(
        cls: type[_StringImageMessagePromptTemplateT],
        template: Union[str, list[Union[str, _TextTemplateParam, _ImageTemplateParam]]],
        template_format: PromptTemplateFormat = "f-string",
        *,
        partial_variables: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> _StringImageMessagePromptTemplateT:
        """Create a class from a string template.

        Args:
            template: a template.
            template_format: format of the template.
                Options are: 'f-string', 'mustache', 'jinja2'. Defaults to "f-string".
            partial_variables: A dictionary of variables that can be used too partially.
                Defaults to None.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.

        Raises:
            ValueError: If the template is not a string or list of strings.
        """
        if isinstance(template, str):
            prompt: Union[StringPromptTemplate, list] = PromptTemplate.from_template(
                template,
                template_format=template_format,
                partial_variables=partial_variables,
            )
            return cls(prompt=prompt, **kwargs)
        elif isinstance(template, list):
            if (partial_variables is not None) and len(partial_variables) > 0:
                msg = "Partial variables are not supported for list of templates."
                raise ValueError(msg)
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
                    input_variables = []
                    if isinstance(img_template, str):
                        vars = get_template_variables(img_template, template_format)
                        if vars:
                            if len(vars) > 1:
                                msg = (
                                    "Only one format variable allowed per image"
                                    f" template.\nGot: {vars}"
                                    f"\nFrom: {tmpl}"
                                )
                                raise ValueError(msg)
                            input_variables = [vars[0]]
                        img_template = {"url": img_template}
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    elif isinstance(img_template, dict):
                        img_template = dict(img_template)
                        for key in ["url", "path", "detail"]:
                            if key in img_template:
                                input_variables.extend(
                                    get_template_variables(
                                        img_template[key], template_format
                                    )
                                )
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    else:
                        msg = f"Invalid image template: {tmpl}"
                        raise ValueError(msg)
                    prompt.append(img_template_obj)
                else:
                    msg = f"Invalid template: {tmpl}"
                    raise ValueError(msg)
            return cls(prompt=prompt, **kwargs)
        else:
            msg = f"Invalid template: {template}"
            raise ValueError(msg)  # noqa: TRY004

    @classmethod
    def from_template_file(
        cls: type[_StringImageMessagePromptTemplateT],
        template_file: Union[str, Path],
        input_variables: list[str],
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
        with open(str(template_file)) as f:
            template = f.read()
        return cls.from_template(template, input_variables=input_variables, **kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> list[str]:
        """Input variables for this prompt template.

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
            content: list = []
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
        """Async format the prompt template.

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
            content: list = []
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
        """Human-readable representation.

        Args:
            html: Whether to format as HTML. Defaults to False.

        Returns:
            Human-readable representation.
        """
        # TODO: Handle partials
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        prompt_reprs = "\n\n".join(prompt.pretty_repr(html=html) for prompt in prompts)
        return f"{title}\n\n{prompt_reprs}"


class HumanMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""

    _msg_class: type[BaseMessage] = HumanMessage


class AIMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """AI message prompt template. This is a message sent from the AI."""

    _msg_class: type[BaseMessage] = AIMessage

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]


class SystemMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """System message prompt template.
    This is a message that is not sent to the user.
    """

    _msg_class: type[BaseMessage] = SystemMessage

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]


class BaseChatPromptTemplate(BasePromptTemplate, ABC):
    """Base class for chat prompt templates."""

    @property
    def lc_attributes(self) -> dict:
        """Return a list of attribute names that should be included in the
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
            formatted string.
        """
        return self.format_prompt(**kwargs).to_string()

    async def aformat(self, **kwargs: Any) -> str:
        """Async format the chat template into a string.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            formatted string.
        """
        return (await self.aformat_prompt(**kwargs)).to_string()

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format prompt. Should return a PromptValue.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            PromptValue.
        """
        messages = self.format_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async format prompt. Should return a PromptValue.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            PromptValue.
        """
        messages = await self.aformat_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format kwargs into a list of messages."""

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format kwargs into a list of messages."""
        return self.format_messages(**kwargs)

    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML. Defaults to False.

        Returns:
            Human-readable representation.
        """
        raise NotImplementedError

    def pretty_print(self) -> None:
        """Print a human-readable representation."""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


MessageLike = Union[BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate]

MessageLikeRepresentation = Union[
    MessageLike,
    tuple[
        Union[str, type],
        Union[str, list[dict], list[object]],
    ],
    str,
    dict,
]


class ChatPromptTemplate(BaseChatPromptTemplate):
    """Prompt template for chat models.

    Use to create flexible templated prompts for chat models.

    Examples:

        .. versionchanged:: 0.2.24

            You can pass any Message-like formats supported by
            ``ChatPromptTemplate.from_messages()`` directly to ``ChatPromptTemplate()``
            init.

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate([
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

            template = ChatPromptTemplate([
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

            template = ChatPromptTemplate([
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

    messages: Annotated[list[MessageLike], SkipValidation()]
    """List of messages consisting of either message prompt templates or messages."""
    validate_template: bool = False
    """Whether or not to try validating the template."""

    def __init__(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> None:
        """Create a chat prompt template from a variety of message formats.

        Args:
            messages: sequence of message representations.
                  A message can be represented using the following formats:
                  (1) BaseMessagePromptTemplate, (2) BaseMessage, (3) 2-tuple of
                  (message type, template); e.g., ("human", "{user_input}"),
                  (4) 2-tuple of (message class, template), (5) a string which is
                  shorthand for ("human", template); e.g., "{user_input}".
            template_format: format of the template. Defaults to "f-string".
            input_variables: A list of the names of the variables whose values are
                required as inputs to the prompt.
            optional_variables: A list of the names of the variables for placeholder
            or MessagePlaceholder that are optional. These variables are auto inferred
            from the prompt and user need not provide them.
            partial_variables: A dictionary of the partial variables the prompt
                template carries. Partial variables populate the template so that you
                don't need to pass them in every time you call the prompt.
            validate_template: Whether to validate the template.
            input_types: A dictionary of the types of the variables the prompt template
                expects. If not provided, all variables are assumed to be strings.

        Returns:
            A chat prompt template.

        Examples:
            Instantiation from a list of message templates:

            .. code-block:: python

                template = ChatPromptTemplate([
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "That's good to hear."),
                ])

            Instantiation from mixed message formats:

            .. code-block:: python

                template = ChatPromptTemplate([
                    SystemMessage(content="hello"),
                    ("human", "Hello, how are you?"),
                ])

        """
        _messages = [
            _convert_to_message(message, template_format) for message in messages
        ]

        # Automatically infer input variables from messages
        input_vars: set[str] = set()
        optional_variables: set[str] = set()
        partial_vars: dict[str, Any] = {}
        for _message in _messages:
            if isinstance(_message, MessagesPlaceholder) and _message.optional:
                partial_vars[_message.variable_name] = []
                optional_variables.add(_message.variable_name)
            elif isinstance(
                _message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)
            ):
                input_vars.update(_message.input_variables)

        kwargs = {
            "input_variables": sorted(input_vars),
            "optional_variables": sorted(optional_variables),
            "partial_variables": partial_vars,
            **kwargs,
        }
        cast(type[ChatPromptTemplate], super()).__init__(messages=_messages, **kwargs)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
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
            msg = f"Unsupported operand type for +: {type(other)}"
            raise NotImplementedError(msg)

    @model_validator(mode="before")
    @classmethod
    def validate_input_variables(cls, values: dict) -> Any:
        """Validate input variables.

        If input_variables is not set, it will be set to the union of
        all input variables in the messages.

        Args:
            values: values to validate.

        Returns:
            Validated values.

        Raises:
            ValueError: If input variables do not match.
        """
        messages = values["messages"]
        input_vars = set()
        optional_variables = set()
        input_types: dict[str, Any] = values.get("input_types", {})
        for message in messages:
            if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
                input_vars.update(message.input_variables)
            if isinstance(message, MessagesPlaceholder):
                if "partial_variables" not in values:
                    values["partial_variables"] = {}
                if (
                    message.optional
                    and message.variable_name not in values["partial_variables"]
                ):
                    values["partial_variables"][message.variable_name] = []
                    optional_variables.add(message.variable_name)
                if message.variable_name not in input_types:
                    input_types[message.variable_name] = list[AnyMessage]
        if "partial_variables" in values:
            input_vars = input_vars - set(values["partial_variables"])
        if optional_variables:
            input_vars = input_vars - optional_variables
        if "input_variables" in values and values.get("validate_template"):
            if input_vars != set(values["input_variables"]):
                msg = (
                    "Got mismatched input_variables. "
                    f"Expected: {input_vars}. "
                    f"Got: {values['input_variables']}"
                )
                raise ValueError(msg)
        else:
            values["input_variables"] = sorted(input_vars)
        if optional_variables:
            values["optional_variables"] = sorted(optional_variables)
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
        cls, string_messages: list[tuple[str, str]]
    ) -> ChatPromptTemplate:
        """Create a chat prompt template from a list of (role, template) tuples.

        Args:
            string_messages: list of (role, template) tuples.

        Returns:
            a chat prompt template.
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
        cls, string_messages: list[tuple[type[BaseMessagePromptTemplate], str]]
    ) -> ChatPromptTemplate:
        """Create a chat prompt template from a list of (role class, template) tuples.

        Args:
            string_messages: list of (role class, template) tuples.

        Returns:
            a chat prompt template.
        """
        return cls.from_messages(string_messages)

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[MessageLikeRepresentation],
        template_format: PromptTemplateFormat = "f-string",
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
                  (4) 2-tuple of (message class, template), (5) a string which is
                  shorthand for ("human", template); e.g., "{user_input}".
            template_format: format of the template. Defaults to "f-string".

        Returns:
            a chat prompt template.
        """
        return cls(messages, template_format=template_format)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format the chat template into a list of finalized messages.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            list of formatted messages.
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
                msg = f"Unexpected input: {message_template}"
                raise ValueError(msg)  # noqa: TRY004
        return result

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format the chat template into a list of finalized messages.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            list of formatted messages.

        Raises:
            ValueError: If unexpected input.
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
                msg = f"Unexpected input: {message_template}"
                raise ValueError(msg)  # noqa:TRY004
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
        """Append a message to the end of the chat template.

        Args:
            message: representation of a message to append.
        """
        self.messages.append(_convert_to_message(message))

    def extend(self, messages: Sequence[MessageLikeRepresentation]) -> None:
        """Extend the chat template with a sequence of messages.

        Args:
            messages: sequence of message representations to append.
        """
        self.messages.extend([_convert_to_message(message) for message in messages])

    @overload
    def __getitem__(self, index: int) -> MessageLike: ...

    @overload
    def __getitem__(self, index: slice) -> ChatPromptTemplate: ...

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
        """Name of prompt type. Used for serialization."""
        return "chat"

    def save(self, file_path: Union[Path, str]) -> None:
        """Save prompt to file.

        Args:
            file_path: path to file.
        """
        raise NotImplementedError

    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML. Defaults to False.

        Returns:
            Human-readable representation.
        """
        # TODO: handle partials
        return "\n\n".join(msg.pretty_repr(html=html) for msg in self.messages)


def _create_template_from_message_type(
    message_type: str,
    template: Union[str, list],
    template_format: PromptTemplateFormat = "f-string",
) -> BaseMessagePromptTemplate:
    """Create a message prompt template from a message type and template string.

    Args:
        message_type: str the type of the message template (e.g., "human", "ai", etc.)
        template: str the template string.
        template_format: format of the template. Defaults to "f-string".

    Returns:
        a message prompt template of the appropriate type.

    Raises:
        ValueError: If unexpected message type.
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
                msg = (
                    f"Invalid placeholder template: {template}."
                    " Expected a variable name surrounded by curly braces."
                )
                raise ValueError(msg)
            var_name = template[1:-1]
            message = MessagesPlaceholder(variable_name=var_name, optional=True)
        elif len(template) == 2 and isinstance(template[1], bool):
            var_name_wrapped, is_optional = template
            if not isinstance(var_name_wrapped, str):
                msg = f"Expected variable name to be a string. Got: {var_name_wrapped}"
                raise ValueError(msg)  # noqa:TRY004
            if var_name_wrapped[0] != "{" or var_name_wrapped[-1] != "}":
                msg = (
                    f"Invalid placeholder template: {var_name_wrapped}."
                    " Expected a variable name surrounded by curly braces."
                )
                raise ValueError(msg)
            var_name = var_name_wrapped[1:-1]

            message = MessagesPlaceholder(variable_name=var_name, optional=is_optional)
        else:
            msg = (
                "Unexpected arguments for placeholder message type."
                " Expected either a single string variable name"
                " or a list of [variable_name: str, is_optional: bool]."
                f" Got: {template}"
            )
            raise ValueError(msg)
    else:
        msg = (
            f"Unexpected message type: {message_type}. Use one of 'human',"
            f" 'user', 'ai', 'assistant', or 'system'."
        )
        raise ValueError(msg)
    return message


def _convert_to_message(
    message: MessageLikeRepresentation,
    template_format: PromptTemplateFormat = "f-string",
) -> Union[BaseMessage, BaseMessagePromptTemplate, BaseChatPromptTemplate]:
    """Instantiate a message from a variety of message formats.

    The message format can be one of the following:

    - BaseMessagePromptTemplate
    - BaseMessage
    - 2-tuple of (role string, template); e.g., ("human", "{user_input}")
    - 2-tuple of (message class, template)
    - string: shorthand for ("human", template); e.g., "{user_input}"

    Args:
        message: a representation of a message in one of the supported formats.
        template_format: format of the template. Defaults to "f-string".

    Returns:
        an instance of a message or a message template.

    Raises:
        ValueError: If unexpected message type.
        ValueError: If 2-tuple does not have 2 elements.
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
    elif isinstance(message, (tuple, dict)):
        if isinstance(message, dict):
            if set(message.keys()) != {"content", "role"}:
                msg = (
                    "Expected dict to have exact keys 'role' and 'content'."
                    f" Got: {message}"
                )
                raise ValueError(msg)
            message = (message["role"], message["content"])
        if len(message) != 2:
            msg = f"Expected 2-tuple of (role, template), got {message}"
            raise ValueError(msg)
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
        msg = f"Unsupported message type: {type(message)}"
        raise NotImplementedError(msg)

    return _message

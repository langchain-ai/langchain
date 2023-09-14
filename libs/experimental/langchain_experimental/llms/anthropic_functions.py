import json
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, DefaultDict, Dict, List, Optional

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain.chat_models.anthropic import ChatAnthropic
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    ChatGeneration,
    ChatResult,
    LLMResult,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)

from langchain_experimental.pydantic_v1 import root_validator

prompt = """In addition to responding, you can use tools. \
You have access to the following tools.

{tools}

In order to use a tool, you can use <tool></tool> to specify the name, \
and the <tool_input></tool_input> tags to specify the parameters. \
Each parameter should be passed in as <$param_name>$value</$param_name>, \
Where $param_name is the name of the specific parameter, and $value \
is the value for that parameter.

You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that accepts a single \
parameter 'query' that could run a google search, in order to search \
for the weather in SF you would respond:

<tool>search</tool><tool_input><query>weather in SF</query></tool_input>
<observation>64 degrees</observation>"""


class TagParser(HTMLParser):
    def __init__(self) -> None:
        """A heavy-handed solution, but it's fast for prototyping.

        Might be re-implemented later to restrict scope to the limited grammar, and
        more efficiency.

        Uses an HTML parser to parse a limited grammar that allows
        for syntax of the form:

            INPUT -> JUNK? VALUE*
            JUNK -> JUNK_CHARACTER+
            JUNK_CHARACTER -> whitespace | ,
            VALUE -> <IDENTIFIER>DATA</IDENTIFIER> | OBJECT
            OBJECT -> <IDENTIFIER>VALUE+</IDENTIFIER>
            IDENTIFIER -> [a-Z][a-Z0-9_]*
            DATA -> .*

        Interprets the data to allow repetition of tags and recursion
        to support representation of complex types.

        ^ Just another approximately wrong grammar specification.
        """
        super().__init__()

        self.parse_data: DefaultDict[str, List[Any]] = defaultdict(list)
        self.stack: List[DefaultDict[str, List[str]]] = [self.parse_data]
        self.success = True
        self.depth = 0
        self.data: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        """Hook when a new tag is encountered."""
        self.depth += 1
        self.stack.append(defaultdict(list))
        self.data = None

    def handle_endtag(self, tag: str) -> None:
        """Hook when a tag is closed."""
        self.depth -= 1
        top_of_stack = dict(self.stack.pop(-1))  # Pop the dictionary we don't need it

        # If a lead node
        is_leaf = self.data is not None
        # Annoying to type here, code is tested, hopefully OK
        value = self.data if is_leaf else top_of_stack
        # Difficult to type this correctly with mypy (maybe impossible?)
        # Can be nested indefinitely, so requires self referencing type
        self.stack[-1][tag].append(value)  # type: ignore
        # Reset the data so we if we encounter a sequence of end tags, we
        # don't confuse an outer end tag for belonging to a leaf node.
        self.data = None

    def handle_data(self, data: str) -> None:
        """Hook when handling data."""
        stripped_data = data.strip()
        # The only data that's allowed is whitespace or a comma surrounded by whitespace
        if self.depth == 0 and stripped_data not in (",", ""):
            # If this is triggered the parse should be considered invalid.
            self.success = False
        if stripped_data:  # ignore whitespace-only strings
            self.data = stripped_data


def _destrip(tool_input: Any) -> Any:
    if isinstance(tool_input, dict):
        return {k: _destrip(v) for k, v in tool_input.items()}
    elif isinstance(tool_input, list):
        if isinstance(tool_input[0], str):
            if len(tool_input) == 1:
                return tool_input[0]
            else:
                raise ValueError
        elif isinstance(tool_input[0], dict):
            return [_destrip(v) for v in tool_input]
        else:
            raise ValueError
    else:
        raise ValueError


class AnthropicFunctions(BaseChatModel):
    model: ChatAnthropic

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        return {"model": ChatAnthropic(**values)}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        forced = False
        function_call = ""
        if "functions" in kwargs:
            content = prompt.format(tools=json.dumps(kwargs["functions"], indent=2))
            system = SystemMessage(content=content)
            messages = [system] + messages
            del kwargs["functions"]
            if stop is None:
                stop = ["</tool_input>"]
            else:
                stop.append("</tool_input>")
            if "function_call" in kwargs:
                forced = True
                function_call = kwargs["function_call"]["name"]
                AIMessage(content=f"<tool>{function_call}</tool>")
                del kwargs["function_call"]
        else:
            if "function_call" in kwargs:
                raise ValueError(
                    "if `function_call` provided, `functions` must also be"
                )
        response = self.model.predict_messages(
            messages, stop=stop, callbacks=run_manager, **kwargs
        )
        completion = response.content
        if forced:
            tag_parser = TagParser()
            tag_parser.feed(completion.strip() + "</tool_input>")
            v1 = tag_parser.parse_data["tool_input"][0]
            kwargs = {
                "function_call": {
                    "name": function_call,
                    "arguments": json.dumps(_destrip(v1)),
                }
            }
            message = AIMessage(content="", additional_kwargs=kwargs)
            return ChatResult(generations=[ChatGeneration(message=message)])
        elif "<tool>" in completion:
            tag_parser = TagParser()
            tag_parser.feed(completion.strip() + "</tool_input>")
            msg = completion.split("<tool>")[0]
            v1 = tag_parser.parse_data["tool_input"][0]
            kwargs = {
                "function_call": {
                    "name": tag_parser.parse_data["tool"][0],
                    "arguments": json.dumps(_destrip(v1)),
                }
            }
            message = AIMessage(content=msg, additional_kwargs=kwargs)
            return ChatResult(generations=[ChatGeneration(message=message)])
        else:
            return ChatResult(generations=[ChatGeneration(message=response)])

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        raise NotImplementedError

    @property
    def _llm_type(self) -> str:
        return "anthropic_functions"

import re
import xml.etree.ElementTree as ET
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables.utils import AddableDict

XML_FORMAT_INSTRUCTIONS = """The output should be formatted as a XML file.
1. Output should conform to the tags below. 
2. If tags are not given, make them on your own.
3. Remember to always open and close all the tags.

As an example, for the tags ["foo", "bar", "baz"]:
1. String "<foo>\n   <bar>\n      <baz></baz>\n   </bar>\n</foo>" is a well-formatted instance of the schema. 
2. String "<foo>\n   <bar>\n   </foo>" is a badly-formatted instance.
3. String "<foo>\n   <tag>\n   </tag>\n</foo>" is a badly-formatted instance.

Here are the output tags:
```
{tags}
```"""  # noqa: E501


class XMLOutputParser(BaseTransformOutputParser):
    """Parse an output using xml format."""

    tags: Optional[List[str]] = None
    encoding_matcher: re.Pattern = re.compile(
        r"<([^>]*encoding[^>]*)>\n(.*)", re.MULTILINE | re.DOTALL
    )

    def get_format_instructions(self) -> str:
        return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)

    def parse(self, text: str) -> Dict[str, List[Any]]:
        # Try to find XML string within triple backticks
        match = re.search(r"```(xml)?(.*)```", text, re.DOTALL)
        if match is not None:
            # If match found, use the content within the backticks
            text = match.group(2)
        encoding_match = self.encoding_matcher.search(text)
        if encoding_match:
            text = encoding_match.group(2)

        text = text.strip()
        if (text.startswith("<") or text.startswith("\n<")) and (
            text.endswith(">") or text.endswith(">\n")
        ):
            root = ET.fromstring(text)
            return self._root_to_dict(root)
        else:
            raise ValueError(f"Could not parse output: {text}")

    def _transform(
        self, input: Iterator[Union[str, BaseMessage]]
    ) -> Iterator[AddableDict]:
        xml_start_re = re.compile(r"<[a-zA-Z:_]")
        parser = ET.XMLPullParser(["start", "end"])
        xml_started = False
        current_path: List[str] = []
        current_path_has_children = False
        buffer = ""
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                # extract text
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                chunk = chunk_content
            # add chunk to buffer of unprocessed text
            buffer += chunk
            # if xml string hasn't started yet, continue to next chunk
            if not xml_started:
                if match := xml_start_re.search(buffer):
                    # if xml string has started, remove all text before it
                    buffer = buffer[match.start() :]
                    xml_started = True
                else:
                    continue
            # feed buffer to parser
            parser.feed(buffer)
            buffer = ""
            # yield all events
            for event, elem in parser.read_events():
                if event == "start":
                    # update current path
                    current_path.append(elem.tag)
                    current_path_has_children = False
                elif event == "end":
                    # remove last element from current path
                    current_path.pop()
                    # yield element
                    if not current_path_has_children:
                        yield nested_element(current_path, elem)
                    # prevent yielding of parent element
                    if current_path:
                        current_path_has_children = True
                    else:
                        xml_started = False
        # close parser
        parser.close()

    async def _atransform(
        self, input: AsyncIterator[Union[str, BaseMessage]]
    ) -> AsyncIterator[AddableDict]:
        parser = ET.XMLPullParser(["start", "end"])
        current_path: List[str] = []
        current_path_has_children = False
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                # extract text
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                chunk = chunk_content
            # pass chunk to parser
            parser.feed(chunk)
            # yield all events
            for event, elem in parser.read_events():
                if event == "start":
                    # update current path
                    current_path.append(elem.tag)
                    current_path_has_children = False
                elif event == "end":
                    # remove last element from current path
                    current_path.pop()
                    # yield element
                    if not current_path_has_children:
                        yield nested_element(current_path, elem)
                    # prevent yielding of parent element
                    current_path_has_children = True
        # close parser
        parser.close()

    def _root_to_dict(self, root: ET.Element) -> Dict[str, List[Any]]:
        """Converts xml tree to python dictionary."""
        result: Dict[str, List[Any]] = {root.tag: []}
        for child in root:
            if len(child) == 0:
                result[root.tag].append({child.tag: child.text})
            else:
                result[root.tag].append(self._root_to_dict(child))
        return result

    @property
    def _type(self) -> str:
        return "xml"


def nested_element(path: List[str], elem: ET.Element) -> Any:
    """Get nested element from path."""
    if len(path) == 0:
        return AddableDict({elem.tag: elem.text})
    else:
        return AddableDict({path[0]: [nested_element(path[1:], elem)]})

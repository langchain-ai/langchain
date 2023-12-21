import re
import xml.etree.ElementTree as ET
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables.utils import AddableDict

from langchain.output_parsers.format_instructions import XML_FORMAT_INSTRUCTIONS


class XMLOutputParser(BaseTransformOutputParser):
    """Parse an output using xml format."""

    tags: Optional[List[str]] = None
    encoding_matcher: re.Pattern = re.compile(
        r"<([^>]*encoding[^>]*)>\n(.*)", re.MULTILINE | re.DOTALL
    )

    def get_format_instructions(self) -> str:
        return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)

    def parse(self, text: str) -> Dict[str, List[Any]]:
        text = text.strip("`").strip("xml")
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
        parser = ET.XMLPullParser(["start", "end"])
        current_path: List[str] = []
        current_path_has_children = False
        for chunk in input:
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

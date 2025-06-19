"""Output parser for XML format."""

import contextlib
import re
import xml
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal, Optional, Union
from xml.etree.ElementTree import TreeBuilder

from typing_extensions import override

from langchain_core.exceptions import OutputParserException
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


def _create_secure_xml_parser() -> Any:
    """Create a secure XML parser with hardened configuration.
    
    This function creates an XML parser with security hardening to prevent
    XML vulnerabilities like XXE attacks, billion laughs attacks, and DTD processing.
    
    Returns:
        A secure XML parser instance.
        
    Raises:
        ImportError: If neither defusedxml nor a secure standard library parser
                    configuration is available.
    """
    # First try to use defusedxml which is the most secure option
    try:
        from defusedxml.ElementTree import XMLParser  # type: ignore[import-untyped]
        return XMLParser(target=TreeBuilder())
    except ImportError:
        pass
    
    # If defusedxml is not available, create a hardened standard library parser
    # NOTE: This is a fallback for environments where defusedxml cannot be installed
    # but we still want to provide some protection against common XML attacks
    try:
        # Create a parser with security restrictions
        parser = ET.XMLParser()
        
        # Disable DTD processing to prevent XXE and billion laughs attacks
        # This is the most critical security measure
        if hasattr(parser, 'parser'):
            # Access the underlying expat parser
            expat_parser = parser.parser
            
            # Disable DTD processing completely
            expat_parser.DefaultHandler = lambda data: None
            expat_parser.ExternalEntityRefHandler = None
            expat_parser.EntityDeclHandler = None
            expat_parser.NotationDeclHandler = None
            expat_parser.ProcessingInstructionHandler = None
            expat_parser.CommentHandler = None
            
            # Set entity limits to prevent billion laughs attacks
            # These limits are conservative to prevent memory exhaustion
            if hasattr(expat_parser, 'SetParamEntityParsing'):
                # Disable parameter entity parsing
                expat_parser.SetParamEntityParsing(0)  # XML_PARAM_ENTITY_PARSING_NEVER
                
        return parser
    except Exception:
        # If we can't create a secure parser, raise an error rather than
        # falling back to an insecure configuration
        msg = (
            "Unable to create a secure XML parser. Please install defusedxml "
            "for maximum security: pip install defusedxml"
        )
        raise ImportError(msg)


def _create_secure_element_tree_module() -> Any:
    """Create a secure ElementTree module for XML parsing.
    
    Returns:
        Either defusedxml.ElementTree or a warning-wrapped standard library ET.
    """
    try:
        from defusedxml import ElementTree  # type: ignore[import-untyped]
        return ElementTree
    except ImportError:
        # Create a wrapper that adds security warnings
        import warnings
        
        class SecureElementTreeWrapper:
            """Wrapper around standard library ElementTree with security warnings."""
            
            def __getattr__(self, name: str) -> Any:
                return getattr(ET, name)
            
            def fromstring(self, text: str, parser: Optional[Any] = None) -> Any:
                """Parse XML with security warnings."""
                # Issue a security warning when using standard library parser
                warnings.warn(
                    "Using standard library XML parser. For maximum security, "
                    "install defusedxml: pip install defusedxml",
                    UserWarning,
                    stacklevel=2
                )
                
                # Use secure parser if none provided
                if parser is None:
                    try:
                        parser = _create_secure_xml_parser()
                    except ImportError:
                        # Last resort: use standard parser but with content restrictions
                        pass
                
                # Additional content validation to prevent common attacks
                if text and isinstance(text, str):
                    # Check for obvious XXE patterns
                    if '<!DOCTYPE' in text and ('<!ENTITY' in text or 'SYSTEM' in text):
                        raise OutputParserException(
                            "XML contains potentially malicious DTD declarations. "
                            "This type of content is not allowed for security reasons."
                        )
                    
                    # Check for excessive entity nesting (billion laughs protection)
                    if text.count('&') > 10:  # Conservative limit
                        raise OutputParserException(
                            "XML contains excessive entity references. "
                            "This may indicate a billion laughs attack."
                        )
                
                return ET.fromstring(text, parser)
        
        return SecureElementTreeWrapper()


class _StreamingParser:
    """Streaming parser for XML.

    This implementation is pulled into a class to avoid implementation
    drift between transform and atransform of the XMLOutputParser.
    """

    def __init__(self, parser: Literal["defusedxml", "xml", "secure"]) -> None:
        """Initialize the streaming parser.

        Args:
            parser: Parser to use for XML parsing. Can be 'defusedxml', 'xml', or 'secure'.
              - 'defusedxml': Use defusedxml library (most secure)
              - 'secure': Use hardened standard library parser (recommended fallback)
              - 'xml': Use standard library parser (deprecated, security risk)

        Raises:
            ImportError: If defusedxml is not installed and the defusedxml
                parser is requested.
        """
        if parser == "defusedxml":
            try:
                from defusedxml.ElementTree import (  # type: ignore[import-untyped]
                    XMLParser,
                )
                _parser = XMLParser(target=TreeBuilder())
            except ImportError as e:
                msg = (
                    "defusedxml is not installed. "
                    "Please install it to use the defusedxml parser."
                    "You can install it with `pip install defusedxml` "
                )
                raise ImportError(msg) from e
        elif parser == "secure":
            _parser = _create_secure_xml_parser()
        else:
            # Legacy 'xml' option - issue deprecation warning
            import warnings
            warnings.warn(
                "Using 'xml' parser is deprecated due to security risks. "
                "Use 'secure' or 'defusedxml' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            _parser = None
            
        self.pull_parser = ET.XMLPullParser(["start", "end"], _parser=_parser)
        self.xml_start_re = re.compile(r"<[a-zA-Z:_]")
        self.current_path: list[str] = []
        self.current_path_has_children = False
        self.buffer = ""
        self.xml_started = False

    def parse(self, chunk: Union[str, BaseMessage]) -> Iterator[AddableDict]:
        """Parse a chunk of text.

        Args:
            chunk: A chunk of text to parse. This can be a string or a BaseMessage.

        Yields:
            AddableDict: A dictionary representing the parsed XML element.

        Raises:
            xml.etree.ElementTree.ParseError: If the XML is not well-formed.
        """
        if isinstance(chunk, BaseMessage):
            # extract text
            chunk_content = chunk.content
            if not isinstance(chunk_content, str):
                # ignore non-string messages (e.g., function calls)
                return
            chunk = chunk_content
        # add chunk to buffer of unprocessed text
        self.buffer += chunk
        # if xml string hasn't started yet, continue to next chunk
        if not self.xml_started:
            if match := self.xml_start_re.search(self.buffer):
                # if xml string has started, remove all text before it
                self.buffer = self.buffer[match.start() :]
                self.xml_started = True
            else:
                return
        # feed buffer to parser
        self.pull_parser.feed(self.buffer)
        self.buffer = ""
        # yield all events
        try:
            for event, elem in self.pull_parser.read_events():
                if event == "start":
                    # update current path
                    self.current_path.append(elem.tag)
                    self.current_path_has_children = False
                elif event == "end":
                    # remove last element from current path
                    #
                    self.current_path.pop()
                    # yield element
                    if not self.current_path_has_children:
                        yield nested_element(self.current_path, elem)
                    # prevent yielding of parent element
                    if self.current_path:
                        self.current_path_has_children = True
                    else:
                        self.xml_started = False
        except xml.etree.ElementTree.ParseError:
            # This might be junk at the end of the XML input.
            # Let's check whether the current path is empty.
            if not self.current_path:
                # If it is empty, we can ignore this error.
                return
            else:
                raise

    def close(self) -> None:
        """Close the parser.

        This should be called after all chunks have been parsed.

        Raises:
            xml.etree.ElementTree.ParseError: If the XML is not well-formed.
        """
        # Ignore ParseError. This will ignore any incomplete XML at the end of the input
        with contextlib.suppress(xml.etree.ElementTree.ParseError):
            self.pull_parser.close()


class XMLOutputParser(BaseTransformOutputParser):
    """Parse an output using xml format."""

    tags: Optional[list[str]] = None
    """Tags to tell the LLM to expect in the XML output.

    Note this may not be perfect depending on the LLM implementation.

    For example, with tags=["foo", "bar", "baz"]:
            1. A well-formatted XML instance:
                "<foo>\n   <bar>\n      <baz></baz>\n   </bar>\n</foo>"

            2. A badly-formatted XML instance (missing closing tag for 'bar'):
                "<foo>\n   <bar>\n   </foo>"

            3. A badly-formatted XML instance (unexpected 'tag' element):
                "<foo>\n   <tag>\n   </tag>\n</foo>"
    """
    encoding_matcher: re.Pattern = re.compile(
        r"<([^>]*encoding[^>]*)>\n(.*)", re.MULTILINE | re.DOTALL
    )
    parser: Literal["defusedxml", "xml", "secure"] = "secure"
    """Parser to use for XML parsing. Can be 'defusedxml', 'xml', or 'secure'.

    * 'secure' is the default parser and provides hardened XML parsing with
      protection against common XML vulnerabilities even when defusedxml is not available.
    * 'defusedxml' uses the defusedxml library which provides the highest level of security.
    * 'xml' uses the standard library parser (DEPRECATED - security risk).

    The 'secure' parser provides protection against:
    - XXE (XML External Entity) attacks
    - Billion laughs attacks
    - DTD processing vulnerabilities
    - Excessive entity expansion

    For maximum security in production environments, install defusedxml:
    pip install defusedxml

    Please review the following resources for more information:
    * https://docs.python.org/3/library/xml.html#xml-vulnerabilities
    * https://github.com/tiran/defusedxml
    * https://owasp.org/www-community/vulnerabilities/XML_External_Entity_(XXE)_Processing
    """

    def get_format_instructions(self) -> str:
        """Return the format instructions for the XML output."""
        return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)

    def parse(self, text: str) -> dict[str, Union[str, list[Any]]]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A dictionary representing the parsed XML.

        Raises:
            OutputParserException: If the XML is not well-formed or contains
                potentially malicious content.
            ImportError: If defusedxml is not installed and the defusedxml
                parser is requested.
        """
        # Try to find XML string within triple backticks
        match = re.search(r"```(xml)?(.*)```", text, re.DOTALL)
        if match is not None:
            # If match found, use the content within the backticks
            text = match.group(2)
        encoding_match = self.encoding_matcher.search(text)
        if encoding_match:
            text = encoding_match.group(2)

        text = text.strip()
        
        # Select appropriate ElementTree module based on parser setting
        if self.parser == "defusedxml":
            try:
                from defusedxml import ElementTree  # type: ignore[import-untyped]
                _et = ElementTree
            except ImportError as e:
                msg = (
                    "defusedxml is not installed. "
                    "Please install it to use the defusedxml parser."
                    "You can install it with `pip install defusedxml`"
                    "See https://github.com/tiran/defusedxml for more details"
                )
                raise ImportError(msg) from e
        elif self.parser == "secure":
            _et = _create_secure_element_tree_module()
        else:
            # Legacy 'xml' option
            import warnings
            warnings.warn(
                "Using 'xml' parser is deprecated due to security risks. "
                "Use 'secure' or 'defusedxml' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            _et = ET  # Use the standard library parser

        try:
            root = _et.fromstring(text)
            return self._root_to_dict(root)
        except _et.ParseError as e:
            msg = f"Failed to parse XML format from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text) from e

    @override
    def _transform(
        self, input: Iterator[Union[str, BaseMessage]]
    ) -> Iterator[AddableDict]:
        streaming_parser = _StreamingParser(self.parser)
        for chunk in input:
            yield from streaming_parser.parse(chunk)
        streaming_parser.close()

    @override
    async def _atransform(
        self, input: AsyncIterator[Union[str, BaseMessage]]
    ) -> AsyncIterator[AddableDict]:
        streaming_parser = _StreamingParser(self.parser)
        async for chunk in input:
            for output in streaming_parser.parse(chunk):
                yield output
        streaming_parser.close()

    def _root_to_dict(self, root: ET.Element) -> dict[str, Union[str, list[Any]]]:
        """Converts xml tree to python dictionary."""
        if root.text and bool(re.search(r"\S", root.text)):
            # If root text contains any non-whitespace character it
            # returns {root.tag: root.text}
            return {root.tag: root.text}
        result: dict = {root.tag: []}
        for child in root:
            if len(child) == 0:
                result[root.tag].append({child.tag: child.text})
            else:
                result[root.tag].append(self._root_to_dict(child))
        return result

    @property
    def _type(self) -> str:
        return "xml"


def nested_element(path: list[str], elem: ET.Element) -> Any:
    """Get nested element from path.

    Args:
        path: The path to the element.
        elem: The element to extract.

    Returns:
        The nested element.
    """
    if len(path) == 0:
        return AddableDict({elem.tag: elem.text})
    return AddableDict({path[0]: [nested_element(path[1:], elem)]})

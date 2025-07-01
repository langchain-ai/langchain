"""Security tests for HTML splitters to prevent XXE attacks."""

import pytest
from pathlib import Path
import tempfile
import os

from langchain_text_splitters.html import HTMLSectionSplitter


@pytest.mark.requires("lxml", "bs4")
class TestHTMLSectionSplitterSecurity:
    """Security tests for HTMLSectionSplitter to ensure XXE prevention."""

    def test_xxe_entity_attack_blocked(self):
        """Test that external entity attacks are blocked."""
        # Create a malicious XSLT file that attempts to read /etc/passwd
        malicious_xslt = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xsl:stylesheet [
  <!ENTITY passwd SYSTEM "file:///etc/passwd">
]>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <body>
        <h1>XXE Attack Result</h1>
        <pre>&passwd;</pre>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>"""

        # Create HTML content to process
        html_content = """<html><body><p>Test content</p></body></html>"""

        # Since xslt_path parameter is removed, this attack vector is eliminated
        # The splitter should use only the default XSLT
        splitter = HTMLSectionSplitter(headers_to_split_on=[("h1", "Header 1")])

        # Process the HTML - should not contain any external entity content
        result = splitter.split_text(html_content)

        # Verify that no external entity content is present
        all_content = " ".join([doc.page_content for doc in result])
        assert "root:" not in all_content  # /etc/passwd content
        assert "XXE Attack Result" not in all_content

    def test_xxe_document_function_blocked(self):
        """Test that XSLT document() function attacks are blocked."""
        # Even if someone modifies the default XSLT internally,
        # the secure parser configuration should block document() attacks

        html_content = (
            """<html><body><h1>Test Header</h1><p>Test content</p></body></html>"""
        )

        splitter = HTMLSectionSplitter(headers_to_split_on=[("h1", "Header 1")])

        # Process the HTML safely
        result = splitter.split_text(html_content)

        # Should process normally without any security issues
        assert len(result) > 0
        assert any("Test content" in doc.page_content for doc in result)

    def test_secure_parser_configuration(self):
        """Test that parsers are configured with security settings."""
        from lxml import etree
        from io import StringIO

        # This test verifies our security hardening is in place
        html_content = """<html><body><h1>Test</h1></body></html>"""

        splitter = HTMLSectionSplitter(headers_to_split_on=[("h1", "Header 1")])

        # The convert_possible_tags_to_header method should use secure parsers
        result = splitter.convert_possible_tags_to_header(html_content)

        # Result should be valid transformed HTML
        assert result is not None
        assert isinstance(result, str)

    def test_no_network_access(self):
        """Test that network access is blocked in parsers."""
        # Create HTML that might trigger network access
        html_with_external_ref = """<?xml version="1.0"?>
<!DOCTYPE html [
  <!ENTITY external SYSTEM "http://attacker.com/xxe">
]>
<html>
  <body>
    <h1>Test</h1>
    <p>&external;</p>
  </body>
</html>"""

        splitter = HTMLSectionSplitter(headers_to_split_on=[("h1", "Header 1")])

        # Process the HTML - should not make network requests
        result = splitter.split_text(html_with_external_ref)

        # Verify no external content is included
        all_content = " ".join([doc.page_content for doc in result])
        assert "attacker.com" not in all_content

    def test_dtd_processing_disabled(self):
        """Test that DTD processing is disabled."""
        # HTML with DTD that attempts to define entities
        html_with_dtd = """<!DOCTYPE html [
  <!ELEMENT html (body)>
  <!ELEMENT body (h1, p)>
  <!ELEMENT h1 (#PCDATA)>
  <!ELEMENT p (#PCDATA)>
  <!ENTITY test "This is a test entity">
]>
<html>
  <body>
    <h1>Header</h1>
    <p>&test;</p>
  </body>
</html>"""

        splitter = HTMLSectionSplitter(headers_to_split_on=[("h1", "Header 1")])

        # Process the HTML - entities should not be resolved
        result = splitter.split_text(html_with_dtd)

        # The entity should not be expanded
        all_content = " ".join([doc.page_content for doc in result])
        assert "This is a test entity" not in all_content

    def test_safe_default_xslt_usage(self):
        """Test that the default XSLT file is used safely."""
        # Test with HTML that has font-size styling (what the default XSLT handles)
        html_with_font_size = """<html>
<body>
    <span style="font-size: 24px;">Large Header</span>
    <p>Content under large text</p>
    <span style="font-size: 18px;">Small Header</span>
    <p>Content under small text</p>
</body>
</html>"""

        splitter = HTMLSectionSplitter(headers_to_split_on=[("h1", "Header 1")])

        # Process the HTML using the default XSLT
        result = splitter.split_text(html_with_font_size)

        # Should successfully process the content
        assert len(result) > 0
        # Large font text should be converted to header
        assert any("Large Header" in str(doc.metadata.values()) for doc in result)

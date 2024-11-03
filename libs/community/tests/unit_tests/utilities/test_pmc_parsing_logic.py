import unittest
from unittest.mock import patch, MagicMock
from langchain_community.utilities.parsing_logic import PubMed_Central_Parser  
import pytest
import os
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@pytest.mark.requires("lxml")
class TestPubMedCentralParser(unittest.TestCase):
    """test the parsing logic for PubMed Central paper in xml form"""
    
    def setUp(self):
        try:
            from lxml import etree
        except ImportError as e:
            logger.error(
               "Could not import `lxml`. Please install it with `pip install lxml`. "
            )
            raise e
        self.parser = PubMed_Central_Parser()
        self.xml_string = '''
        <root>
            <article>
                <front>
                    <article-meta>
                        <title-group>
                            <article-title>Article Title</article-title>
                            <alt-title>Alternative Title</alt-title>
                        </title-group>
                        <abstract>
                            <p>Abstract paragraph 1.</p>
                            <p>Abstract paragraph 2.</p>
                        </abstract>
                    </article-meta>
                </front>
                <body>
                    <fig>
                        <caption>Figure 1 caption.</caption>
                    </fig>
                    <table-wrap>
                        <caption>Table 1 caption.</caption>
                    </table-wrap>
                    <p>Body paragraph 1.</p>
                </body>
                <sub-article>
                    <body>
                        <p>Subarticle body paragraph.</p>
                    </body>
                </sub-article>
                <footer>
                    <p>Footer paragraph.</p>
                </footer>
            </article>
        </root>
        '''
        self.tree = etree.fromstring(self.xml_string.encode("utf-8"))

    @patch('requests.get')
    def test_get_xml(self, mock_get):
        """Test get_xml method with both successful and error cases"""
        # Setup mock response for successful case
        mock_successful_response = MagicMock()
        mock_successful_response.status_code = 200
        mock_successful_response.content = self.xml_string.encode('utf-8')
        
        # Setup mock response for error case
        mock_error_response = MagicMock()
        mock_error_response.status_code = 404
        
        # Test successful case
        mock_get.return_value = mock_successful_response
        result = self.parser.get_xml('PMC9771812')
        self.assertEqual(result, self.xml_string)
        
        # Verify PMC prefix handling
        result_without_prefix = self.parser.get_xml('9771812')
        self.assertEqual(result_without_prefix, self.xml_string)
        
        # Test error case
        mock_get.return_value = mock_error_response
        result = self.parser.get_xml('PMC9771812')
        self.assertIsNone(result)
        
        # Verify the request parameters
        expected_params = {
            'verb': 'GetRecord',
            'identifier': 'oai:pubmedcentral.nih.gov:9771812',
            'metadataPrefix': 'pmc'
        }
        mock_get.assert_called_with(self.parser.pmc_url, params=expected_params)

    def test_extract_text(self):
        """test if article text can be extracted"""
        expected_output = 'Article Title\nAlternative Title\nAbstract paragraph 1.\nAbstract paragraph 2.\nBody paragraph 1.\nSubarticle body paragraph.'
        text = self.parser.extract_text(self.xml_string).rstrip()
        self.assertEqual(text, expected_output)

    def test_extract_paragraphs(self):
        """Verify that extract_paragraphs correctly parses and returns the expected list of paragraphs."""
        expected_output = [
            'Article Title',
            'Alternative Title',
            'Abstract paragraph 1.',
            'Abstract paragraph 2.',
            'Body paragraph 1.',
            'Subarticle body paragraph.'
        ]

        output = self.parser.extract_paragraphs(self.xml_string)
        self.assertEqual(output, expected_output)

    def test_select_from_top_level(self):
        """Verify correctly retrieves elements from top"""
        front_elements = self.parser._select_from_top_level(self.tree, 'front')
        self.assertEqual(len(front_elements), 1)
        self.assertEqual(front_elements[0].xpath("./article-meta/title-group/article-title")[0].text, 'Article Title')

        body_elements = self.parser._select_from_top_level(self.tree, 'body')
        self.assertEqual(len(body_elements), 1)
        self.assertEqual(body_elements[0].xpath("./p")[0].text, 'Body paragraph 1.')

        sub_article_elements = self.parser._select_from_top_level(self.tree, 'sub-article')
        self.assertEqual(len(sub_article_elements), 1)
        self.assertEqual(sub_article_elements[0].xpath("./body/p")[0].text, 'Subarticle body paragraph.')

    def test_extract_from_front(self):
        """Ensure correctly extracts the title and abstract paragraphs from the 'front' element."""
        front_element = self.tree.find(".//front")
        expected_output = [
            'Article Title',
            'Alternative Title',
            'Abstract paragraph 1.',
            'Abstract paragraph 2.'
        ]
        result = self.parser._extract_from_front(front_element)
        self.assertEqual(result, expected_output)

    def test_extract_from_body(self):
        body_elements = self.tree.findall(".//body")
        expected_output = [
            'Body paragraph 1.',
            'Subarticle body paragraph.'
        ]
        result = []
        for body_element in body_elements:
            result.extend(self.parser._extract_from_body(body_element))
        self.assertEqual(result, expected_output)

    def test_extract_from_subarticle(self):
        """Extract sub-article elements"""
        subarticle_elements = self.tree.xpath(".//sub-article")
        # Expected output
        expected_output = [
            'Subarticle body paragraph.'
        ]
        result = []
        for subarticle_element in subarticle_elements:
            result.extend(self.parser._extract_from_subarticle(subarticle_element))
        
        self.assertEqual(result, expected_output)

    def test_remove_elements_by_tag(self):
        """test if tag can be removed"""
        tags_to_remove = ['footer']
        self.parser._remove_elements_by_tag(self.tree, *tags_to_remove)

        for tag in tags_to_remove:
            elements = self.tree.xpath(f".//{tag}")
            self.assertFalse(elements, f"Element with tag '{tag}' should be removed but still exists.")

    def test_replace_unwanted_elements_with_their_captions(self):
        """test if unwanted elements with their captions are replaced"""
        self.parser._replace_unwanted_elements_with_their_captions(self.tree)
        
        # Check for unwanted elements
        unwanted_elements = self.tree.xpath(".//*[@position='float'] | .//fig | .//table-wrap")
        captions_elements = self.tree.xpath(".//captions")
        
        # Assert unwanted elements are removed
        self.assertEqual(len(unwanted_elements), 0, "Unwanted elements are still present")
        self.assertGreater(len(captions_elements), 0, "Captions elements are missing")

    def test_retain_only_pars(self):
        """test if the method can retain only 'p' tags and change 'title' tags to 'p'"""
        self.parser._retain_only_pars(self.tree)

        # Check if 'title-group' is still in the tree
        title_group_elements = self.tree.xpath(".//title-group")
        self.assertEqual(len(title_group_elements), 0, "title-group element should be removed")

    def test_pull_nested_paragraphs_to_top(self):
        try:
            from lxml import etree
        except ImportError as e:
            logger.error(
               "Could not import `lxml`. Please install it with `pip install lxml`. "
            )
            raise e
        """test if nested paragraphs be flatten"""
        xml_string = '''
        <root>
            <p>Paragraph 1
                <p>Nested Paragraph 1.1</p>
                <p>Nested Paragraph 1.2
                    <p>Deeply Nested Paragraph 1.2.1</p>
                </p>
            </p>
            <p>Paragraph 2</p>
        </root>
        '''
        tree = etree.fromstring(xml_string.encode("utf-8"))
        self.parser._pull_nested_paragraphs_to_top(tree)
        expected_contents = {
            "Paragraph 1",
            "Nested Paragraph 1.1",
            "Nested Paragraph 1.2",
            "Deeply Nested Paragraph 1.2.1",
            "Paragraph 2"
        }
        paragraphs = tree.xpath("//p")
        actual_contents = set(p.text.strip() if p.text else '' for p in paragraphs)
        self.assertEqual(actual_contents, expected_contents, "All expected paragraph contents should be present")

        # Check that there are no nested paragraphs
        nested_paragraphs = tree.xpath("./p/p")
        self.assertEqual(len(nested_paragraphs), 0, "There should be no nested paragraphs")

    def test_extract_paragraphs_from_tree(self):
        """test if paragraphs in article can be all extracted"""
        try:
            from lxml import etree
        except ImportError as e:
            logger.error(
               "Could not import `lxml`. Please install it with `pip install lxml`. "
            )
            raise e
        # Create an XML tree for the test
        xml_string = """
        <article>
            <body>
                <p>This is the first paragraph.</p>
                <p>This is the second paragraph.</p>
            </body>
        </article>
        """
        tree = etree.fromstring(xml_string)
        # Call the method
        paragraphs = self.parser._extract_paragraphs_from_tree(tree)

        # Assert the paragraphs are extracted correctly
        expected_paragraphs = [
            "This is the first paragraph.",
            "This is the second paragraph."
        ]
        self.assertEqual(paragraphs, expected_paragraphs)

    def test_xpath_union(self):
        """test if xpath expressions can be unioned """
        self.assertEqual(self.parser._xpath_union("//p"), "//p")
        self.assertEqual(
            self.parser._xpath_union("//p", "//div", "//span"),
            "//p | //div | //span"
        )


if __name__ == '__main__':
    unittest.main()

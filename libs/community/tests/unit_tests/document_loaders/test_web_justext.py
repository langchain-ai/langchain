from typing import List
from langchain_community.document_loaders.web_justext import JustextWebLoader
import pytest
import responses
import justext
import requests

# Test content constants
ENGLISH_HTML = """
<html lang="en">
    <head>
        <title>Important Research Findings</title>
        <meta name="description" content="Research on renewable energy technology and efficiency.">
    </head>
    <body>
        <main>
            <article>
                <h1>Important Research Findings</h1>
                <p>Recent studies have shown significant advances in renewable energy technology. 
                Scientists at leading research institutions have demonstrated improved efficiency 
                in solar cell design. The new approach combines novel materials with innovative 
                manufacturing processes. These developments could revolutionize the renewable 
                energy sector in coming years.</p>
                
                <p>Further experiments conducted across multiple laboratories confirmed these results. 
                The research teams documented consistent improvements in energy conversion rates. 
                Independent verification has strengthened confidence in these findings. Industry 
                experts are now evaluating potential commercial applications.</p>
            </article>
        </main>
    </body>
</html>
"""

GERMAN_HTML = """
<html lang="de">
    <head>
        <title>Wichtige Forschungsergebnisse</title>
        <meta name="description" content="Forschung zu erneuerbaren Energien und Effizienz.">
    </head>
    <body>
        <main>
            <article>
                <h1>Wichtige Forschungsergebnisse</h1>
                <p>Aktuelle Studien zeigen bedeutende Fortschritte in der Technologie erneuerbarer Energien. 
                Wissenschaftler an führenden Forschungseinrichtungen haben verbesserte Effizienz im 
                Solarzellen-Design nachgewiesen. Der neue Ansatz kombiniert neuartige Materialien mit 
                innovativen Herstellungsprozessen. Diese Entwicklungen könnten den Sektor der erneuerbaren 
                Energien in den kommenden Jahren revolutionieren.</p>
                
                <p>Weitere Experimente, die in mehreren Laboratorien durchgeführt wurden, bestätigten 
                diese Ergebnisse. Die Forschungsteams dokumentierten konstante Verbesserungen der 
                Energieumwandlungsraten. Unabhängige Überprüfungen haben das Vertrauen in diese 
                Erkenntnisse gestärkt. Branchenexperten evaluieren nun mögliche kommerzielle Anwendungen.</p>
            </article>
        </main>
    </body>
</html>
"""

EMPTY_HTML = "<html><body></body></html>"

MIXED_CONTENT_HTML = """
<html lang="en">
    <head>
        <title>Main Article</title>
        <meta name="description" content="Test article with mixed content types.">
    </head>
    <body>
        <main>
            <div class="navigation">
                <p>Home | About | Contact</p>
                <p>Terms of Service | Privacy Policy</p>
            </div>
            <article>
                <h1>Main Article Title</h1>
                <p>This is a detailed paragraph with substantial content that discusses an important topic
                in great detail. The length and structure of this content should make justext classify it
                as good content. It contains multiple sentences and provides meaningful information that
                would be considered part of the main content.</p>
                
                <p>This is another substantial paragraph that continues the discussion with more detailed
                information. Having multiple sentences and proper structure helps justext identify this
                as main content rather than boilerplate text.</p>
            </article>
            <div class="footer">
                <p>Copyright © 2024 | All rights reserved</p>
                <p>Follow us on social media</p>
            </div>
        </main>
    </body>
</html>
"""

SPECIAL_CHARS_HTML = """
<html lang="en">
    <head>
        <title>Special Characters Test</title>
        <meta name="description" content="Testing special character handling.">
    </head>
    <body>
        <article>
            <h1>Advanced Mathematical and Scientific Analysis</h1>
            <p>This comprehensive study examines various mathematical symbols and their applications
            in real-world scenarios. Key findings include the mathematical constant π ≈ 3.14159,
            which plays a crucial role in circular geometries and wave analysis. The research team,
            led by Dr. María González from München University, demonstrated several practical
            applications.</p>
            
            <p>The economic implications were significant, with cost savings of €15.99 per unit
            in European markets. Further experiments in São Paulo and München showed consistent
            results across different testing conditions. The team's findings suggest important
            implications for future developments in the field of mathematical applications
            in international markets.</p>
        </article>
    </body>
</html>
"""

@responses.activate
def test_metadata_extraction():
    """Test metadata extraction functionality."""
    url = "https://example.com/test"
    responses.add(responses.GET, url, body=ENGLISH_HTML, status=200)
    
    loader = JustextWebLoader(url)
    docs = loader.load()
    
    metadata = docs[0].metadata
    assert metadata["title"] == "Important Research Findings"
    assert "Research on renewable energy" in metadata["description"]
    assert metadata["html_language"] == "en"
    
    # Check justext metadata is preserved
    assert "language" in metadata
    assert "headings" in metadata
    assert "paragraph_count" in metadata
    assert "class_type" in metadata
    assert "source" in metadata

@responses.activate
def test_basic_loading():
    """Test basic document loading functionality."""
    url = "https://example.com/test"
    responses.add(responses.GET, url, body=ENGLISH_HTML, status=200)
    
    loader = JustextWebLoader(url)
    docs = loader.load()
    
    assert len(docs) > 0
    assert isinstance(docs[0].page_content, str)
    assert isinstance(docs[0].metadata, dict)

@responses.activate
def test_english_content():
    """Test loader with English content."""
    url = "https://example.com/english"
    responses.add(responses.GET, url, body=ENGLISH_HTML, status=200)
    
    loader = JustextWebLoader(url)
    docs = loader.load()
    
    assert len(docs) > 0
    assert any("renewable energy" in doc.page_content.lower() for doc in docs)
    assert all(doc.metadata["language"] == "English" for doc in docs)
    assert all("headings" in doc.metadata for doc in docs)
    assert all(isinstance(doc.metadata["headings"], list) for doc in docs)

@responses.activate
def test_german_content():
    """Test loader with German content."""
    url = "https://example.com/german"
    responses.add(responses.GET, url, body=GERMAN_HTML, status=200)
    
    loader = JustextWebLoader(url, language="German")
    docs = loader.load()
    
    assert len(docs) > 0
    assert any("erneuerbar" in doc.page_content.lower() for doc in docs)
    assert all(doc.metadata["language"] == "German" for doc in docs)
    assert all("headings" in doc.metadata for doc in docs)

@responses.activate
def test_error_handling():
    """Test error handling for failed requests."""
    url = "https://example.com/notfound"
    responses.add(responses.GET, url, status=404)
    
    loader = JustextWebLoader(url)
    with pytest.raises(requests.exceptions.HTTPError):
        _ = loader.load()

@responses.activate
def test_split_mode():
    """Test split_by_justext_paragraphs=True mode."""
    url = "https://example.com/split"
    responses.add(responses.GET, url, body=ENGLISH_HTML, status=200)
    
    loader = JustextWebLoader(url, split_by_justext_paragraphs=True)
    docs = loader.load()
    
    assert len(docs) > 1
    assert all(doc.metadata.get("paragraph_count") == 1 for doc in docs)
    assert all("class_type" in doc.metadata for doc in docs)
    assert all("title" in doc.metadata for doc in docs)
    assert all("description" in doc.metadata for doc in docs)

@responses.activate
def test_combined_mode():
    """Test split_by_justext_paragraphs=False mode (default)."""
    url = "https://example.com/combined"
    responses.add(responses.GET, url, body=ENGLISH_HTML, status=200)
    
    loader = JustextWebLoader(url)
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].metadata["paragraph_count"] > 1
    assert "class_type" in docs[0].metadata
    assert "title" in docs[0].metadata
    assert "description" in docs[0].metadata

@responses.activate
def test_empty_content():
    """Test loader behavior with empty content."""
    url = "https://example.com/empty"
    responses.add(responses.GET, url, body=EMPTY_HTML, status=200)
    
    loader = JustextWebLoader(url)
    docs = loader.load()
    
    assert len(docs) == 0

@responses.activate
def test_special_characters():
    """Test handling of special characters."""
    url = "https://example.com/special"
    responses.add(responses.GET, url, body=SPECIAL_CHARS_HTML, status=200)
    
    loader = JustextWebLoader(url)
    docs = loader.load()
    
    assert len(docs) > 0
    assert any("π ≈" in doc.page_content for doc in docs)
    assert any("€15.99" in doc.page_content for doc in docs)
    assert any("München" in doc.page_content for doc in docs)

@responses.activate
def test_custom_params():
    """Test loader with custom justext parameters."""
    url = "https://example.com/custom"
    responses.add(responses.GET, url, body=MIXED_CONTENT_HTML, status=200)
    
    custom_params = {
        "length_low": 100,
        "length_high": 200,
        "stopwords_low": 0.3,
        "stopwords_high": 0.4,
        "max_heading_distance": 30,
        "no_headings": False,
    }
    
    loader = JustextWebLoader(url, **custom_params)
    docs = loader.load()
    
    assert len(docs) > 0
    assert all("class_type" in doc.metadata for doc in docs)

@responses.activate
def test_missing_metadata():
    """Test handling of missing metadata fields."""
    minimal_html = """
    <html><body>
        <p>This is some minimal test content that should be extracted.</p>
        <p>Adding a second paragraph to ensure we get some content.</p>
    </body></html>
    """
    url = "https://example.com/minimal"
    responses.add(responses.GET, url, body=minimal_html, status=200)
    
    loader = JustextWebLoader(url)
    docs = loader.load()
    
    assert len(docs) > 0, "Should extract at least one document even from minimal content"
    metadata = docs[0].metadata
    
    # Check default values for missing metadata
    assert metadata["title"] == ""
    assert metadata["description"] == "No description found."
    assert metadata["html_language"] == "No language found."
    assert metadata["language"] == "English"
    assert "source" in metadata
    assert "paragraph_count" in metadata
    assert "headings" in metadata
    assert "class_type" in metadata

@responses.activate
def test_metadata_consistency():
    """Test consistency of metadata across different modes."""
    url = "https://example.com/metadata"
    responses.add(responses.GET, url, body=ENGLISH_HTML, status=200)
    
    expected_fields = {
        "language", "headings", "paragraph_count", "class_type", "source",
        "title", "description", "html_language"
    }
    
    # Test regular mode
    regular_loader = JustextWebLoader(url)
    regular_docs = regular_loader.load()
    
    # Test split mode
    split_loader = JustextWebLoader(url, split_by_justext_paragraphs=True)
    split_docs = split_loader.load()
    
    # Print metadata for debugging
    print("\nRegular mode metadata:", regular_docs[0].metadata)
    if split_docs:
        print("Split mode metadata:", split_docs[0].metadata)
    
    # Check metadata fields present in both modes
    assert len(regular_docs) > 0, "Should have at least one document in regular mode"
    assert len(split_docs) > 0, "Should have at least one document in split mode"
    
    for doc in regular_docs:
        assert all(field in doc.metadata for field in expected_fields), \
            f"Missing fields in regular mode: {expected_fields - doc.metadata.keys()}"
    
    for doc in split_docs:
        assert all(field in doc.metadata for field in expected_fields), \
            f"Missing fields in split mode: {expected_fields - doc.metadata.keys()}"


@responses.activate
def test_language_options():
    """Test handling of different languages."""
    url = "https://example.com/language"
    responses.add(responses.GET, url, body=GERMAN_HTML, status=200)
    
    german_loader = JustextWebLoader(url, language="German")
    german_docs = german_loader.load()
    
    default_loader = JustextWebLoader(url)
    default_docs = default_loader.load()
    
    assert all(doc.metadata["language"] == "German" for doc in german_docs)
    assert all(doc.metadata["language"] == "English" for doc in default_docs)

if __name__ == "__main__":
    pytest.main([__file__])
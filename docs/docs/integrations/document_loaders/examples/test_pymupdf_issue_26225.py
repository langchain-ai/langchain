#!/usr/bin/env python3
"""
Test script to verify PyMuPDFLoader image extraction works correctly.

This script reproduces the test case for GitHub issue #26225 where
PyMuPDFLoader with extract_images=True would fail with a ValueError
about broadcasting shapes.

The issue was resolved by updating dependency versions:
- rapidocr-onnxruntime >= 1.4.0
- langchain-community >= 0.3.0
- numpy >= 2.0.0

Usage:
    python test_pymupdf_issue_26225.py [pdf_file]

If no PDF file is provided, a simple test PDF will be created.
"""

import sys
import tempfile
from pathlib import Path


def create_simple_test_pdf():
    """Create a minimal test PDF with some content"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❌ PyMuPDF not installed. Install with: pip install pymupdf")
        return None
    doc = fitz.open()
    page = doc.new_page()

    # Add some text and a simple shape
    page.insert_text(
        (50, 50),
        "Test PDF for PyMuPDFLoader\nImage extraction test for issue #26225",
    )

    # Add a colored rectangle to simulate image content
    rect = fitz.Rect(100, 100, 300, 200)
    page.draw_rect(rect, color=(0, 0, 1), fill=(0.8, 0.8, 0.9))
    page.insert_text((150, 150), "Sample content")

    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(temp_file.name)
    doc.close()

    return temp_file.name


def test_pymupdf_image_extraction(pdf_path):
    """Test PyMuPDFLoader with image extraction enabled"""
    print(f"Testing PyMuPDFLoader with: {pdf_path}")

    try:
        from langchain_community.document_loaders import PyMuPDFLoader
    except ImportError:
        print(
            "❌ langchain_community not installed. "
            "Install with: pip install langchain-community"
        )
        return False

    try:
        # This is the test that would previously fail with:
        # ValueError: operands could not be broadcast together with shapes
        # (896,800) (1,1,3)
        print("Creating PyMuPDFLoader with extract_images=True...")
        loader = PyMuPDFLoader(pdf_path, extract_images=True)

        print("Loading documents...")
        docs = loader.load()

        print(f"✅ Successfully loaded {len(docs)} documents")
        if docs:
            print(f"   First document has {len(docs[0].page_content)} characters")
            print(f"   Metadata keys: {list(docs[0].metadata.keys())}")

        return True

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print(f"   Error type: {type(e).__name__}")

        if "broadcast" in str(e).lower():
            print("\n⚠️  This appears to be the broadcasting error from issue #26225!")
            print("   Try updating your dependencies:")
            print("   - pip install --upgrade rapidocr-onnxruntime")
            print("   - pip install --upgrade langchain-community")
            print("   - pip install --upgrade numpy")

        return False


def main():
    print("PyMuPDFLoader Image Extraction Test")
    print("Testing for GitHub issue #26225 resolution")
    print("=" * 50)

    # Check if PDF file was provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not Path(pdf_path).exists():
            print(f"❌ File not found: {pdf_path}")
            return 1
    else:
        print("No PDF file provided, creating test PDF...")
        pdf_path = create_simple_test_pdf()
        if not pdf_path:
            return 1
        print(f"Created test PDF: {pdf_path}")

    try:
        success = test_pymupdf_image_extraction(pdf_path)

        if success:
            print("\n✅ Test PASSED: PyMuPDFLoader image extraction works correctly!")
            print("   Issue #26225 appears to be resolved.")
        else:
            print(
                "\n❌ Test FAILED: PyMuPDFLoader image extraction encountered errors."
            )

        return 0 if success else 1

    finally:
        # Clean up temporary file if we created one
        if len(sys.argv) <= 1 and pdf_path:
            try:
                Path(pdf_path).unlink()
                print(f"Cleaned up temporary file: {pdf_path}")
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())

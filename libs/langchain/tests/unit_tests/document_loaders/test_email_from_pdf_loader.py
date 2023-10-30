# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:41:52 2023

@author: tevsl
"""
from pathlib import Path
import pytest

from langchain.document_loaders import EmailFromPDF
from langchain.docstore.document import Document

@pytest.mark.requires("dateparser","pdf2image","pytesseract")
class TestEmailFromPDFLoader:
    #tests that PDF of email is successfully parsed
    def test_emailfrompdf_loader(self)->None:
        file_path=str(Path(__file__).resolve().
                      parent / "test_docs" / "sample_pdfemail" / "fake_email.pdf")
        expected_docs=[
            Document(
                page_content="""sender@sample.com
SSS
From: sender@sample.com
To: recip1 @org1.com; recip2@org2.com
Ce: cc1@org3.com; recip4@org4.com
Subject: This is a test subject
This is some content\n1
""",             
                metadata={"source":file_path, "page":0,
                          'from': 'sender@sample.com',
                          'to': ['recip1 @org1.com', 'recip2@org2.com'],
                          'cc': ['cc1@org3.com', 'recip4@org4.com'],
                          'subject': 'This is a test subject'}
                )]
        
        loader=EmailFromPDF(file_path)
        docs=loader.load()
        assert docs==expected_docs,"PDF not loaded correctly as email"

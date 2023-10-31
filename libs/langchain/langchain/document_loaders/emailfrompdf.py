import logging
import os
import re
from typing import Iterator, List

import dateparser
import pytesseract
from pdf2image import convert_from_path

from langchain.docstore.document import Document
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.pdf import BasePDFLoader

logger = logging.getLogger(__file__)


class EmailFromPDF(BasePDFLoader):
    """Loads a PDF with pdf2imager and then scans with tesseract.
    Deliberately avoids considering multiple columns to better expose header data
    Attempts to find email fields in text and returns them in metadata
    if all mandatory fields exist.
    One document is returned for each page in the pdf.
    Optionally supports adding header to subsequent pages of a multipage email.
    Always returns page number (zero-based) in metadata. By default
    returns from, to, cc, subject, inline-images, date, attachments tags
    and values if these fields are detected in the document.
    The values of from, subject, and date are strings.
    The values of to, cc, attachments, and inline-images are lists of strings.
    udate is a binary universal date derived from date which is the string
    found in the document. The document is not considered an email
    unless both to and from field are detected.
    Nevertheless, the document will be returned, just without the extra metadata.
    Note that non-email memos which have detectable to and from fields
    will be returned with metadata.

    Can be used standalone or with DirectoryLoader.

    Example
    -------
    from langchain.document_loaders import EmailFromPDF
    loader=EmailFromPDF("sample_documents/sample_pdfemail/fake_email.pdf")
    documents=loader.load()

    Example
    -------
    from langchain.document_loaders import DirectoryLoader
    from langchain.document_loaders import EmailFromPDF
    loader=DirectoryLoader("example_data/example_pdfemail","*.pdf",loader_cls=EmailFromPDF,
            loader_kwargs={"replicate_headers":True},
            show_progress=True)
    documents=loader.load()



    """

    def __init__(
        self,
        file: str,
        # if true, headers inserted at beginning of each page of multipage docs
        replicate_headers: bool = False,
        # list of email headers to look for
        headers: dict = {
            "from": ["From:", "FROM:"],
            "to": ["To:", "TO:"],
            "cc": ["Cc:", "CC:", "cc:", "Ce:", "ce:"],
            "subject": ["Subject:", "Re:", "RE:", "SUBJECT:"],
            "date": ["Date:", "Sent:", "DATE:"],
            "inline-images": ["Inline-Images:"],
            "attachments": ["Attachments:"],
        },
        # headers which must be present in email.
        mandatory: List[str] = ["from", "to"],
        # headers which return lists of strings in order of expected appearance
        lists: List[str] = ["to", "cc", "inline-images", "attachments"],
        date_tag: str = "date",  # local tag for date
        tesseract_location: str = r"C:\Program Files\Tesseract-OCR\tesseract",
    ) -> None:
        """initialize with file name and password"""
        self.__saved_head__ = ""  # saved header for optional replication
        self.__saved_metadata__: dict = {}  # saved metadata for subsequent pages
        self.__saved_source__ = ""  # saved name of last doc processed
        self.replicate_headers = replicate_headers
        self.mandatory = mandatory
        self.lists = lists
        self.allheads = set().union(*headers.values())
        self.date_tag = date_tag
        assert os.path.exists(tesseract_location + ".exe"), (
            "tesseract executable not found at "
            + tesseract_location
            + ". Please install and/or specify the correct location."
        )
        self.tesseract_location = tesseract_location
        super().__init__(file)
        self.headers: dict = headers  # was getting wiped out by --init--

    def parse_email(self, thedoc: Document) -> Document:
        """does the work of parsing pdf as possible email"""
        if thedoc.metadata["page"] != 0:  # if not first page
            if (thedoc.metadata["source"] == self.__saved_source__) and (
                self.mandatory[0] in self.__saved_metadata__
            ):
                # if ff page of a known email
                thedoc.metadata.update(self.__saved_metadata__)
                # propagate the email metadata
                if self.replicate_headers:  # if supposed to replicate headers
                    thedoc.page_content = (
                        self.__saved_head__ + thedoc.page_content
                    )  # do it
            return thedoc  # exit for all but first page
        self.__saved_source__ = thedoc.metadata["source"]  # remember doc name
        self.__saved_head__ = ""  # clear out saved head
        self.__saved_metadata__ = {}  # and metadata
        fields = {}
        lomatch = 99999  # get the span of  header to replicate in following pages
        himatch = 0
        email_text = thedoc.page_content.replace(":\n", ": ")
        # to help isolate header fields
        first_head_pos = min(
            (
                email_text.find(sub)
                for sub in self.allheads
                if email_text.find(sub) != -1
            ),
            default=None,
        )
        if first_head_pos is None:  # if none found, not email
            return thedoc
        email_text = email_text[first_head_pos:]  # drop all before
        first_head = email_text[: email_text.find(":") + 1]
        second_head_pos = email_text.find(
            first_head, 1
        )  # look for repeat of first head
        if second_head_pos == -1:
            docend = len(email_text)
        else:  # but if second header
            docend = second_head_pos + 1  # stop search preserving one trailing char
        # special cleanup for header searching
        searchtext = email_text[:docend].replace("”", '"').replace(": \n", ":  ")
        quote = False
        # getting rid of newlines inside quotes
        # will not be able to handle single quotes since they may be apostrophes
        clean = ""
        for char in searchtext:
            if char == '"':
                quote = not quote
            elif char == "<":
                quote = False  # cures malformed quotes in email addresses
            elif char == "\n" and quote:
                char = ""
            clean += char
        for key in self.headers:
            for header in self.headers[key]:
                match = re.search(
                    f"{header}(.*?)(?=\n([a-zA-Z]|!|\||\.|\*|-))", clean, re.DOTALL
                )
                if match:
                    lomatch = min(lomatch, match.span()[0])
                    himatch = max(himatch, match.span()[1])
                    field_content = match.group(1).strip().replace("\n", "")
                    fields[key] = field_content
                    if key in self.lists:
                        if ";" in fields[key]:
                            # if there's a semicolon, probably it's the separator
                            split = re.split(
                                r';(?=(?:[^"]*"[^"]*")*[^"]*$)', fields[key]
                            )
                        elif (
                            fields[key].count(",") <= 1
                        ):  # if only one ","" and no ";"s
                            split = [
                                field_content,
                            ]  # its not really a list
                        else:
                            split = re.split(
                                r',(?=(?:[^"]*"[^"]*")*[^"]*$)', fields[key]
                            )
                            # split on commas
                        fields[key] = [field.strip() for field in split]
                    break
            if (key in self.mandatory) and (key not in fields or len(fields[key]) == 0):
                # should get rid of malformed
                return thedoc
        if "cc" in fields:  # if there's a cc field
            if isinstance(fields["cc"], list):  # if its a list
                for item in ("date:", "subject:"):  # for possible subsumed tags
                    if fields["cc"][0].startswith(item):  # if actually subsumed
                        del fields["cc"]  # drop bad tag
        if self.date_tag in fields:  # change date to unix timestamp at start of day
            try:  # don't want to die on malformed date
                date = dateparser.parse(fields[self.date_tag])
                if date:
                    date = date.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )  # go back to midnight
                    fields["udate"] = int(date.timestamp())
            except Exception as e:
                logger.warning(
                    f"malformed date not converted {fields[self.date_tag]} "
                    f"in {thedoc.metadata['source']}. "
                    f" Exception {e}."
                )
        thedoc.metadata.update(fields)  # add any new metadata
        self.__saved_metadata__ = fields  # remember for propagation
        if self.replicate_headers:  # if replicating headers
            self.__saved_head__ = clean[lomatch:himatch]  # remember them
        return thedoc

    def pdf2image2text(self, blob: Blob) -> Iterator[Document]:
        """return the pages from the pdf as text derived from images"""
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_location
        logger.warning(blob.source)
        assert blob.source is not None, "file not specified"
        try:
            images = convert_from_path(blob.source)  # get bit mapped images of pdf
        except Exception as e:
            logger.warning(f"{blob.source} had error {str(e)}. Dummy substituted.")
            images = ["[doc couldn't be converted. this is a dummy page.]"]
        for page, image in enumerate(images):
            metadata = {"source": blob.source, "page": page}
            if isinstance(image, str):  # if already a string
                content = image  # just use it
            else:  # otherwise convert image
                content = pytesseract.image_to_string(image, config="--psm 6")
            yield Document(page_content=content, metadata=metadata)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        the_list = list(self.pdf2image2text(Blob.from_path(self.file_path)))
        for item in the_list:  # for each chunk returned
            yield self.parse_email(item)

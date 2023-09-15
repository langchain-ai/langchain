import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Union
import re
import xml.etree.cElementTree
import xml.sax.saxutils
import requests
import urllib
from PIL import Image
from pytesseract import pytesseract

try:
    from quip_api.quip import QuipClient, HTTPError
    from quip_api.quip import QuipError
except ImportError:
    raise ImportError(
        "`quip_api` package not found, please run "
        "`pip install quip_api`"
    )

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

_MAXIMUM_TITLE_LENGTH = 64


class QuipLoader(BaseLoader):
    """Load `Quip` pages.
    """

    def __init__(
            self,
            api_url: str,
            doc_url: str,
            access_token: str,
            request_timeout: Optional[int] = 60
    ):
        self.doc_url = doc_url
        self.quip_client = QuipClient(access_token=access_token, base_url=api_url, request_timeout=request_timeout)

    def load(
            self,
            folder_ids: Optional[set[str]] = None,
            thread_ids: Optional[set[str]] = None,
            include_comments: bool = False,
            include_images: bool = False,
    ) -> List[Document]:
        """
        :param folder_ids: List of specific folder IDs to load, defaults to None
        :param thread_ids: List of specific thread IDs to load, defaults to None
        :param include_comments: include comments, defaults to False
        :param include_images: include images, defaults to False
        """
        if not folder_ids and not thread_ids:
            raise ValueError("Must specify at least one among `folder_ids`, `thread_ids`")

        docs = []

        if not thread_ids:
            thread_ids = set(str)

        if folder_ids:
            for folder_id in folder_ids:
                self.get_thread_ids_by_folder_id(folder_id, 0, thread_ids)

        return self.process_threads(thread_ids, include_images, include_comments)

    def get_thread_ids_by_folder_id(self, folder_id: str, depth: int, thread_ids: set[str]) -> Optional[List[str]]:
        """Get thread ids by folder id"""
        try:
            folder = self.quip_client.get_folder(folder_id)
        except QuipError as e:
            if e.code == 403:
                logging.warning(f"depth {depth}, Skipped over restricted folder {folder_id}, {e}")
            else:
                logging.warning(f"depth {depth}, Skipped over folder {folder_id} due to unknown error {e.code}")
            return
        except HTTPError as e:
            logging.warning(f"depth {depth}, Skipped over folder {folder_id} due to HTTP error {e.code}")
            return

        title = folder["folder"].get("title", "Folder %s" % folder_id)

        logging.info(f"depth {depth}, Processing folder {title}")
        for child in folder["children"]:
            if "folder_id" in child:
                self.get_thread_ids_by_folder_id(child["folder_id"], depth + 1, thread_ids)
            elif "thread_id" in child:
                thread_ids.add(child["thread_id"])

    def process_threads(self, thread_ids: set[str], include_images: bool, include_messages: bool) -> List[Document]:
        """Process a list of thread into a list of documents."""
        docs = []
        for thread_id in thread_ids:
            doc = self.process_thread(thread_id, include_images, include_messages)
            docs.append(doc)
        return docs

    def process_thread(self, thread_id: str, include_images: bool, include_messages: bool) -> Optional[Document]:
        thread = self.quip_client.get_thread(thread_id)
        thread_id = thread["thread"]["id"]
        title = thread["thread"]["title"]
        sanitized_title = QuipLoader._sanitize_title(title)
        logger.info(f"processing thread {thread_id} title {sanitized_title}")

        if "html" in thread:
            # Parse the document
            try:
                tree = self.quip_client.parse_document_html(thread["html"])
            except xml.etree.cElementTree.ParseError as e:
                logger.error(f"Error parsing thread {title} {thread_id}, skipping, {e}")
                return

            xml_bytes = xml.etree.cElementTree.tostring(tree, encoding='utf-8')
            html = xml_bytes.decode('utf-8')
            # Strip the <html> tags that were introduced in parse_document_html
            html = html[6:-7]

            metadata = {
                "title": sanitized_title,
                "id": thread_id,
                "source": f"{self.doc_url}/{thread_id}",
            }

            # Download each image and replace with the new URL
            text = ""
            if include_images:
                text = self.process_thread_images(tree)

            if include_messages:
                text = text + "/n" + self.process_thread_messages(thread_id)

            return Document(
                page_content=html.encode("utf-8") + text,
                metadata=metadata,
            )

    def process_thread_images(self, tree):
        text = ""
        for img in tree.iter("img"):
            src = img.get("src")
            if not src.startswith("/blob"):
                continue
            _, _, thread_id, blob_id = src.split("/")
            blob_response = self.quip_client.get_blob(thread_id, blob_id)
            try:
                image = Image.open(BytesIO(blob_response.read()))
                text = text + "\n" + pytesseract.image_to_string(image)
            except OSError as e:
                logger.error(f"failed to convert image to text, {e}")
        return text

    def process_thread_messages(self, thread_id):
        max_created_usec = None
        messages = ""
        while True:
            chunk = self.quip_client.get_messages(
                thread_id, max_created_usec=max_created_usec, count=100)
            messages = messages + "\n" + chunk
            if chunk:
                max_created_usec = chunk[-1]["created_usec"] - 1
            else:
                break
        messages.reverse()
        return messages

    @staticmethod
    def _sanitize_title(title):
        sanitized_title = re.sub(r"\s", " ", title)
        sanitized_title = re.sub(r"(?u)[^- \w.]", "", sanitized_title)
        if len(sanitized_title) > _MAXIMUM_TITLE_LENGTH:
            sanitized_title = sanitized_title[:_MAXIMUM_TITLE_LENGTH]
        return sanitized_title

    @staticmethod
    def _escape(s):
        return xml.sax.saxutils.escape(s, {'"': "&quot;"})

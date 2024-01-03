import logging
import re
import xml.etree.cElementTree
import xml.sax.saxutils
from io import BytesIO
from typing import List, Optional, Sequence
from xml.etree.ElementTree import ElementTree

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

_MAXIMUM_TITLE_LENGTH = 64


class QuipLoader(BaseLoader):
    """Load `Quip` pages.

    Port of https://github.com/quip/quip-api/tree/master/samples/baqup
    """

    def __init__(
        self, api_url: str, access_token: str, request_timeout: Optional[int] = 60
    ):
        """
        Args:
            api_url: https://platform.quip.com
            access_token: token of access quip API. Please refer:
            https://quip.com/dev/automation/documentation/current#section/Authentication/Get-Access-to-Quip's-APIs
            request_timeout: timeout of request, default 60s.
        """
        try:
            from quip_api.quip import QuipClient
        except ImportError:
            raise ImportError(
                "`quip_api` package not found, please run " "`pip install quip_api`"
            )

        self.quip_client = QuipClient(
            access_token=access_token, base_url=api_url, request_timeout=request_timeout
        )

    def load(
        self,
        folder_ids: Optional[List[str]] = None,
        thread_ids: Optional[List[str]] = None,
        max_docs: Optional[int] = 1000,
        include_all_folders: bool = False,
        include_comments: bool = False,
        include_images: bool = False,
    ) -> List[Document]:
        """
        Args:
            :param folder_ids: List of specific folder IDs to load, defaults to None
            :param thread_ids: List of specific thread IDs to load, defaults to None
            :param max_docs: Maximum number of docs to retrieve in total, defaults 1000
            :param include_all_folders: Include all folders that your access_token
                   can access, but doesn't include your private folder
            :param include_comments: Include comments, defaults to False
            :param include_images: Include images, defaults to False
        """
        if not folder_ids and not thread_ids and not include_all_folders:
            raise ValueError(
                "Must specify at least one among `folder_ids`, `thread_ids` "
                "or set `include_all`_folders as True"
            )

        thread_ids = thread_ids or []

        if folder_ids:
            for folder_id in folder_ids:
                self.get_thread_ids_by_folder_id(folder_id, 0, thread_ids)

        if include_all_folders:
            user = self.quip_client.get_authenticated_user()
            if "group_folder_ids" in user:
                self.get_thread_ids_by_folder_id(
                    user["group_folder_ids"], 0, thread_ids
                )
            if "shared_folder_ids" in user:
                self.get_thread_ids_by_folder_id(
                    user["shared_folder_ids"], 0, thread_ids
                )

        thread_ids = list(set(thread_ids[:max_docs]))
        return self.process_threads(thread_ids, include_images, include_comments)

    def get_thread_ids_by_folder_id(
        self, folder_id: str, depth: int, thread_ids: List[str]
    ) -> None:
        """Get thread ids by folder id and update in thread_ids"""
        from quip_api.quip import HTTPError, QuipError

        try:
            folder = self.quip_client.get_folder(folder_id)
        except QuipError as e:
            if e.code == 403:
                logging.warning(
                    f"depth {depth}, Skipped over restricted folder {folder_id}, {e}"
                )
            else:
                logging.warning(
                    f"depth {depth}, Skipped over folder {folder_id} "
                    f"due to unknown error {e.code}"
                )
            return
        except HTTPError as e:
            logging.warning(
                f"depth {depth}, Skipped over folder {folder_id} "
                f"due to HTTP error {e.code}"
            )
            return

        title = folder["folder"].get("title", "Folder %s" % folder_id)

        logging.info(f"depth {depth}, Processing folder {title}")
        for child in folder["children"]:
            if "folder_id" in child:
                self.get_thread_ids_by_folder_id(
                    child["folder_id"], depth + 1, thread_ids
                )
            elif "thread_id" in child:
                thread_ids.append(child["thread_id"])

    def process_threads(
        self, thread_ids: Sequence[str], include_images: bool, include_messages: bool
    ) -> List[Document]:
        """Process a list of thread into a list of documents."""
        docs = []
        for thread_id in thread_ids:
            doc = self.process_thread(thread_id, include_images, include_messages)
            if doc is not None:
                docs.append(doc)
        return docs

    def process_thread(
        self, thread_id: str, include_images: bool, include_messages: bool
    ) -> Optional[Document]:
        thread = self.quip_client.get_thread(thread_id)
        thread_id = thread["thread"]["id"]
        title = thread["thread"]["title"]
        link = thread["thread"]["link"]
        update_ts = thread["thread"]["updated_usec"]
        sanitized_title = QuipLoader._sanitize_title(title)

        logger.info(
            f"processing thread {thread_id} title {sanitized_title} "
            f"link {link} update_ts {update_ts}"
        )

        if "html" in thread:
            # Parse the document
            try:
                tree = self.quip_client.parse_document_html(thread["html"])
            except xml.etree.cElementTree.ParseError as e:
                logger.error(f"Error parsing thread {title} {thread_id}, skipping, {e}")
                return None

            metadata = {
                "title": sanitized_title,
                "update_ts": update_ts,
                "id": thread_id,
                "source": link,
            }

            # Download each image and replace with the new URL
            text = ""
            if include_images:
                text = self.process_thread_images(tree)

            if include_messages:
                text = text + "/n" + self.process_thread_messages(thread_id)

            return Document(
                page_content=thread["html"] + text,
                metadata=metadata,
            )
        return None

    def process_thread_images(self, tree: ElementTree) -> str:
        text = ""

        try:
            from PIL import Image
            from pytesseract import pytesseract
        except ImportError:
            raise ImportError(
                "`Pillow or pytesseract` package not found, "
                "please run "
                "`pip install Pillow` or `pip install pytesseract`"
            )

        for img in tree.iter("img"):
            src = img.get("src")
            if not src or not src.startswith("/blob"):
                continue
            _, _, thread_id, blob_id = src.split("/")
            blob_response = self.quip_client.get_blob(thread_id, blob_id)
            try:
                image = Image.open(BytesIO(blob_response.read()))
                text = text + "\n" + pytesseract.image_to_string(image)
            except OSError as e:
                logger.error(f"failed to convert image to text, {e}")
                raise e
        return text

    def process_thread_messages(self, thread_id: str) -> str:
        max_created_usec = None
        messages = []
        while True:
            chunk = self.quip_client.get_messages(
                thread_id, max_created_usec=max_created_usec, count=100
            )
            messages.extend(chunk)
            if chunk:
                max_created_usec = chunk[-1]["created_usec"] - 1
            else:
                break
        messages.reverse()

        texts = [message["text"] for message in messages]

        return "\n".join(texts)

    @staticmethod
    def _sanitize_title(title: str) -> str:
        sanitized_title = re.sub(r"\s", " ", title)
        sanitized_title = re.sub(r"(?u)[^- \w.]", "", sanitized_title)
        if len(sanitized_title) > _MAXIMUM_TITLE_LENGTH:
            sanitized_title = sanitized_title[:_MAXIMUM_TITLE_LENGTH]
        return sanitized_title

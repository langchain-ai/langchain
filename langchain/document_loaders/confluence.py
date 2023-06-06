"""Load Data from a Confluence Space"""
import logging
from io import BytesIO
from typing import Any, Callable, List, Optional, Union

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class ConfluenceLoader(BaseLoader):
    """
    Load Confluence pages. Port of https://llamahub.ai/l/confluence
    This currently supports username/api_key, Oauth2 login or personal access token
    authentication.

    Specify a list page_ids and/or space_key to load in the corresponding pages into
    Document objects, if both are specified the union of both sets will be returned.

    You can also specify a boolean `include_attachments` to include attachments, this
    is set to False by default, if set to True all attachments will be downloaded and
    ConfluenceReader will extract the text from the attachments and add it to the
    Document object. Currently supported attachment types are: PDF, PNG, JPEG/JPG,
    SVG, Word and Excel.

    Hint: space_key and page_id can both be found in the URL of a page in Confluence
    - https://yoursite.atlassian.com/wiki/spaces/<space_key>/pages/<page_id>

    Example:
        .. code-block:: python

            from langchain.document_loaders import ConfluenceLoader

            loader = ConfluenceLoader(
                url="https://yoursite.atlassian.com/wiki",
                username="me",
                api_key="12345"
            )
            documents = loader.load(space_key="SPACE",limit=50)

    :param url: _description_
    :type url: str
    :param api_key: _description_, defaults to None
    :type api_key: str, optional
    :param username: _description_, defaults to None
    :type username: str, optional
    :param oauth2: _description_, defaults to {}
    :type oauth2: dict, optional
    :param token: _description_, defaults to None
    :type token: str, optional
    :param cloud: _description_, defaults to True
    :type cloud: bool, optional
    :param number_of_retries: How many times to retry, defaults to 3
    :type number_of_retries: Optional[int], optional
    :param min_retry_seconds: defaults to 2
    :type min_retry_seconds: Optional[int], optional
    :param max_retry_seconds:  defaults to 10
    :type max_retry_seconds: Optional[int], optional
    :param confluence_kwargs: additional kwargs to initialize confluence with
    :type confluence_kwargs: dict, optional
    :raises ValueError: Errors while validating input
    :raises ImportError: Required dependencies not installed.
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        oauth2: Optional[dict] = None,
        token: Optional[str] = None,
        cloud: Optional[bool] = True,
        number_of_retries: Optional[int] = 3,
        min_retry_seconds: Optional[int] = 2,
        max_retry_seconds: Optional[int] = 10,
        confluence_kwargs: Optional[dict] = None,
    ):
        confluence_kwargs = confluence_kwargs or {}
        errors = ConfluenceLoader.validate_init_args(
            url, api_key, username, oauth2, token
        )
        if errors:
            raise ValueError(f"Error(s) while validating input: {errors}")

        self.base_url = url
        self.number_of_retries = number_of_retries
        self.min_retry_seconds = min_retry_seconds
        self.max_retry_seconds = max_retry_seconds

        try:
            from atlassian import Confluence  # noqa: F401
        except ImportError:
            raise ImportError(
                "`atlassian` package not found, please run "
                "`pip install atlassian-python-api`"
            )

        if oauth2:
            self.confluence = Confluence(
                url=url, oauth2=oauth2, cloud=cloud, **confluence_kwargs
            )
        elif token:
            self.confluence = Confluence(
                url=url, token=token, cloud=cloud, **confluence_kwargs
            )
        else:
            self.confluence = Confluence(
                url=url,
                username=username,
                password=api_key,
                cloud=cloud,
                **confluence_kwargs,
            )

    @staticmethod
    def validate_init_args(
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        oauth2: Optional[dict] = None,
        token: Optional[str] = None,
    ) -> Union[List, None]:
        """Validates proper combinations of init arguments"""

        errors = []
        if url is None:
            errors.append("Must provide `base_url`")

        if (api_key and not username) or (username and not api_key):
            errors.append(
                "If one of `api_key` or `username` is provided, "
                "the other must be as well."
            )

        if (api_key or username) and oauth2:
            errors.append(
                "Cannot provide a value for `api_key` and/or "
                "`username` and provide a value for `oauth2`"
            )

        if oauth2 and oauth2.keys() != [
            "access_token",
            "access_token_secret",
            "consumer_key",
            "key_cert",
        ]:
            errors.append(
                "You have either ommited require keys or added extra "
                "keys to the oauth2 dictionary. key values should be "
                "`['access_token', 'access_token_secret', 'consumer_key', 'key_cert']`"
            )

        if token and (api_key or username or oauth2):
            errors.append(
                "Cannot provide a value for `token` and a value for `api_key`, "
                "`username` or `oauth2`"
            )

        if errors:
            return errors
        return None

    def load(
        self,
        space_key: Optional[str] = None,
        page_ids: Optional[List[str]] = None,
        label: Optional[str] = None,
        cql: Optional[str] = None,
        include_restricted_content: bool = False,
        include_archived_content: bool = False,
        include_attachments: bool = False,
        include_comments: bool = False,
        limit: Optional[int] = 50,
        max_pages: Optional[int] = 1000,
    ) -> List[Document]:
        """
        :param space_key: Space key retrieved from a confluence URL, defaults to None
        :type space_key: Optional[str], optional
        :param page_ids: List of specific page IDs to load, defaults to None
        :type page_ids: Optional[List[str]], optional
        :param label: Get all pages with this label, defaults to None
        :type label: Optional[str], optional
        :param cql: CQL Expression, defaults to None
        :type cql: Optional[str], optional
        :param include_restricted_content: defaults to False
        :type include_restricted_content: bool, optional
        :param include_archived_content: Whether to include archived content,
                                         defaults to False
        :type include_archived_content: bool, optional
        :param include_attachments: defaults to False
        :type include_attachments: bool, optional
        :param include_comments: defaults to False
        :type include_comments: bool, optional
        :param limit: Maximum number of pages to retrieve per request, defaults to 50
        :type limit: int, optional
        :param max_pages: Maximum number of pages to retrieve in total, defaults 1000
        :type max_pages: int, optional
        :raises ValueError: _description_
        :raises ImportError: _description_
        :return: _description_
        :rtype: List[Document]
        """
        if not space_key and not page_ids and not label and not cql:
            raise ValueError(
                "Must specify at least one among `space_key`, `page_ids`, "
                "`label`, `cql` parameters."
            )

        docs = []

        if space_key:
            pages = self.paginate_request(
                self.confluence.get_all_pages_from_space,
                space=space_key,
                limit=limit,
                max_pages=max_pages,
                status="any" if include_archived_content else "current",
                expand="body.storage.value",
            )
            docs += self.process_pages(
                pages, include_restricted_content, include_attachments, include_comments
            )

        if label:
            pages = self.paginate_request(
                self.confluence.get_all_pages_by_label,
                label=label,
                limit=limit,
                max_pages=max_pages,
            )
            ids_by_label = [page["id"] for page in pages]
            if page_ids:
                page_ids = list(set(page_ids + ids_by_label))
            else:
                page_ids = list(set(ids_by_label))

        if cql:
            pages = self.paginate_request(
                self.confluence.cql,
                cql=cql,
                limit=limit,
                max_pages=max_pages,
                include_archived_spaces=include_archived_content,
                expand="body.storage.value",
            )
            docs += self.process_pages(
                pages, include_restricted_content, include_attachments, include_comments
            )

        if page_ids:
            for page_id in page_ids:
                get_page = retry(
                    reraise=True,
                    stop=stop_after_attempt(
                        self.number_of_retries  # type: ignore[arg-type]
                    ),
                    wait=wait_exponential(
                        multiplier=1,  # type: ignore[arg-type]
                        min=self.min_retry_seconds,  # type: ignore[arg-type]
                        max=self.max_retry_seconds,  # type: ignore[arg-type]
                    ),
                    before_sleep=before_sleep_log(logger, logging.WARNING),
                )(self.confluence.get_page_by_id)
                page = get_page(page_id=page_id, expand="body.storage.value")
                if not include_restricted_content and not self.is_public_page(page):
                    continue
                doc = self.process_page(page, include_attachments, include_comments)
                docs.append(doc)

        return docs

    def paginate_request(self, retrieval_method: Callable, **kwargs: Any) -> List:
        """Paginate the various methods to retrieve groups of pages.

        Unfortunately, due to page size, sometimes the Confluence API
        doesn't match the limit value. If `limit` is  >100 confluence
        seems to cap the response to 100. Also, due to the Atlassian Python
        package, we don't get the "next" values from the "_links" key because
        they only return the value from the results key. So here, the pagination
        starts from 0 and goes until the max_pages, getting the `limit` number
        of pages with each request. We have to manually check if there
        are more docs based on the length of the returned list of pages, rather than
        just checking for the presence of a `next` key in the response like this page
        would have you do:
        https://developer.atlassian.com/server/confluence/pagination-in-the-rest-api/

        :param retrieval_method: Function used to retrieve docs
        :type retrieval_method: callable
        :return: List of documents
        :rtype: List
        """

        max_pages = kwargs.pop("max_pages")
        docs: List[dict] = []
        while len(docs) < max_pages:
            get_pages = retry(
                reraise=True,
                stop=stop_after_attempt(
                    self.number_of_retries  # type: ignore[arg-type]
                ),
                wait=wait_exponential(
                    multiplier=1,
                    min=self.min_retry_seconds,  # type: ignore[arg-type]
                    max=self.max_retry_seconds,  # type: ignore[arg-type]
                ),
                before_sleep=before_sleep_log(logger, logging.WARNING),
            )(retrieval_method)
            batch = get_pages(**kwargs, start=len(docs))
            if not batch:
                break
            docs.extend(batch)
        return docs[:max_pages]

    def is_public_page(self, page: dict) -> bool:
        """Check if a page is publicly accessible."""
        restrictions = self.confluence.get_all_restrictions_for_content(page["id"])

        return (
            page["status"] == "current"
            and not restrictions["read"]["restrictions"]["user"]["results"]
            and not restrictions["read"]["restrictions"]["group"]["results"]
        )

    def process_pages(
        self,
        pages: List[dict],
        include_restricted_content: bool,
        include_attachments: bool,
        include_comments: bool,
    ) -> List[Document]:
        """Process a list of pages into a list of documents."""
        docs = []
        for page in pages:
            if not include_restricted_content and not self.is_public_page(page):
                continue
            doc = self.process_page(page, include_attachments, include_comments)
            docs.append(doc)

        return docs

    def process_page(
        self,
        page: dict,
        include_attachments: bool,
        include_comments: bool,
    ) -> Document:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError:
            raise ImportError(
                "`beautifulsoup4` package not found, please run "
                "`pip install beautifulsoup4`"
            )

        if include_attachments:
            attachment_texts = self.process_attachment(page["id"])
        else:
            attachment_texts = []
        text = BeautifulSoup(page["body"]["storage"]["value"], "lxml").get_text(
            " ", strip=True
        ) + "".join(attachment_texts)
        if include_comments:
            comments = self.confluence.get_page_comments(
                page["id"], expand="body.view.value", depth="all"
            )["results"]
            comment_texts = [
                BeautifulSoup(comment["body"]["view"]["value"], "lxml").get_text(
                    " ", strip=True
                )
                for comment in comments
            ]
            text = text + "".join(comment_texts)

        return Document(
            page_content=text,
            metadata={
                "title": page["title"],
                "id": page["id"],
                "source": self.base_url.strip("/") + page["_links"]["webui"],
            },
        )

    def process_attachment(self, page_id: str) -> List[str]:
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            raise ImportError(
                "`Pillow` package not found, " "please run `pip install Pillow`"
            )

        # depending on setup you may also need to set the correct path for
        # poppler and tesseract
        attachments = self.confluence.get_attachments_from_content(page_id)["results"]
        texts = []
        for attachment in attachments:
            media_type = attachment["metadata"]["mediaType"]
            absolute_url = self.base_url + attachment["_links"]["download"]
            title = attachment["title"]
            if media_type == "application/pdf":
                text = title + self.process_pdf(absolute_url)
            elif (
                media_type == "image/png"
                or media_type == "image/jpg"
                or media_type == "image/jpeg"
            ):
                text = title + self.process_image(absolute_url)
            elif (
                media_type == "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ):
                text = title + self.process_doc(absolute_url)
            elif media_type == "application/vnd.ms-excel":
                text = title + self.process_xls(absolute_url)
            elif media_type == "image/svg+xml":
                text = title + self.process_svg(absolute_url)
            else:
                continue
            texts.append(text)

        return texts

    def process_pdf(self, link: str) -> str:
        try:
            import pytesseract  # noqa: F401
            from pdf2image import convert_from_bytes  # noqa: F401
        except ImportError:
            raise ImportError(
                "`pytesseract` or `pdf2image` package not found, "
                "please run `pip install pytesseract pdf2image`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text
        try:
            images = convert_from_bytes(response.content)
        except ValueError:
            return text

        for i, image in enumerate(images):
            image_text = pytesseract.image_to_string(image)
            text += f"Page {i + 1}:\n{image_text}\n\n"

        return text

    def process_image(self, link: str) -> str:
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            raise ImportError(
                "`pytesseract` or `Pillow` package not found, "
                "please run `pip install pytesseract Pillow`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text
        try:
            image = Image.open(BytesIO(response.content))
        except OSError:
            return text

        return pytesseract.image_to_string(image)

    def process_doc(self, link: str) -> str:
        try:
            import docx2txt  # noqa: F401
        except ImportError:
            raise ImportError(
                "`docx2txt` package not found, please run `pip install docx2txt`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text
        file_data = BytesIO(response.content)

        return docx2txt.process(file_data)

    def process_xls(self, link: str) -> str:
        try:
            import xlrd  # noqa: F401
        except ImportError:
            raise ImportError("`xlrd` package not found, please run `pip install xlrd`")

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text

        workbook = xlrd.open_workbook(file_contents=response.content)
        for sheet in workbook.sheets():
            text += f"{sheet.name}:\n"
            for row in range(sheet.nrows):
                for col in range(sheet.ncols):
                    text += f"{sheet.cell_value(row, col)}\t"
                text += "\n"
            text += "\n"

        return text

    def process_svg(self, link: str) -> str:
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
            from reportlab.graphics import renderPM  # noqa: F401
            from svglib.svglib import svg2rlg  # noqa: F401
        except ImportError:
            raise ImportError(
                "`pytesseract`, `Pillow`, `reportlab` or `svglib` package not found, "
                "please run `pip install pytesseract Pillow reportlab svglib`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text

        drawing = svg2rlg(BytesIO(response.content))

        img_data = BytesIO()
        renderPM.drawToFile(drawing, img_data, fmt="PNG")
        img_data.seek(0)
        image = Image.open(img_data)

        return pytesseract.image_to_string(image)

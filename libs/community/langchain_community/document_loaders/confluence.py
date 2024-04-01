import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import requests
from langchain_core.documents import Document
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class ContentFormat(str, Enum):
    """Enumerator of the content formats of Confluence page."""

    EDITOR = "body.editor"
    EXPORT_VIEW = "body.export_view"
    ANONYMOUS_EXPORT_VIEW = "body.anonymous_export_view"
    STORAGE = "body.storage"
    VIEW = "body.view"

    def get_content(self, page: dict) -> str:
        return page["body"][self.name.lower()]["value"]


class ConfluenceLoader(BaseLoader):
    """Load `Confluence` pages.

    Port of https://llamahub.ai/l/confluence
    This currently supports username/api_key, Oauth2 login or personal access token
    authentication.

    Specify a list page_ids and/or space_key to load in the corresponding pages into
    Document objects, if both are specified the union of both sets will be returned.

    You can also specify a boolean `include_attachments` to include attachments, this
    is set to False by default, if set to True all attachments will be downloaded and
    ConfluenceReader will extract the text from the attachments and add it to the
    Document object. Currently supported attachment types are: PDF, PNG, JPEG/JPG,
    SVG, Word and Excel.

    Confluence API supports difference format of page content. The storage format is the
    raw XML representation for storage. The view format is the HTML representation for
    viewing with macros are rendered as though it is viewed by users. You can pass
    a enum `content_format` argument to specify the content format, this is
    set to `ContentFormat.STORAGE` by default, the supported values are:
    `ContentFormat.EDITOR`, `ContentFormat.EXPORT_VIEW`,
    `ContentFormat.ANONYMOUS_EXPORT_VIEW`, `ContentFormat.STORAGE`,
    and `ContentFormat.VIEW`.

    Hint: space_key and page_id can both be found in the URL of a page in Confluence
    - https://yoursite.atlassian.com/wiki/spaces/<space_key>/pages/<page_id>

    Example:
        .. code-block:: python

            from langchain_community.document_loaders import ConfluenceLoader

            loader = ConfluenceLoader(
                url="https://yoursite.atlassian.com/wiki",
                username="me",
                api_key="12345",
                space_key="SPACE",
                limit=50,
            )
            documents = loader.load()

            # Server on perm
            loader = ConfluenceLoader(
                url="https://confluence.yoursite.com/",
                username="me",
                api_key="your_password",
                cloud=False,
                space_key="SPACE",
                limit=50,
            )
            documents = loader.load()

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
    :param content_format: Specify content format, defaults to
                            ContentFormat.STORAGE, the supported values are:
                            `ContentFormat.EDITOR`, `ContentFormat.EXPORT_VIEW`,
                            `ContentFormat.ANONYMOUS_EXPORT_VIEW`,
                            `ContentFormat.STORAGE`, and `ContentFormat.VIEW`.
    :type content_format: ContentFormat
    :param limit: Maximum number of pages to retrieve per request, defaults to 50
    :type limit: int, optional
    :param max_pages: Maximum number of pages to retrieve in total, defaults 1000
    :type max_pages: int, optional
    :param ocr_languages: The languages to use for the Tesseract agent. To use a
                          language, you'll first need to install the appropriate
                          Tesseract language pack.
    :type ocr_languages: str, optional
    :param keep_markdown_format: Whether to keep the markdown format, defaults to
        False
    :type keep_markdown_format: bool
    :param keep_newlines: Whether to keep the newlines format, defaults to
        False
    :type keep_newlines: bool
    :raises ValueError: Errors while validating input
    :raises ImportError: Required dependencies not installed.
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        session: Optional[requests.Session] = None,
        oauth2: Optional[dict] = None,
        token: Optional[str] = None,
        cloud: Optional[bool] = True,
        number_of_retries: Optional[int] = 3,
        min_retry_seconds: Optional[int] = 2,
        max_retry_seconds: Optional[int] = 10,
        confluence_kwargs: Optional[dict] = None,
        *,
        space_key: Optional[str] = None,
        page_ids: Optional[List[str]] = None,
        label: Optional[str] = None,
        cql: Optional[str] = None,
        include_restricted_content: bool = False,
        include_archived_content: bool = False,
        include_attachments: bool = False,
        include_comments: bool = False,
        content_format: ContentFormat = ContentFormat.STORAGE,
        limit: Optional[int] = 50,
        max_pages: Optional[int] = 1000,
        ocr_languages: Optional[str] = None,
        keep_markdown_format: bool = False,
        keep_newlines: bool = False,
    ):
        self.space_key = space_key
        self.page_ids = page_ids
        self.label = label
        self.cql = cql
        self.include_restricted_content = include_restricted_content
        self.include_archived_content = include_archived_content
        self.include_attachments = include_attachments
        self.include_comments = include_comments
        self.content_format = content_format
        self.limit = limit
        self.max_pages = max_pages
        self.ocr_languages = ocr_languages
        self.keep_markdown_format = keep_markdown_format
        self.keep_newlines = keep_newlines

        confluence_kwargs = confluence_kwargs or {}
        errors = ConfluenceLoader.validate_init_args(
            url=url,
            api_key=api_key,
            username=username,
            session=session,
            oauth2=oauth2,
            token=token,
        )
        if errors:
            raise ValueError(f"Error(s) while validating input: {errors}")
        try:
            from atlassian import Confluence  # noqa: F401
        except ImportError:
            raise ImportError(
                "`atlassian` package not found, please run "
                "`pip install atlassian-python-api`"
            )

        self.base_url = url
        self.number_of_retries = number_of_retries
        self.min_retry_seconds = min_retry_seconds
        self.max_retry_seconds = max_retry_seconds

        if session:
            self.confluence = Confluence(url=url, session=session, **confluence_kwargs)
        elif oauth2:
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
        session: Optional[requests.Session] = None,
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

        non_null_creds = list(
            x is not None for x in ((api_key or username), session, oauth2, token)
        )
        if sum(non_null_creds) > 1:
            all_names = ("(api_key, username)", "session", "oath2", "token")
            provided = tuple(n for x, n in zip(non_null_creds, all_names) if x)
            errors.append(
                f"Cannot provide a value for more than one of: {all_names}. Received "
                f"values for: {provided}"
            )
        if oauth2 and set(oauth2.keys()) != {
            "access_token",
            "access_token_secret",
            "consumer_key",
            "key_cert",
        }:
            errors.append(
                "You have either omitted require keys or added extra "
                "keys to the oauth2 dictionary. key values should be "
                "`['access_token', 'access_token_secret', 'consumer_key', 'key_cert']`"
            )
        return errors or None

    def _resolve_param(self, param_name: str, kwargs: Any) -> Any:
        return kwargs[param_name] if param_name in kwargs else getattr(self, param_name)

    def _lazy_load(self, **kwargs: Any) -> Iterator[Document]:
        if kwargs:
            logger.warning(
                f"Received runtime arguments {kwargs}. Passing runtime args to `load`"
                f" is deprecated. Please pass arguments during initialization instead."
            )
        space_key = self._resolve_param("space_key", kwargs)
        page_ids = self._resolve_param("page_ids", kwargs)
        label = self._resolve_param("label", kwargs)
        cql = self._resolve_param("cql", kwargs)
        include_restricted_content = self._resolve_param(
            "include_restricted_content", kwargs
        )
        include_archived_content = self._resolve_param(
            "include_archived_content", kwargs
        )
        include_attachments = self._resolve_param("include_attachments", kwargs)
        include_comments = self._resolve_param("include_comments", kwargs)
        content_format = self._resolve_param("content_format", kwargs)
        limit = self._resolve_param("limit", kwargs)
        max_pages = self._resolve_param("max_pages", kwargs)
        ocr_languages = self._resolve_param("ocr_languages", kwargs)
        keep_markdown_format = self._resolve_param("keep_markdown_format", kwargs)
        keep_newlines = self._resolve_param("keep_newlines", kwargs)

        if not space_key and not page_ids and not label and not cql:
            raise ValueError(
                "Must specify at least one among `space_key`, `page_ids`, "
                "`label`, `cql` parameters."
            )

        if space_key:
            pages = self.paginate_request(
                self.confluence.get_all_pages_from_space,
                space=space_key,
                limit=limit,
                max_pages=max_pages,
                status="any" if include_archived_content else "current",
                expand=f"{content_format.value},version",
            )
            yield from self.process_pages(
                pages,
                include_restricted_content,
                include_attachments,
                include_comments,
                content_format,
                ocr_languages=ocr_languages,
                keep_markdown_format=keep_markdown_format,
                keep_newlines=keep_newlines,
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
                self._search_content_by_cql,
                cql=cql,
                limit=limit,
                max_pages=max_pages,
                include_archived_spaces=include_archived_content,
                expand=f"{content_format.value},version",
            )
            yield from self.process_pages(
                pages,
                include_restricted_content,
                include_attachments,
                include_comments,
                content_format,
                ocr_languages,
                keep_markdown_format,
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
                page = get_page(
                    page_id=page_id, expand=f"{content_format.value},version"
                )
                if not include_restricted_content and not self.is_public_page(page):
                    continue
                yield self.process_page(
                    page,
                    include_attachments,
                    include_comments,
                    content_format,
                    ocr_languages,
                    keep_markdown_format,
                )

    def load(self, **kwargs: Any) -> List[Document]:
        return list(self._lazy_load(**kwargs))

    def lazy_load(self) -> Iterator[Document]:
        yield from self._lazy_load()

    def _search_content_by_cql(
        self, cql: str, include_archived_spaces: Optional[bool] = None, **kwargs: Any
    ) -> List[dict]:
        url = "rest/api/content/search"

        params: Dict[str, Any] = {"cql": cql}
        params.update(kwargs)
        if include_archived_spaces is not None:
            params["includeArchivedSpaces"] = include_archived_spaces

        response = self.confluence.get(url, params=params)
        return response.get("results", [])

    def paginate_request(self, retrieval_method: Callable, **kwargs: Any) -> List:
        """Paginate the various methods to retrieve groups of pages.

        Unfortunately, due to page size, sometimes the Confluence API
        doesn't match the limit value. If `limit` is >100 confluence
        seems to cap the response to 100. Also, due to the Atlassian Python
        package, we don't get the "next" values from the "_links" key because
        they only return the value from the result key. So here, the pagination
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
        content_format: ContentFormat,
        ocr_languages: Optional[str] = None,
        keep_markdown_format: Optional[bool] = False,
        keep_newlines: bool = False,
    ) -> Iterator[Document]:
        """Process a list of pages into a list of documents."""
        for page in pages:
            if not include_restricted_content and not self.is_public_page(page):
                continue
            yield self.process_page(
                page,
                include_attachments,
                include_comments,
                content_format,
                ocr_languages=ocr_languages,
                keep_markdown_format=keep_markdown_format,
                keep_newlines=keep_newlines,
            )

    def process_page(
        self,
        page: dict,
        include_attachments: bool,
        include_comments: bool,
        content_format: ContentFormat,
        ocr_languages: Optional[str] = None,
        keep_markdown_format: Optional[bool] = False,
        keep_newlines: bool = False,
    ) -> Document:
        if keep_markdown_format:
            try:
                from markdownify import markdownify
            except ImportError:
                raise ImportError(
                    "`markdownify` package not found, please run "
                    "`pip install markdownify`"
                )
        if include_comments or not keep_markdown_format:
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                raise ImportError(
                    "`beautifulsoup4` package not found, please run "
                    "`pip install beautifulsoup4`"
                )
        if include_attachments:
            attachment_texts = self.process_attachment(page["id"], ocr_languages)
        else:
            attachment_texts = []

        content = content_format.get_content(page)
        if keep_markdown_format:
            # Use markdownify to keep the page Markdown style
            text = markdownify(content, heading_style="ATX") + "".join(attachment_texts)

        else:
            if keep_newlines:
                text = BeautifulSoup(
                    content.replace("</p>", "\n</p>").replace("<br />", "\n"), "lxml"
                ).get_text(" ") + "".join(attachment_texts)
            else:
                text = BeautifulSoup(content, "lxml").get_text(
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

        metadata = {
            "title": page["title"],
            "id": page["id"],
            "source": self.base_url.strip("/") + page["_links"]["webui"],
        }

        if "version" in page and "when" in page["version"]:
            metadata["when"] = page["version"]["when"]

        return Document(
            page_content=text,
            metadata=metadata,
        )

    def process_attachment(
        self,
        page_id: str,
        ocr_languages: Optional[str] = None,
    ) -> List[str]:
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
            try:
                if media_type == "application/pdf":
                    text = title + self.process_pdf(absolute_url, ocr_languages)
                elif (
                    media_type == "image/png"
                    or media_type == "image/jpg"
                    or media_type == "image/jpeg"
                ):
                    text = title + self.process_image(absolute_url, ocr_languages)
                elif (
                    media_type == "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ):
                    text = title + self.process_doc(absolute_url)
                elif media_type == "application/vnd.ms-excel":
                    text = title + self.process_xls(absolute_url)
                elif media_type == "image/svg+xml":
                    text = title + self.process_svg(absolute_url, ocr_languages)
                else:
                    continue
                texts.append(text)
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"Attachment not found at {absolute_url}")  # noqa: T201
                    continue
                else:
                    raise

        return texts

    def process_pdf(
        self,
        link: str,
        ocr_languages: Optional[str] = None,
    ) -> str:
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
            image_text = pytesseract.image_to_string(image, lang=ocr_languages)
            text += f"Page {i + 1}:\n{image_text}\n\n"

        return text

    def process_image(
        self,
        link: str,
        ocr_languages: Optional[str] = None,
    ) -> str:
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

        return pytesseract.image_to_string(image, lang=ocr_languages)

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
        import io
        import os

        try:
            import xlrd  # noqa: F401

        except ImportError:
            raise ImportError("`xlrd` package not found, please run `pip install xlrd`")

        try:
            import pandas as pd

        except ImportError:
            raise ImportError(
                "`pandas` package not found, please run `pip install pandas`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text

        filename = os.path.basename(link)
        # Getting the whole content of the url after filename,
        # Example: ".csv?version=2&modificationDate=1631800010678&cacheVersion=1&api=v2"
        file_extension = os.path.splitext(filename)[1]

        if file_extension.startswith(
            ".csv"
        ):  # if the extension found in the url is ".csv"
            content_string = response.content.decode("utf-8")
            df = pd.read_csv(io.StringIO(content_string))
            text += df.to_string(index=False, header=False) + "\n\n"
        else:
            workbook = xlrd.open_workbook(file_contents=response.content)
            for sheet in workbook.sheets():
                text += f"{sheet.name}:\n"
                for row in range(sheet.nrows):
                    for col in range(sheet.ncols):
                        text += f"{sheet.cell_value(row, col)}\t"
                    text += "\n"
                text += "\n"

        return text

    def process_svg(
        self,
        link: str,
        ocr_languages: Optional[str] = None,
    ) -> str:
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

        return pytesseract.image_to_string(image, lang=ocr_languages)

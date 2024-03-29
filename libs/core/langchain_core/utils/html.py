import logging
import re
from typing import List, Optional, Sequence, Union
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

PREFIXES_TO_IGNORE = ("javascript:", "mailto:", "#")
SUFFIXES_TO_IGNORE = (
    ".css",
    ".js",
    ".ico",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".csv",
    ".bz2",
    ".zip",
    ".epub",
)
SUFFIXES_TO_IGNORE_REGEX = (
    "(?!" + "|".join([re.escape(s) + r"[\#'\"]" for s in SUFFIXES_TO_IGNORE]) + ")"
)
PREFIXES_TO_IGNORE_REGEX = (
    "(?!" + "|".join([re.escape(s) for s in PREFIXES_TO_IGNORE]) + ")"
)
DEFAULT_LINK_REGEX = (
    rf"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)[\#'\"]"
)


def find_all_links(
    raw_html: str, *, pattern: Union[str, re.Pattern, None] = None
) -> List[str]:
    """Extract all links from a raw html string.

    Args:
        raw_html: original html.
        pattern: Regex to use for extracting links from raw html.

    Returns:
        List[str]: all links
    """
    pattern = pattern or DEFAULT_LINK_REGEX
    return list(set(re.findall(pattern, raw_html)))


def extract_sub_links(
    raw_html: str,
    url: str,
    *,
    base_url: Optional[str] = None,
    pattern: Union[str, re.Pattern, None] = None,
    prevent_outside: bool = True,
    exclude_prefixes: Sequence[str] = (),
    continue_on_failure: bool = False,
) -> List[str]:
    """Extract all links from a raw html string and convert into absolute paths.

    Args:
        raw_html: original html.
        url: the url of the html.
        base_url: the base url to check for outside links against.
        pattern: Regex to use for extracting links from raw html.
        prevent_outside: If True, ignore external links which are not children
            of the base url.
        exclude_prefixes: Exclude any URLs that start with one of these prefixes.
        continue_on_failure: If True, continue if parsing a specific link raises an
            exception. Otherwise, raise the exception.
    Returns:
        List[str]: sub links
    """
    base_url_to_use = base_url if base_url is not None else url
    parsed_base_url = urlparse(base_url_to_use)
    parsed_url = urlparse(url)
    all_links = find_all_links(raw_html, pattern=pattern)
    absolute_paths = set()
    for link in all_links:
        try:
            parsed_link = urlparse(link)
            # Some may be absolute links like https://to/path
            if parsed_link.scheme == "http" or parsed_link.scheme == "https":
                absolute_path = link
            # Some may have omitted the protocol like //to/path
            elif link.startswith("//"):
                absolute_path = f"{parsed_url.scheme}:{link}"
            else:
                absolute_path = urljoin(url, parsed_link.path)
            absolute_paths.add(absolute_path)
        except Exception as e:
            if continue_on_failure:
                logger.warning(f"Unable to load link {link}. Raised exception:\n\n{e}")
                continue
            else:
                raise e

    results = []
    for path in absolute_paths:
        if any(path.startswith(exclude_prefix) for exclude_prefix in exclude_prefixes):
            continue

        if prevent_outside:
            parsed_path = urlparse(path)

            if parsed_base_url.netloc != parsed_path.netloc:
                continue

            # Will take care of verifying rest of path after netloc
            # if it's more specific
            if not path.startswith(base_url_to_use):
                continue

        results.append(path)
    return results

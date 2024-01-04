import re
from typing import List, Optional, Sequence, Union
from urllib.parse import urljoin, urlparse

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

    Returns:
        List[str]: sub links
    """
    parsed_base_url = urlparse(base_url) if base_url is not None else urlparse(url)
    all_links = find_all_links(raw_html, pattern=pattern)
    absolute_paths = set()
    for link in all_links:
        parsed_link = urlparse(link)
        # Some may be absolute links like https://to/path
        if parsed_link.scheme == "http" or parsed_link.scheme == "https":
            absolute_paths.add(link)
        # Some may have omitted the protocol like //to/path
        elif parsed_link.scheme == "":
            absolute_paths.add(f"{urlparse(url).scheme}:{link}")
        else:
            absolute_paths.add(urljoin(url, parsed_link.path))

    results = []
    for path in absolute_paths:
        parsed_path = urlparse(path)

        if any(
            parsed_path.netloc == urlparse(exclude).netloc
            for exclude in exclude_prefixes
        ):
            continue

        if prevent_outside:
            if parsed_path.netloc != parsed_base_url.netloc:
                continue

        results.append(path)
    return results

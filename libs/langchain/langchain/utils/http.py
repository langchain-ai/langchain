import warnings

DEFAULT_REQUEST_HEADERS = {
    "User-Agent": "",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
              ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def get_request_headers(header_template: dict = None) -> dict:
    headers = (header_template or DEFAULT_REQUEST_HEADERS).copy()
    if not headers.get("User-Agent"):
        try:
            from fake_useragent import UserAgent

            headers["User-Agent"] = UserAgent().random
        except ImportError:
            warnings.warn(
                "fake_useragent not found, using default user agent."
                "To get a realistic header for requests, "
                "`pip install fake_useragent`."
            )

            headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                                    "AppleWebKit/537.36 (KHTML, like Gecko)" \
                                    " Chrome/104.0.0.0 Safari/537.36"
    return headers

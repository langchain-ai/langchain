import json
import re
import warnings
from typing import List, Tuple

import requests
from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader


class BiliBiliLoader(BaseLoader):
    """Load `BiliBili` video transcripts."""

    def __init__(self, video_urls: List[str]):
        """Initialize with bilibili url.

        Args:
            video_urls: List of bilibili urls.
        """
        self.video_urls = video_urls

    def load(self) -> List[Document]:
        """Load Documents from bilibili url."""
        results = []
        for url in self.video_urls:
            transcript, video_info = self._get_bilibili_subs_and_info(url)
            doc = Document(page_content=transcript, metadata=video_info)
            results.append(doc)

        return results

    def _get_bilibili_subs_and_info(self, url: str) -> Tuple[str, dict]:
        try:
            from bilibili_api import sync, video
        except ImportError:
            raise ImportError(
                "requests package not found, please install it with "
                "`pip install bilibili-api-python`"
            )

        bvid = re.search(r"BV\w+", url)
        if bvid is not None:
            v = video.Video(bvid=bvid.group())
        else:
            aid = re.search(r"av[0-9]+", url)
            if aid is not None:
                try:
                    v = video.Video(aid=int(aid.group()[2:]))
                except AttributeError:
                    raise ValueError(f"{url} is not bilibili url.")
            else:
                raise ValueError(f"{url} is not bilibili url.")

        video_info = sync(v.get_info())
        video_info.update({"url": url})
        sub = sync(v.get_subtitle(video_info["cid"]))

        # Get subtitle url
        sub_list = sub["subtitles"]
        if sub_list:
            sub_url = sub_list[0]["subtitle_url"]
            if not sub_url.startswith("http"):
                sub_url = "https:" + sub_url
            result = requests.get(sub_url)
            raw_sub_titles = json.loads(result.content)["body"]
            raw_transcript = " ".join([c["content"] for c in raw_sub_titles])

            raw_transcript_with_meta_info = (
                f"Video Title: {video_info['title']},"
                f"description: {video_info['desc']}\n\n"
                f"Transcript: {raw_transcript}"
            )
            return raw_transcript_with_meta_info, video_info
        else:
            raw_transcript = ""
            warnings.warn(
                f"""
                No subtitles found for video: {url}.
                Return Empty transcript.
                """
            )
            return raw_transcript, video_info

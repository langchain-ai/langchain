import json
import re
import warnings
from typing import List, Tuple

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

try:
    from bilibili_api import Credential, sync, video
except ImportError:
    raise ImportError(
        "requests package not found, please install it with "
        "`pip install bilibili-api-python`"
    )


class BiliBiliLoader(BaseLoader):
    """Load `BiliBili` video transcripts."""

    def __init__(self, video_urls: List[str], credential: Credential = None):
        """Initialize with bilibili url.

        Args:
            video_urls: List of bilibili urls.
            credential: The credential to access bilibili api.
        """
        self.video_urls = video_urls
        self.credential = credential

    def load(self) -> List[Document]:
        """Load Documents from bilibili url."""
        results = []
        for url in self.video_urls:
            transcript, video_info = self._get_bilibili_subs_and_info(url)
            doc = Document(page_content=transcript, metadata=video_info)
            results.append(doc)

        return results

    def _get_bilibili_subs_and_info(self, url: str) -> Tuple[str, dict]:
        bvid = re.search(r"BV\w+", url)
        if bvid is not None:
            v = video.Video(bvid=bvid.group(), credential=self.credential)
        else:
            aid = re.search(r"av[0-9]+", url)
            if aid is not None:
                try:
                    v = video.Video(
                        aid=int(aid.group()[2:]), credential=self.credential
                    )
                except AttributeError:
                    raise ValueError(f"{url} is not bilibili url.")
            else:
                raise ValueError(f"{url} is not bilibili url.")

        video_info = sync(v.get_info())
        video_info.update({"url": url})

        if (
            self.credential and len(video_info["subtitle"]["list"]) > 0
        ):  # video has subtitle
            if not sync(self.credential.check_valid()):  # check credential is valid
                raise ValueError("credential is invalid.")

            sub = sync(v.get_subtitle(video_info["cid"]))
            # Get subtitle url
            sub_list = sub["subtitles"]
        else:
            sub_list = []

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
            warnings.warn(
                f"""
                No subtitles found for video: {url}.
                Return Empty transcript.
                """
            )
            raw_transcript_with_meta_info = (
                f"Video Title: {video_info['title']},"
                f"description: {video_info['desc']}\n\n"
            )
            return raw_transcript_with_meta_info, video_info

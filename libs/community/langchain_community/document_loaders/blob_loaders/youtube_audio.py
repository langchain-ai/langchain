from typing import Iterable, List

from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.blob_loaders.schema import Blob, BlobLoader


class YoutubeAudioLoader(BlobLoader):
    """Load YouTube urls as audio file(s)."""

    def __init__(self, urls: List[str], save_dir: str):
        if not isinstance(urls, list):
            raise TypeError("urls must be a list")

        self.urls = urls
        self.save_dir = save_dir

    def yield_blobs(self) -> Iterable[Blob]:
        """Yield audio blobs for each url."""

        try:
            import yt_dlp
        except ImportError:
            raise ImportError(
                "yt_dlp package not found, please install it with "
                "`pip install yt_dlp`"
            )

        # Use yt_dlp to download audio given a YouTube url
        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "noplaylist": True,
            "outtmpl": self.save_dir + "/%(title)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
        }

        for url in self.urls:
            # Download file
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(url)

        # Yield the written blobs
        loader = FileSystemBlobLoader(self.save_dir, glob="*.m4a")
        for blob in loader.yield_blobs():
            yield blob

from langchain.document_loaders.blob_loaders.schema import Blob, BlobLoader
from langchain.document_loaders.blob_loaders import FileSystemBlobLoader
from typing import Iterable, List

class YoutubeAudioLoader(BlobLoader):

    """ Load YouTube urls as audio file(s). """

    def __init__(self,urls,save_dir):

        if not isinstance(urls, list):
            raise TypeError("urls must be a list")

        self.urls = urls
        self.save_dir = save_dir

    def yield_blobs(self) -> Iterable[Blob]: 

            """ Yield audio blobs for each url. """

            import io
            import yt_dlp

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
                    info = ydl.extract_info(url,download=False)
                    title = info.get('title', 'video')
                    print(f"Writing file: {title} to {self.save_dir}")
                    ydl.download(url)

            # Yield the written blobs
            loader = FileSystemBlobLoader(self.save_dir, glob="*.m4a")
            for blob in loader.yield_blobs():
                yield blob
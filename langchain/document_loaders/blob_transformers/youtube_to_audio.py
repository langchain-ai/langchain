from langchain.document_loaders.base import BaseBlobTransformer
from langchain.document_loaders.blob_loaders import Blob


class YoutubeToAudioTransformer(BaseBlobTransformer):

    """Dump YouTube url as audio file."""

    def lazy_transform(self, blob: Blob) -> Blob:
        """Lazily transform the blob."""

        import io
        import yt_dlp

        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
        }

        # Create a BytesIO object to store the downloaded file in memory
        output_file = io.BytesIO()

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Set the file output to the BytesIO object
            ydl.params['outtmpl'] = output_file
            # TODO: Need to add a WebUrlBlobLoader (or similar)
            ydl.download(blob.url)

        # Save to Blob 
        blob.data = output_file

        return blob

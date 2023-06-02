from langchain.document_loaders.base import BaseBlobTransformer
from langchain.document_loaders.blob_loaders import Blob


class YoutubeToAudioTransformer(BaseBlobTransformer):

    """Dump YouTube url as audio file."""

    # TODO: Output should be the blob path?
    def lazy_transform(self, input_url: Blob) -> Blob:
        """Lazily transform the blob."""

        import yt_dlp

        # TODO: Determine best way to pass this
        output_file_path = "path_to_files"

        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "outtmpl": output_file_path,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # TODO: Need a blob loader for urls
            ydl.download(input_url)

        return Blob.from_path(output_file_path)

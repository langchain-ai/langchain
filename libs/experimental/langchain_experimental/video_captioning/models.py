from datetime import datetime
from typing import Any


class BaseModel:
    def __init__(self, start_time: int, end_time: int) -> None:
        # Start and end times representing milliseconds
        self._start_time = start_time
        self._end_time = end_time

    @property
    def start_time(self) -> int:
        return self._start_time

    @start_time.setter
    def start_time(self, value: int) -> None:
        self._start_time = value

    @property
    def end_time(self) -> int:
        return self._end_time

    @end_time.setter
    def end_time(self, value: int) -> None:
        self._end_time = value

    def __str__(self) -> str:
        return f"start_time: {self.start_time}, end_time: {self.end_time}"

    @classmethod
    def from_srt(cls, start_time: str, end_time: str, *args: Any) -> "BaseModel":
        return cls(
            cls._srt_time_to_ms(start_time), cls._srt_time_to_ms(end_time), *args
        )

    @staticmethod
    def _srt_time_to_ms(srt_time_string: str) -> int:
        # Parse SRT time string into a datetime object
        time_format = "%H:%M:%S,%f"
        dt = datetime.strptime(srt_time_string, time_format)
        ms = dt.microsecond // 1000
        return dt.second * 1000 + ms


class VideoModel(BaseModel):
    def __init__(self, start_time: int, end_time: int, image_description: str) -> None:
        super().__init__(start_time, end_time)
        self._image_description = image_description

    @property
    def image_description(self) -> str:
        return self._image_description

    @image_description.setter
    def image_description(self, value: str) -> None:
        self._image_description = value

    def __str__(self) -> str:
        return f"{super().__str__()}, image_description: {self.image_description}"

    def similarity_score(self, other: "VideoModel") -> float:
        # Tokenize the image descriptions by extracting individual words, stripping
        # trailing 's' (plural = singular) and converting the words to lowercase in
        # order to be case-insensitive
        self_tokenized = set(
            word.lower().rstrip("s") for word in self.image_description.split()
        )
        other_tokenized = set(
            word.lower().rstrip("s") for word in other.image_description.split()
        )

        # Find common words
        common_words = self_tokenized.intersection(other_tokenized)

        # Calculate similarity score
        similarity_score = (
            len(common_words) / max(len(self_tokenized), len(other_tokenized)) * 100
        )

        return similarity_score


class AudioModel(BaseModel):
    def __init__(self, start_time: int, end_time: int, subtitle_text: str) -> None:
        super().__init__(start_time, end_time)
        self._subtitle_text = subtitle_text

    @property
    def subtitle_text(self) -> str:
        return self._subtitle_text

    @subtitle_text.setter
    def subtitle_text(self, value: str) -> None:
        self._subtitle_text = value

    def __str__(self) -> str:
        return f"{super().__str__()}, subtitle_text: {self.subtitle_text}"


class CaptionModel(BaseModel):
    def __init__(self, start_time: int, end_time: int, closed_caption: str) -> None:
        super().__init__(start_time, end_time)
        self._closed_caption = closed_caption

    @property
    def closed_caption(self) -> str:
        return self._closed_caption

    @closed_caption.setter
    def closed_caption(self, value: str) -> None:
        self._closed_caption = value

    def add_subtitle_text(self, subtitle_text: str) -> "CaptionModel":
        self._closed_caption = self._closed_caption + " " + subtitle_text
        return self

    def __str__(self) -> str:
        return f"{super().__str__()}, closed_caption: {self.closed_caption}"

    def to_srt_entry(self, index: int) -> str:
        def _ms_to_srt_time(ms: int) -> str:
            """Converts milliseconds to SRT time format 'HH:MM:SS,mmm'."""
            hours = int(ms // 3600000)
            minutes = int((ms % 3600000) // 60000)
            seconds = int((ms % 60000) // 1000)
            milliseconds = int(ms % 1000)

            return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

        return "\n".join(
            [
                f"""{index}
            {_ms_to_srt_time(self._start_time)} --> {_ms_to_srt_time(self._end_time)}
            {self._closed_caption}""",
            ]
        )

    @classmethod
    def from_audio_model(cls, audio_model: AudioModel) -> "CaptionModel":
        return cls(
            audio_model.start_time, audio_model.end_time, audio_model.subtitle_text
        )

    @classmethod
    def from_video_model(cls, video_model: VideoModel) -> "CaptionModel":
        return cls(
            video_model.start_time,
            video_model.end_time,
            f"[{video_model.image_description}]",
        )

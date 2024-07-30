from typing import List, Optional

import numpy as np
from langchain_community.document_loaders import ImageCaptionLoader
from langchain_core.callbacks import CallbackManagerForChainRun

from langchain_experimental.video_captioning.models import VideoModel


class ImageProcessor:
    _SAMPLES_PER_SECOND: int = 4

    def __init__(self, frame_skip: int = -1, threshold: int = 3000000) -> None:
        self.threshold = threshold
        self.frame_skip = frame_skip

    def process(
        self,
        video_file_path: str,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> list:
        return self._extract_frames(video_file_path)

    def _extract_frames(self, video_file_path: str) -> list:
        try:
            import cv2
            from cv2.typing import MatLike
        except ImportError as e:
            raise ImportError(
                "Unable to import cv2, please install it with "
                "`pip install -U opencv-python`"
            ) from e
        video_models: List[VideoModel] = []

        def _add_model(start_time: int, end_time: int) -> None:
            middle_frame_time = start_time / end_time
            cap.set(cv2.CAP_PROP_POS_MSEC, middle_frame_time)

            # Convert the frame to bytes
            _, encoded_frame = cv2.imencode(".jpg", frame)
            notable_frame_bytes = encoded_frame.tobytes()

            cap.set(cv2.CAP_PROP_POS_MSEC, end_time)

            # Create an instance of the ImageCaptionLoader
            loader = ImageCaptionLoader(images=notable_frame_bytes)

            # Load captions for the images
            list_docs = loader.load()

            video_model = VideoModel(
                start_time,
                end_time,
                list_docs[len(list_docs) - 1].page_content.replace("[SEP]", "").strip(),
            )
            video_models.append(video_model)

        def _is_notable_frame(frame1: MatLike, frame2: MatLike, threshold: int) -> bool:
            # Convert frames to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Compute absolute difference between frames
            frame_diff = cv2.absdiff(gray1, gray2)

            # Apply threshold to identify notable differences
            _, thresholded_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

            # Count the number of white pixels (indicating differences)
            num_diff_pixels = np.sum(thresholded_diff)

            return num_diff_pixels > threshold

        # Open the video file
        cap = cv2.VideoCapture(video_file_path)

        if self.frame_skip == -1:
            self.frame_skip = int(cap.get(cv2.CAP_PROP_FPS)) // self._SAMPLES_PER_SECOND

        # Read the first frame
        ret, prev_frame = cap.read()

        # Loop through the video frames
        start_time = 0
        end_time = 0

        while True:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if there are no more frames

            # Check if the current frame is notable
            if _is_notable_frame(prev_frame, frame, self.threshold):
                end_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                _add_model(start_time, end_time)
                start_time = end_time

            # Update the previous frame
            prev_frame = frame.copy()

            # Increment the frame position by the skip value
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                cap.get(cv2.CAP_PROP_POS_FRAMES) + self.frame_skip,
            )

        # Release the video capture object
        cap.release()

        return video_models

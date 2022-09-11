import typing as tp
import cv2
from pathlib import Path

import numpy as np


class VideoHandler:
    def __init__(self, source: tp.Union[int, str] = 0):
        """
        :param source: to set webcam as a source of frames provide 0, for files provide path to the file
        """
        if isinstance(source, str) and not Path(source).is_file():
            raise ValueError(f"File {source} is not existing")

        self.__video = None
        self.__source = source

    def __enter__(self):
        self.__video = cv2.VideoCapture(self.__source)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__video.release()
        cv2.destroyAllWindows()

    def read_video_frame(self) -> np.ndarray:
        ret, frame = self.__video.read()
        if not ret:
            raise ValueError(f"Cannot read video from source {self.__source}")
        return frame

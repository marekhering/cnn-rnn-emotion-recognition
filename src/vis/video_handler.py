import math
import typing as tp
from datetime import timedelta

import cv2
from pathlib import Path

import numpy as np

from config import GeneralConfig


class VideoHandler:
    def __init__(self, source: tp.Union[int, str] = 0):
        """
        :param source: to set webcam as a source of frames provide 0, for files provide path to the file
        """
        if isinstance(source, str) and not Path(source).is_file():
            raise ValueError(f"File {source} is not existing")

        self.__video = None
        self.__source = source
        self.__fps = None
        self.frame_counter = 0

    def __enter__(self):
        self.__video = cv2.VideoCapture(self.__source)
        self.__video.set(cv2.CAP_PROP_FPS, 2)
        self.__fps = self.__video.get(cv2.CAP_PROP_FPS)
        self.__fps = self.__fps if not math.isnan(self.__fps) else GeneralConfig.DEFAULT_FPS
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__video.release()
        cv2.destroyAllWindows()

    def read_video_frame(self) -> np.ndarray:
        ret, frame = None, None
        for _ in range(max((round(self.__fps / GeneralConfig.REQUIRED_FPS), 1))):
            self.frame_counter += 1
            ret, frame = self.__video.read()
        return frame if ret else None

    def get_frame_time(self):
        return timedelta(seconds=self.frame_counter / self.__fps)

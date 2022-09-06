import typing as tp
import cv2
from pathlib import Path

from config import VideoConfig


class VideoHandler:

    def __init__(self, source: tp.Union[int, str] = 0, quit_button: str = VideoConfig.DEFAULT_QUIT_BUTTON):
        """
        :param source: to set webcam as a source of frames provide 0, for files provide path to the file
        """
        if isinstance(source, str) and not Path(source).is_file():
            raise ValueError(f"File {source} is not existing")

        self.__video = None
        self.__source = source
        self.__quit_button = quit_button

    def __enter__(self, ):
        self.__video = cv2.VideoCapture(self.__source)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__video.release()
        cv2.destroyAllWindows()

    def read_video_frame(self):
        ret, frame = self.__video.read()
        if not ret:
            raise ValueError(f"Cannot read video from source {self.__source}")
        return frame

    def listen_for_quit_button(self):
        if cv2.waitKey(1) == ord(self.__quit_button):
            return True
        return False

    @staticmethod
    def show_video_frame(frame):
        cv2.imshow('frame', frame)

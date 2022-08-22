import cv2

from config import DEFAULT_QUIT_BUTTON


class WebcamHandler:

    def __init__(self, quit_button: str = None):

        if quit_button is None or len(quit_button) != 1:
            quit_button = DEFAULT_QUIT_BUTTON
        quit_button = quit_button.lower()

        self.__video = None
        self.__quit_button = quit_button

    def __enter__(self):
        self.__video = cv2.VideoCapture(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__video.release()
        cv2.destroyAllWindows()

    def read_video_frame(self):
        ret, frame = self.__video.read()
        return frame

    def listen_for_quit_button(self):
        if cv2.waitKey(1) == ord(self.__quit_button):
            return True
        return False

    @staticmethod
    def show_video_frame(frame):
        cv2.imshow('frame', frame)

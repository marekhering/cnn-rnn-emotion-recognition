import numpy as np
import cv2

from .models import CNNModel
from .webcam_handler import WebcamHandler
from config import CNN_MODEL_PATH


class App:
    def __init__(self):
        self.cnn_model = CNNModel(CNN_MODEL_PATH)

    def webcam_emotion_recognition(self):
        with WebcamHandler() as webcam:
            while True:
                frame = webcam.read_video_frame()
                prediction = self._inference(frame)
                print(f"\r{prediction}", end='')
                webcam.show_video_frame(frame)
                if webcam.listen_for_quit_button():
                    break

    def predict_image_file(self, image_path: str):
        img = cv2.imread(image_path)
        prediction = self._inference(img)
        print(prediction)
        return prediction

    def _inference(self, img: np.ndarray):
        prepared_img = self._prepare_img(img)
        x = np.expand_dims(prepared_img, axis=0)
        prediction = self.cnn_model.predict(x)
        return prediction

    def _prepare_img(self, frame: np.ndarray) -> np.ndarray:
        resized_img = cv2.resize(frame, self.cnn_model.image_shape, interpolation=cv2.INTER_AREA)
        grayscale_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        grayscale_image = np.expand_dims(grayscale_image, axis=-1)
        return (grayscale_image / 255.).astype(np.float)

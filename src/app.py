import typing as tp
import numpy as np
import cv2

from .analysis import Analyst
from .models import CNNModel, RNNModel
from .utils import ValenceArousal
from .vis import VideoHandler, Frame, ValenceArousalSpace
from config import PathConfig, GeneralConfig, FrameConfig


class App:
    def __init__(self):
        self.analyst = Analyst()
        self.cnn_model = CNNModel(PathConfig.CNN_MODEL_PATH)
        self.rnn_model = RNNModel(PathConfig.RNN_MODEL_PATH)
        self.face_detection_model = cv2.CascadeClassifier(PathConfig.FACE_RECOGNITION_MODEL_PATH)

        self._feature_buffer = []
        self._cnn_va = ValenceArousal()
        self._rnn_va = ValenceArousal()

    def rnn_video_emotion_recognition(self, source: tp.Union[str, int]):
        with VideoHandler(source) as video_handler:
            while True:
                video_frame = video_handler.read_video_frame()
                face_img = self._find_face(video_frame)
                prepared_img = self._prepare_img(face_img)

                self._inference_feature_extractor(prepared_img)
                self._inference_rnn_model()
                self._inference_classification_model()
                self.analyst.add_inference_result(self._rnn_va)

                self.log_predictions()
                self.visualize(video_frame, prepared_img)
                if self.listen_for_quit_button():
                    break

    def cnn_predict_image_file(self, image_path: str):
        img = cv2.imread(image_path)
        prepared_img = self._prepare_img(img)
        self._inference_feature_extractor(prepared_img)
        self._inference_classification_model()
        self.log_predictions()

    def _find_face(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection_model.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return frame[y: y + h, x: x + w]
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def _prepare_img(self, img: np.ndarray) -> np.ndarray:
        resized_img = cv2.resize(img, self.cnn_model.image_shape)
        grayscale_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        grayscale_image = np.expand_dims(grayscale_image, axis=-1)
        normalized_img = (grayscale_image / 255.).astype(np.float)
        return normalized_img

    def _inference_feature_extractor(self, img: np.ndarray):
        x = np.expand_dims(img, axis=0)
        features = self.cnn_model.extract_features(x)[0]
        self._feature_buffer.append(features)
        return features

    def _inference_rnn_model(self):
        if len(self._feature_buffer) >= self.rnn_model.window_size:
            window = self._feature_buffer[-self.rnn_model.window_size:]
            x = np.expand_dims(np.asarray(window), axis=0)
            self._rnn_va = ValenceArousal(*self.rnn_model.predict(x)[0])
            self._feature_buffer = window

    def _inference_classification_model(self):
        x = np.expand_dims(self._feature_buffer[-1], axis=0)
        self._cnn_va = ValenceArousal(*self.cnn_model.classify_features(x)[0])

    def visualize(self, video_frame: np.ndarray, inference_input: np.ndarray):
        frame = Frame(FrameConfig.MAIN_FRAME_SIZE, FrameConfig.FRAME_BACKGROUND)
        frame.add(video_frame, (0, 0), (.5, .33))
        frame.add(inference_input * 255, (0, 0))
        frame.add(ValenceArousalSpace.create_chart(self._cnn_va, self._rnn_va), (0., .3), (.5, .33))
        frame.add(self.analyst.create_va_chart(), (0, .66), (.5, .33))
        frame.add(self.analyst.create_valence_average_chart(), (.5, 0), (.5, .5))
        frame.add(self.analyst.create_arousal_average_chart(), (.5, .5), (.5, .5))
        frame.show()

    def log_predictions(self):
        print(f"CNN: {self._cnn_va}   RNN: {self._rnn_va}")

    @staticmethod
    def listen_for_quit_button() -> bool:
        if cv2.waitKey(1) == ord(GeneralConfig.DEFAULT_QUIT_BUTTON):
            return True
        return False

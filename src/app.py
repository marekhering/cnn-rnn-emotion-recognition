import typing as tp
import numpy as np
import cv2

from src.analysis.analyst import Analyst
from config import PathConfig
from .models import CNNModel, RNNModel
from .valence_arousal import ValenceArousal
from .video_handler import VideoHandler


class App:
    def __init__(self):
        self.analyst = Analyst()
        self.cnn_model = CNNModel(PathConfig.CNN_MODEL_PATH)
        self.rnn_model = RNNModel(PathConfig.RNN_MODEL_PATH)
        self._feature_buffer = []

        self.cnn_va = ValenceArousal()
        self.rnn_va = ValenceArousal()

        self.arousal_valence_space: np.ndarray = cv2.imread(PathConfig.AROUSAL_VALENCE_SPACE_PATH)
        self.face_cascade = cv2.CascadeClassifier(PathConfig.FACE_RECOGNITION_MODEL_PATH)

    def rnn_video_emotion_recognition(self, source: tp.Union[str, int]):
        with VideoHandler(source) as webcam:
            while True:
                frame = webcam.read_video_frame()
                face_img = self._find_face(frame)
                prepared_img = self._prepare_img(face_img)
                self._inference_feature_extractor(prepared_img)
                self._inference_rnn_model()
                self._inference_classification_model()
                self.log_predictions()

                frame = cv2.resize(frame, (frame.shape[1], frame.shape[1]))
                frame = self._add_valence_arousal_space(frame)
                frame = self._add_input_frame(frame, prepared_img)
                webcam.show_video_frame(frame)
                if webcam.listen_for_quit_button():
                    break

    def cnn_predict_image_file(self, image_path: str):
        img = cv2.imread(image_path)
        prepared_img = self._prepare_img(img)
        self._inference_feature_extractor(prepared_img)
        self._inference_classification_model()
        self.log_predictions()

    def _find_face(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return frame[y: y + h, x: x + w]
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def _inference_rnn_model(self):
        if len(self._feature_buffer) >= self.rnn_model.window_size:
            window = self._feature_buffer[-self.rnn_model.window_size:]
            x = np.expand_dims(np.asarray(window), axis=0)
            self.rnn_va = ValenceArousal(*self.rnn_model.predict(x)[0])
            self._feature_buffer = window

    def _inference_feature_extractor(self, img: np.ndarray):
        x = np.expand_dims(img, axis=0)
        features = self.cnn_model.extract_features(x)[0]
        self._feature_buffer.append(features)
        return features

    def _inference_classification_model(self):
        x = np.expand_dims(self._feature_buffer[-1], axis=0)
        self.cnn_va = ValenceArousal(*self.cnn_model.classify_features(x)[0])

    def _prepare_img(self, img: np.ndarray) -> np.ndarray:
        resized_img = cv2.resize(img, self.cnn_model.image_shape)
        grayscale_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        grayscale_image = np.expand_dims(grayscale_image, axis=-1)
        normalized_img = (grayscale_image / 255.).astype(np.float)
        return normalized_img

    def log_predictions(self):
        print(f"CNN: {self.cnn_va}   RNN: {self.rnn_va}")

    def _add_valence_arousal_space(self, frame: np.ndarray):
        def get_value_coord(_metric: float, _scale: int, multiplier: float = 0.8) -> int:
            _value = _metric * multiplier
            _value = (_value + 1) / 2              # From (-1, 1) to (0, 1)
            _value = int(_value * _scale)          # From (0, 1) to (0, scale)
            _value = min([_value, _scale])         # Ensure that value is lower than scale
            _value = max([_value, 0])              # Ensure that value is greater than 0
            return _value

        def add_point(_frame: np.ndarray, x: int, y: int, width: int, color: np.ndarray):
            _frame[x-width:x+width, y-width:y+width] = color

        shape = frame.shape
        av_space = cv2.resize(self.arousal_valence_space, shape[0:2])

        cnn_x, cnn_y = get_value_coord(-self.cnn_va.arousal, shape[0]), get_value_coord(self.cnn_va.valence, shape[1])
        rnn_x, rnn_y = get_value_coord(-self.rnn_va.valence, shape[0]), get_value_coord(self.rnn_va.valence, shape[1])

        add_point(av_space, cnn_x, cnn_y, 4, np.array([255, 0, 0]))
        add_point(av_space, rnn_x, rnn_y, 4, np.array([0, 255, 0]))

        return np.concatenate((frame, av_space), axis=1)

    def _add_input_frame(self, frame: np.ndarray, input_image: np.ndarray):
        shape = input_image.shape
        frame[0:shape[0], 0:shape[1]] = (input_image * 255).astype(np.uint8)
        return frame

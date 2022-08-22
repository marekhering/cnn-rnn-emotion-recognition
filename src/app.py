import typing as tp

import numpy as np
import cv2

from .models import CNNModel, RNNModel
from .webcam_handler import WebcamHandler
from config import CNN_MODEL_PATH, RNN_MODEL_PATH


class App:
    def __init__(self):
        self.cnn_model = CNNModel(CNN_MODEL_PATH)
        self.rnn_model = RNNModel(RNN_MODEL_PATH)

        self._feature_buffer = []
        self.cnn_valence = 0
        self.cnn_arousal = 0

        self.rnn_valence = 0
        self.rnn_arousal = 0

    def rnn_webcam_emotion_recognition(self):
        with WebcamHandler() as webcam:
            while True:
                frame = webcam.read_video_frame()
                self._inference_feature_extractor(frame)
                self._inference_rnn_model()
                self._inference_classification_model(self._feature_buffer[-1])
                self.show_predictions()

                webcam.show_video_frame(frame)
                if webcam.listen_for_quit_button():
                    break

    def cnn_predict_image_file(self, image_path: str):
        img = cv2.imread(image_path)
        features = self._inference_feature_extractor(img)
        self._inference_classification_model(features)
        self.show_predictions()

    def _inference_rnn_model(self):
        if len(self._feature_buffer) >= self.rnn_model.window_size:
            window = self._feature_buffer[-self.rnn_model.window_size:]
            x = np.expand_dims(np.asarray(window), axis=0)
            self.rnn_valence, self.rnn_arousal = self.rnn_model.predict(x)[0]
            self._feature_buffer = window
        return self.rnn_valence, self.rnn_arousal

    def _inference_feature_extractor(self, img: np.ndarray):
        prepared_img = self._prepare_img(img)
        x = np.expand_dims(prepared_img, axis=0)
        features = self.cnn_model.extract_features(x)[0]
        self._feature_buffer.append(features)
        return features

    def _inference_classification_model(self, features: np.ndarray):
        x = np.expand_dims(features, axis=0)
        self.cnn_valence, self.cnn_arousal = self.cnn_model.classify_features(x)[0]
        return self.cnn_valence, self.cnn_arousal

    def _prepare_img(self, img: np.ndarray) -> np.ndarray:
        resized_img = cv2.resize(img, self.cnn_model.image_shape, interpolation=cv2.INTER_AREA)
        grayscale_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        grayscale_image = np.expand_dims(grayscale_image, axis=-1)
        normalized_img = (grayscale_image / 255.).astype(np.float)
        return normalized_img

    def show_predictions(self):
        print("CNN: Valence %.6s | Arousal %.6s   RNN: Valence %.6s | Arousal %.6s" %
              (self.cnn_valence, self.cnn_arousal, self.rnn_valence, self.rnn_arousal))

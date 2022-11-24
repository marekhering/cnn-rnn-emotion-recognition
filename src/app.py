import json
import os
import typing as tp
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from .analysis import Analyst
from .models import CNNModel, RNNModel
from .utils import ValenceArousal, Buffer
from .vis import VideoHandler, Frame, ValenceArousalSpace
from config import PathConfig, GeneralConfig, FrameConfig


class App:
    def __init__(self):
        self.cnn_model = CNNModel(PathConfig.CNN_MODEL_PATH)
        self.rnn_model = RNNModel(PathConfig.RNN_MODEL_PATH)
        self.face_detection_model = cv2.CascadeClassifier(PathConfig.FACE_RECOGNITION_MODEL_PATH)
        self._reset_attributes()

    def _reset_attributes(self):
        self.analyst = Analyst()
        self._feature_buffer = []
        self._cnn_va = ValenceArousal()
        self._rnn_va = ValenceArousal()

    def videos_inference(self):
        for file_name in sorted(os.listdir(PathConfig.VIDEOS_PATH)):
            source = os.path.join(PathConfig.VIDEOS_PATH, file_name)
            self.video_inference(source, vis=False)

    def video_inference(self, source: tp.Union[str, int], vis: bool = True, save: bool = True):
        self._reset_attributes()
        with VideoHandler(source) as video_handler:
            while (video_frame := video_handler.read_video_frame()) is not None:
                face_img = self._find_face(video_frame)
                prepared_img = self._prepare_img(face_img)
                self._inference_feature_extractor(prepared_img)
                rnn_run = self._inference_rnn_model()
                self._inference_classification_model()
                self.log_predictions(source, video_handler.get_frame_time())

                if rnn_run:
                    self.analyst.add_inference_result(self._rnn_va, video_handler.get_frame_time())

                if vis:
                    self._visualize(video_frame, prepared_img)

                if self.listen_for_quit_button():
                    break
        if save:
            self.save_output(source)

    def save_output(self, source: str):
        output_dir = Path(PathConfig.OUTPUT_VIDEOS)
        output_dir.mkdir(exist_ok=True, parents=True)
        # Get file from path, change existing extension to .json
        output_file = f'{os.path.split(source)[1].split(".")[0]}.txt'
        output_file = os.path.join(output_dir, output_file)
        with open(output_file, 'w') as f:
            f.write(self.intersections_as_boris_format())

    def intersections_as_boris_format(self):
        boris_format = []
        for _time, activator in self.analyst.intersections:
            boris_time = str(_time.total_seconds()).split('.')
            boris_time = f"{boris_time[0]}.{boris_time[1][:3]}"
            boris_format.append("\t".join([f"{boris_time}", "", f"{activator.name}", "", ""]))
        return "\n".join(boris_format)

    def image_inference(self, image_path: str):
        self._reset_attributes()
        img = cv2.imread(image_path)
        prepared_img = self._prepare_img(img)
        self._inference_feature_extractor(prepared_img)
        self._inference_classification_model()
        self.log_predictions(image_path)

    def _find_face(self, frame: np.ndarray, b: int = 10) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection_model.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return frame[y - b: y + h + b, x - b: x + w + b]
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
            return True
        return False

    def _inference_classification_model(self):
        x = np.expand_dims(self._feature_buffer[-1], axis=0)
        self._cnn_va = ValenceArousal(*self.cnn_model.classify_features(x)[0])

    def _visualize(self, video_frame: np.ndarray, inference_input: np.ndarray):
        frame = Frame(FrameConfig.MAIN_FRAME_SIZE, FrameConfig.FRAME_BACKGROUND)
        frame.add(video_frame, (0, 0), (.5, .33))
        frame.add(inference_input * 255, (0, 0))
        frame.add(ValenceArousalSpace.create_chart(self._cnn_va, self._rnn_va), (0., .3), (.5, .33))
        try:
            frame.add(self.analyst.create_va_chart(), (0, .66), (.5, .33))
            frame.add(self.analyst.create_deviation_chart(), (.5, 0), (.5, .5))
            frame.add(self.analyst.create_deprecation_chart(), (.5, .5), (.5, .5))
        except (IndexError, ValueError) as e:
            pass
        frame.show()

    def log_predictions(self, source: tp.Union[str, int], _time: timedelta = None):
        print(f"Source: {source} time: {str(_time)[:12].ljust(8, '.').ljust(12, '0') if _time is not None else ''} | "
              f"RNN: {self._rnn_va} "
              f"Detected Troubles: {[f'{ti}: {act.name}' for ti, act in self.analyst.intersections]}")

    @staticmethod
    def listen_for_quit_button() -> bool:
        if cv2.waitKey(1) == ord(GeneralConfig.DEFAULT_QUIT_BUTTON):
            return True
        return False

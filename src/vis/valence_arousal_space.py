import numpy as np
import cv2

from src.utils import ValenceArousal
from config import PathConfig


class ValenceArousalSpace:
    __VA_SPACE: np.ndarray = cv2.imread(PathConfig.VALENCE_AROUSAL_SPACE_PATH)

    @staticmethod
    def create_chart(cnn_va: ValenceArousal, rnn_va: ValenceArousal) -> np.ndarray:
        def get_value_coord(_metric: float, _scale: int, multiplier: float = 0.8) -> int:
            _value = _metric * multiplier
            _value = (_value + 1) / 2              # From (-1, 1) to (0, 1)
            _value = int(_value * _scale)          # From (0, 1) to (0, scale)
            _value = min([_value, _scale])         # Ensure that value is lower than scale
            _value = max([_value, 0])              # Ensure that value is greater than 0
            return _value

        def add_point(_frame: np.ndarray, y: int, x: int, size: int, color: np.ndarray):
            _frame[y-size:y+size, x-size:x+size] = color

        av_space = ValenceArousalSpace.__VA_SPACE.copy()
        shape = av_space.shape

        cnn_y, cnn_x = get_value_coord(-cnn_va.arousal, shape[0]), get_value_coord(cnn_va.valence, shape[1])
        rnn_y, rnn_x = get_value_coord(-rnn_va.arousal, shape[0]), get_value_coord(rnn_va.valence, shape[1])

        add_point(av_space, cnn_y, cnn_x, 4, np.array([255, 0, 0]))
        add_point(av_space, rnn_y, rnn_x, 4, np.array([0, 255, 0]))
        return av_space

import typing as tp

import numpy as np
import cv2


_INT_FLOAT = tp.Union[int, float]
_INT_FLOAT_TUPLE = tp.Tuple[_INT_FLOAT, _INT_FLOAT]


class Frame:
    def __init__(self, shape: tp.Tuple[int, int], background: tp.Tuple[int, int, int]):
        self.__frame = np.full(shape + (3,), fill_value=np.array(background, dtype=np.uint8))

    def show(self):
        cv2.imshow('Emotion recognition', self.__frame)

    def add(self, frame: np.ndarray, pos: _INT_FLOAT_TUPLE, shape: _INT_FLOAT_TUPLE = None):
        _pos, _shape = self.get_absolute_value(pos), self.get_absolute_value(shape)
        _frame = cv2.resize(frame, _shape[::-1]) if _shape else frame
        _shape = _frame.shape
        assert _pos[0] + _shape[0] <= self.__frame.shape[0] and _pos[1] + _shape[1] <= self.__frame.shape[1]
        self.__frame[_pos[0]: _pos[0] + _shape[0], _pos[1]: _pos[1] + _shape[1]] = _frame

    def get_absolute_value(self, relative_value: _INT_FLOAT_TUPLE):
        if relative_value is None or all([isinstance(v, int) for v in relative_value]):
            return relative_value
        return tuple(np.array(relative_value * np.array(self.__frame.shape[0:2]), dtype=int))

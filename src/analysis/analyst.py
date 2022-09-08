import typing as tp

import numpy as np

from .buffer import Buffer
from src.valence_arousal import ValenceArousal


class Analyst:
    def __init__(self):
        self.__va_buffer = Buffer()
        self.__va_moving_average_9 = Buffer()
        self.__va_moving_average_26 = Buffer()

    def va_mean(self, window_size: int):
        return np.mean(self.__va_buffer.last(window_size), axis=0)

    def add_inference_result(self, valence_arousal: ValenceArousal):
        self.__va_buffer.append(valence_arousal)
        self.__va_moving_average_9.append(self.va_mean(9))
        self.__va_moving_average_26.append(self.va_mean(26))

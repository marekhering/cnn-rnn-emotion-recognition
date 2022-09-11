import typing as tp

import numpy as np
from matplotlib import pyplot as plt

from src.utils import Buffer, ValenceArousal


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

    def create_valence_average_chart(self):
        return self.__create_analysis_chart([
            (self.__va_moving_average_26.np()[:, 0], "26 Period"),
            (self.__va_moving_average_9.np()[:, 0], "9 Period")
        ], "RNN valence average")

    def create_arousal_average_chart(self):
        return self.__create_analysis_chart([
            (self.__va_moving_average_26.np()[:, 1], "26 Period"),
            (self.__va_moving_average_9.np()[:, 1], "9 Period")
        ], "RNN arousal average")

    def create_va_chart(self):
        return self.__create_analysis_chart([
            (self.__va_buffer.np()[:, 0], "Valence"),
            (self.__va_buffer.np()[:, 1], "Arousal")
        ], "RNN inference result")

    @staticmethod
    def __create_analysis_chart(data: tp.List[tp.Tuple[tp.List[float], str]], title: str):
        fig, ax = plt.subplots()
        for values, label in data:
            ax.plot(values, label=label)

        plt.title(title)
        plt.legend()
        fig.canvas.draw()
        return np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

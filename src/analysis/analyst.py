import typing as tp

import numpy as np
from matplotlib import pyplot as plt

from src.utils import Buffer, ValenceArousal


class Analyst:
    def __init__(self):
        self.__number_of_reads = 0

        self.__va_buffer = Buffer()
        self.__va_moving_average_9 = Buffer()
        self.__va_moving_average_26 = Buffer()
        self.__va_mean_buffer = Buffer()
        self.__va_variance_buffer = Buffer()
        self.__va_std_buffer = Buffer()

    def va_moving_average(self, window_size: int):
        return np.mean(self.__va_buffer.last(window_size), axis=0)

    def add_inference_result(self, valence_arousal: ValenceArousal):
        self.__va_buffer.append(valence_arousal)
        self.__va_moving_average_9.append(self.va_moving_average(9))
        self.__va_moving_average_26.append(self.va_moving_average(26))
        self.__va_mean_buffer.append(self.va_mean(valence_arousal))
        _var, _std = self.va_variance(valence_arousal), self.va_std(valence_arousal)
        self.__va_variance_buffer.append(_var)
        self.__va_std_buffer.append(_std)
        self.__number_of_reads += 1

    def va_mean(self, va_2: ValenceArousal):
        if self.__va_mean_buffer.is_empty():
            return va_2
        _va_mean = self.__va_mean_buffer[-1] * self.__number_of_reads
        _va_mean = _va_mean + va_2
        _va_mean = _va_mean / (self.__number_of_reads + 1)
        return _va_mean

    def va_variance(self, va_2: ValenceArousal):
        if len(self.__va_mean_buffer) < 2:
            return np.var(self.__va_buffer.np(), axis=0)
        # https://math.stackexchange.com/questions/775391
        _va_variance = self.__va_variance_buffer[-1] * self.__number_of_reads
        _va_variance = _va_variance + (va_2 - self.__va_mean_buffer[-1]) * (va_2 - self.__va_mean_buffer[-2])
        _va_variance = _va_variance / (self.__number_of_reads + 1)
        return _va_variance

    def va_std(self, va_2: ValenceArousal):
        return np.sqrt(self.va_variance(va_2))

    def create_valence_average_chart(self):
        return self.__create_analysis_chart([
            # (self.__va_buffer.np()[:, 0], "Valence"),
            (self.__va_moving_average_26.np()[:, 0], "26 Period"),
            (self.__va_moving_average_9.np()[:, 0], "9 Period"),
            (self.__va_mean_buffer.np()[:, 0],  "Overall"),
            (self.__va_mean_buffer.np()[:, 0] - self.__va_std_buffer.np()[:, 0], "Overall - STD1")
        ], "RNN valence average")

    def create_arousal_average_chart(self):
        return self.__create_analysis_chart([
            (self.__va_moving_average_26.np()[:, 1], "26 Period"),
            (self.__va_moving_average_9.np()[:, 1], "9 Period"),
            (self.__va_mean_buffer.np()[:, 1],  "Overall")
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

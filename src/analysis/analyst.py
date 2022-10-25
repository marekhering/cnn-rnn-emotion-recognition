import typing as tp
from datetime import timedelta

import numpy as np
from matplotlib import pyplot as plt

from config import AnalysisConing
from src.utils import Buffer, ValenceArousal
from .trouble import Trouble


class Analyst:
    def __init__(self):
        self.__va_buffer = Buffer()

        # Long term troubles
        self.__va_moving_average_9 = Buffer()
        self.__va_moving_average_45 = Buffer()
        self.__va_mean_buffer = Buffer()
        self.__va_variance_buffer = Buffer()
        self.__va_std_buffer = Buffer()
        self.__va_std_delta_buffer = Buffer()
        self.__last_long_term_trouble_read = 0

        # Short term troubles
        self.__derivative_buffer = Buffer()
        self.__derivative_moving_average_10 = Buffer()
        self.__derivative_threshold = Buffer()

        self.__number_of_reads = 0
        self.troubles: tp.List[tp.Tuple[Trouble, timedelta]] = []

    def add_inference_result(self, valence_arousal: ValenceArousal, _time: timedelta):
        self.__va_buffer.append(valence_arousal)

        # Long term troubles
        self.__va_moving_average_9.append(self.va_moving_average(9))
        self.__va_moving_average_45.append(self.va_moving_average(45))
        self.__va_mean_buffer.append(self.va_mean())
        self.__va_variance_buffer.append(self.va_variance())
        self.__va_std_buffer.append(self.va_std())
        self.__va_std_delta_buffer.append(self.__va_moving_average_9[-1] - self.sensitive_va_std()[-1])

        # Short term troubles
        self.__derivative_buffer.append(self.derivative())
        self.__derivative_moving_average_10.append(self.derivative_moving_average(10))
        self.__derivative_threshold.append(AnalysisConing.DERIVATIVE_SENSITIVITY)

        self.__number_of_reads += 1
        self.__find_troubles(_time)

    def __find_troubles(self, _time: timedelta):
        if _time < timedelta(seconds=AnalysisConing.DELAY):
            return

        # Looking for long term troubles
        va_std_values = self.__va_std_delta_buffer.np()[-AnalysisConing.LONG_TERM_TROUBLE_LENGTH:, 0]
        if self.__number_of_reads - self.__last_long_term_trouble_read > AnalysisConing.LONG_TERM_TROUBLE_LENGTH:
            if (va_std_values < 0).all():
                self.troubles.append((Trouble.long_term, _time))
                self.__last_long_term_trouble_read = self.__number_of_reads

        # Looking for short term troubles
        if self.is_intersection(self.__derivative_moving_average_10.np()[:, 0], self.__derivative_threshold.np()):
            self.troubles.append((Trouble.short_term, _time))

    def va_moving_average(self, window_size: int):
        return np.mean(self.__va_buffer.last(window_size), axis=0)

    def va_mean(self):
        if self.__va_mean_buffer.is_empty():
            return self.__va_buffer[-1]
        _va_mean = self.__va_mean_buffer[-1] * self.__number_of_reads
        _va_mean = _va_mean + self.__va_buffer[-1]
        _va_mean = _va_mean / (self.__number_of_reads + 1)
        return _va_mean

    def va_variance(self):
        if len(self.__va_mean_buffer) < 2:
            return np.var(self.__va_buffer.np(), axis=0)
        # https://math.stackexchange.com/questions/775391
        _va = self.__va_buffer[-1]
        _va_variance = self.__va_variance_buffer[-1] * self.__number_of_reads
        _va_variance = _va_variance + (_va - self.__va_mean_buffer[-1]) * (_va - self.__va_mean_buffer[-2])
        _va_variance = _va_variance / (self.__number_of_reads + 1)
        return _va_variance

    def va_std(self):
        return np.sqrt(self.va_variance())

    def sensitive_va_std(self):
        return self.__va_mean_buffer.np() - (self.__va_std_buffer.np() * AnalysisConing.STD_SENSITIVITY)

    def derivative(self):
        if len(self.__va_buffer) < 2:
            return ValenceArousal(0, 0)
        return self.__va_buffer[-1] - self.__va_buffer[-2]

    def derivative_moving_average(self, window_size: int):
        return np.mean(self.__derivative_buffer.last(window_size), axis=0)

    @staticmethod
    def is_intersection(values_1: tp.List, values_2: tp.List) -> bool:
        """
        A function that checks whether the first buffer intersects the second one from above
        :param values_1: first buffered values
        :param values_2: second buffered values
        :return: True is there is an intersection, False otherwise
        """
        try:
            if values_1[-1] < values_2[-1]:
                if values_1[-2] > values_2[-2]:
                    return True
        except IndexError:
            pass
        return False

    # Charts
    def create_valence_average_chart(self):
        return self.__create_analysis_chart([
            (self.__va_buffer.np()[:, 0], "Valence"),
            (self.__va_moving_average_9.np()[:, 0], "9 period average"),
            (self.__va_mean_buffer.np()[:, 0],  "Average"),
            (self.sensitive_va_std()[:, 0], "Detection threshold")
        ], "Long term trouble detection")

    def create_arousal_average_chart(self):
        return self.__create_analysis_chart([
            (self.__va_moving_average_9.np()[:, 1], "9 period average"),
            (self.__va_mean_buffer.np()[:, 1],  "Average")
        ], "RNN arousal average")

    def create_va_chart(self):
        return self.__create_analysis_chart([
            (self.__va_buffer.np()[:, 1], "Arousal"),
            (self.__va_buffer.np()[:, 0], "Valence")
        ], "RNN raw inference result")

    def create_derivative_chart(self):
        return self.__create_analysis_chart([
            (self.__derivative_buffer.np()[:, 0], "Derivative value"),
            (self.__derivative_moving_average_10.np()[:, 0], "10 period average"),
            (self.__derivative_threshold.np(), "Detection threshold")
        ], "Short term trouble detection")

    @staticmethod
    def __create_analysis_chart(data: tp.List[tp.Tuple[tp.List[float], str]], title: str):
        plt.close()
        fig, ax = plt.subplots()
        for values, label in data:
            ax.plot(values, label=label)

        plt.title(title)
        plt.legend()
        fig.canvas.draw()
        return np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

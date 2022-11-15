import typing as tp
from datetime import timedelta

import numpy as np
from matplotlib import pyplot as plt

from config import AnalysisConing
from src.utils import Buffer, ValenceArousal
from .activator import Activator


class Analyst:
    def __init__(self):
        self.__va_buffer = Buffer()
        self.__number_of_reads = 0
        self.troubles: tp.List[tp.Tuple[Activator, str]] = []

        # Deviation activators
        self.__va_moving_average = Buffer()
        self.__last_deviation_read = 0

        # # Local deviation activator
        self.__va_mean_local_buffer = Buffer()
        self.__va_std_local_buffer = Buffer()

        # # Global deviation activator
        self.__va_mean_global_buffer = Buffer()
        self.__va_variance_global_buffer = Buffer()
        self.__va_std_global_buffer = Buffer()

        # Rapid deprecation activator
        self.__derivative_buffer = Buffer()
        self.__derivative_moving_average = Buffer()

        # # Local deprecation activator
        self.__local_derivative_mean = Buffer()
        self.__local_derivative_std = Buffer()

        # # Global deprecation activator
        self.__global_derivative_mean = Buffer()
        self.__global_derivative_variance = Buffer()
        self.__global_derivative_std = Buffer()

    def add_inference_result(self, valence_arousal: ValenceArousal, _time: timedelta):
        self.__va_buffer.append(valence_arousal)
        # Deviation activators
        self.__va_moving_average.append(self.va_moving_average(10))

        # # Local deviation activator
        self.__va_mean_local_buffer.append(self.__va_buffer.np().mean(axis=0))
        self.__va_std_local_buffer.append(self.__va_buffer.np().std(axis=0))

        # # Global deviation activator
        self.__va_mean_global_buffer.append(self.va_mean(self.__va_mean_global_buffer, valence_arousal))
        self.__va_variance_global_buffer.append(self.va_variance(
            self.__va_variance_global_buffer, self.__va_mean_global_buffer, valence_arousal))
        self.__va_std_global_buffer.append(self.va_std(self.__va_variance_global_buffer))

        # Rapid deprecation activator
        self.__derivative_buffer.append(self.derivative())
        self.__derivative_moving_average.append(self.derivative_moving_average(10))

        # # Local deprecation activator
        self.__local_derivative_mean.append(self.__derivative_buffer.np().mean(axis=0))
        self.__local_derivative_std.append(self.__derivative_buffer.np().std(axis=0))

        # # Global deprecation activator
        self.__global_derivative_mean.append(self.va_mean(self.__global_derivative_mean, self.derivative()))
        self.__global_derivative_variance.append(self.va_variance(self.__global_derivative_variance, self.__global_derivative_mean, self.derivative()))
        self.__global_derivative_std.append(self.va_std(self.__global_derivative_variance))

        self.__number_of_reads += 1
        self.__find_troubles(_time)

    def __find_troubles(self, _time: timedelta):
        if _time < timedelta(seconds=AnalysisConing.DELAY):
            return

        # Looking for deviation troubles
        def find_deviation_trouble(threshold: np.ndarray, activator: Activator):
            n = AnalysisConing.LONG_TERM_TROUBLE_LENGTH
            delta_values = (self.__va_moving_average.np() - threshold)[-n:, 0]
            if self.__number_of_reads - self.__last_deviation_read > AnalysisConing.LONG_TERM_TROUBLE_LENGTH:
                if (delta_values < 0).all():
                    self.troubles.append((activator, str(_time)))
                    self.__last_deviation_read = self.__number_of_reads

        find_deviation_trouble(self.deviation_threshold(self.__va_std_global_buffer), Activator.global_deviation)
        find_deviation_trouble(self.deviation_threshold(self.__va_std_local_buffer), Activator.local_deviation)
        find_deviation_trouble(self.sigmoid_threshold(self.__va_std_global_buffer), Activator.sigmoid_deviation)
        find_deviation_trouble(self.sigmoid_threshold(self.__va_std_local_buffer), Activator.sigmoid_deviation)

        # Looking for deprecation troubles
        if self.is_intersection(self.__derivative_moving_average.np()[:, 0],
                                self.derivative_threshold(self.__local_derivative_std)[:, 0]):
            self.troubles.append((Activator.local_rapid_deprecation, str(_time)))

        if self.is_intersection(self.__derivative_moving_average.np()[:, 0],
                                self.derivative_threshold(self.__global_derivative_std)[:, 0]):
            self.troubles.append((Activator.global_rapid_deprecation, str(_time)))


    def va_moving_average(self, window_size: int):
        return np.mean(self.__va_buffer.last(window_size), axis=0)

    def va_mean(self, mean: Buffer, va: ValenceArousal):
        if mean.is_empty():
            return va
        _va_mean: ValenceArousal = mean[-1] * self.__number_of_reads
        _va_mean = _va_mean + va
        _va_mean = _va_mean / (self.__number_of_reads + 1)
        return _va_mean

    def va_variance(self, variance: Buffer, mean: Buffer, va: ValenceArousal):
        if len(mean) < 2:
            return np.var(self.__va_buffer.np(), axis=0)
        # https://math.stackexchange.com/questions/775391
        _va_variance = variance[-1] * self.__number_of_reads
        _va_variance = _va_variance + (va - mean[-1]) * (va - mean[-2])
        _va_variance = _va_variance / (self.__number_of_reads + 1)
        return _va_variance

    def va_std(self, variance: Buffer):
        return np.sqrt(variance[-1])

    @staticmethod
    def sigmoid(x: np.ndarray):
        return (2 / (1 + np.exp((-x) * 2.2))) - 1

    def deviation_threshold(self, std_buffer: Buffer):
        return self.__va_mean_local_buffer.np() - (std_buffer.np() * AnalysisConing.STD_SENSITIVITY)

    def sigmoid_threshold(self, std_buffer: Buffer):
        return self.sigmoid(self.__va_mean_local_buffer.np() - (std_buffer.np() * AnalysisConing.STD_SENSITIVITY))

    def derivative(self):
        if len(self.__va_buffer) < 2:
            return ValenceArousal(0, 0)
        return self.__va_buffer[-1] - self.__va_buffer[-2]

    def derivative_moving_average(self, window_size: int):
        return np.mean(self.__derivative_buffer.last(window_size), axis=0)

    def derivative_threshold(self, std_buffer: Buffer):
        return self.__local_derivative_mean.np() - (std_buffer.np() * AnalysisConing.STD_SENSITIVITY)

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
    def create_va_chart(self):
        return self.__create_analysis_chart([
            (self.__va_buffer.np()[:, 1], "Arousal"),
            (self.__va_buffer.np()[:, 0], "Valence")
        ], "RNN raw inference result")

    def create_deviation_chart(self):
        return self.__create_analysis_chart([
            (self.__va_buffer.np()[:, 0], "Valence"),
            (self.__va_moving_average.np()[:, 0], "10 period average"),
            (self.deviation_threshold(self.__va_std_local_buffer)[:, 0], "Local threshold"),
            (self.deviation_threshold(self.__va_std_global_buffer)[:, 0], "Global threshold"),
            (self.sigmoid_threshold(self.__va_std_local_buffer)[:, 0], "Local sigmoid threshold"),
            (self.sigmoid_threshold(self.__va_std_global_buffer)[:, 0], "Global sigmoid threshold"),
        ], "Deviation detection")

    def create_deprecation_chart(self):
        return self.__create_analysis_chart([
            (self.__derivative_buffer.np()[:, 0], "Derivative value"),
            (self.__derivative_moving_average.np()[:, 0], "10 period average"),
            (self.derivative_threshold(self.__local_derivative_std)[:, 0], "Local threshold"),
            (self.derivative_threshold(self.__global_derivative_std)[:, 0], "Global threshold"),
        ], "Rapid deprecation detection")

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

import typing as tp
import itertools
from collections import defaultdict
from datetime import timedelta

from src.utils import ValenceArousal
from config import AnalysisConing
from .activator import Activator


class NaiveAnalyst:

    def __init__(self):
        self.__v = []
        self.__a = []
        self._intersections: tp.Dict[Activator, tp.List[timedelta]] = defaultdict(list)

    @property
    def events(self) -> tp.List[tp.Tuple[timedelta, Activator]]:
        return list(itertools.chain(*[[(t, act) for t in times] for act, times in self._intersections.items()]))

    def add_inference_result(self, valence_arousal: ValenceArousal, _time: timedelta):
        self.__v.append(valence_arousal.valence)
        self.__a.append(valence_arousal.arousal)
        self.find_intersections(_time)

    def find_intersections(self, _time: timedelta):
        if _time.total_seconds() < AnalysisConing.DELAY:
            return

        def add_intersect(data_line, control_line, activator, check_oddity: bool = False):
            if not check_oddity or len(self._intersections[activator]) % 2 == 1:
                if self.is_intersection(data_line, control_line):
                    self._intersections[activator].append(_time)

        for activator in Activator.get_naive_bounds():
            control_line = [float(f"-0.{activator.name[-1]}")] * 2
            add_intersect(self.__v, control_line, activator)
            add_intersect(control_line, self.__v, activator, check_oddity=True)

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

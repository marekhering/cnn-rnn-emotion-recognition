from enum import Enum


class Activator(str, Enum):
    global_deviation = 0
    local_deviation = 1
    global_sigmoid_deviation = 2
    local_sigmoid_deviation = 3
    local_rapid_deprecation = 4
    global_rapid_deprecation = 5
    naive_bound_00 = 6
    naive_bound_01 = 7
    naive_bound_03 = 8
    naive_bound_05 = 9

    @staticmethod
    def get_naive_bounds():
        return [Activator.naive_bound_05, Activator.naive_bound_03, Activator.naive_bound_01, Activator.naive_bound_00]

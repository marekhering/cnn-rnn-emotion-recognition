from enum import Enum


class Activator(str, Enum):
    global_deviation = 0
    local_deviation = 1
    sigmoid_deviation = 2
    local_rapid_deprecation = 3
    global_rapid_deprecation = 4

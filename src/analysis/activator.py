from enum import Enum


class Activator(str, Enum):
    global_deviation = 0
    local_deviation = 1
    sigmoid_deviation = 2
    rapid_deprecation = 3

from enum import Enum


class Activator(str, Enum):
    global_deviation = 0
    local_deviation = 1
    global_sigmoid_deviation = 2
    local_sigmoid_deviation = 3
    local_rapid_deprecation = 4
    global_rapid_deprecation = 5

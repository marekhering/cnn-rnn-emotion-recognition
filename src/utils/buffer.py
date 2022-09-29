import typing as tp

import numpy as np

from collections import deque

from config import GeneralConfig


_T = tp.TypeVar('_T')


class Buffer(tp.Generic[_T]):
    def __init__(self, max_len: int = GeneralConfig.MAX_BUFFER_SIZE):
        self.__deque = deque(maxlen=max_len)

    def __getitem__(self, item):
        return self.__deque[item]

    def last(self, n: int) -> tp.List[_T]:
        return list(self.__deque)[-n:]

    def append(self, item: _T):
        self.__deque.append(item)

    def np(self):
        return np.array(self.__deque)

    def is_empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self.__deque)

    def __repr__(self):
        return self.__deque.__repr__()

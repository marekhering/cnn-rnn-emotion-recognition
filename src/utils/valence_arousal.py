from __future__ import annotations

import typing as tp


class ValenceArousal(tp.List):
    def __init__(self, valence: float = 0, arousal: float = 0):
        super().__init__([valence, arousal])
        self.__valence = valence
        self.__arousal = arousal

    @property
    def valence(self):
        return self.__valence

    @property
    def arousal(self):
        return self.__arousal

    def __repr__(self):
        return "Valence %.6s | Arousal %.6s" % (self.valence, self.__arousal)
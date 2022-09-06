import numpy as np


class ValenceArousal:
    def __init__(self, valence: float = 0, arousal: float = 0):
        self.__valence = valence
        self.__arousal = arousal
        self.__values = np.array([valence, arousal])

    @property
    def valence(self):
        return self.__valence

    @property
    def arousal(self):
        return self.__arousal

    def __repr__(self):
        return "Valence %.6s | Arousal %.6s" % (self.valence, self.__arousal)
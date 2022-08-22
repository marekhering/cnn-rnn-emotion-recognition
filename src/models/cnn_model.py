from keras import Model
import numpy as np

from .base_model import BaseModel


class CNNModel(BaseModel):
    def __init__(self, model_path):
        super(CNNModel, self).__init__(model_path)
        self._extract_model = Model(inputs=self._model.inputs, outputs=self._model.get_layer('dense').output)

    def extract_features(self, x: np.ndarray):
        self._predict(self._extract_model, x)

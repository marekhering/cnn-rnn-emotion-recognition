from keras import Model, Input
import numpy as np

from .base_model import BaseModel


class CNNModel(BaseModel):
    def __init__(self, model_path):
        super(CNNModel, self).__init__(model_path)
        self._extract_model = Model(inputs=self._model.inputs, outputs=self._model.get_layer('dense').output)

        classification_input = Input((None, 300))
        classification_output = self._model.get_layer('dense_1')(classification_input)
        self._classification_model = Model(inputs=classification_input, outputs=classification_output)

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        return self._predict(self._extract_model, x)

    def classify_features(self, x: np.ndarray) -> np.ndarray:
        return self._predict(self._classification_model, x)

    @property
    def image_shape(self):
        return self._model.input_shape[1:3]

    @property
    def is_grayscale(self):
        return self._model.input_shape[3] == 1

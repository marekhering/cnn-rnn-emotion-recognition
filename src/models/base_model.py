from keras.callbacks import CallbackList
from keras.models import load_model, Model
import numpy as np

from src.models import metrics


class BaseModel:
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model: Model = load_model(self._model_path, custom_objects=self._custom_objects)
        print()

    @property
    def _custom_objects(self):
        return {
            'ccc_loss': metrics.ccc_loss,
            'rmse': metrics.rmse,
            'rmse_v': metrics.rmse_v,
            'rmse_a': metrics.rmse_a,
            'cc_v': metrics.cc_v,
            'cc_a': metrics.cc_a,
            'ccc_v': metrics.ccc_v,
            'ccc_a': metrics.ccc_a
        }

    def predict(self, x):
        return self._predict(self._model, x)

    @classmethod
    def _predict(cls, model: Model, x: np.ndarray):
        return model.predict(x, callbacks=CallbackList())

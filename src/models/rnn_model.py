from .base_model import BaseModel


class RNNModel(BaseModel):

    @property
    def window_size(self):
        return self._model.input_shape[1]

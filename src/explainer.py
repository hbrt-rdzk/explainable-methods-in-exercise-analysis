import numpy as np
from torch import nn


class Explainer:
    def __init__(self, model: nn.Module, data: np.ndarray, label: float) -> None:
        self.model = model
        self.data = data
        self.label = label

    def generate_cf_explainations(self, window_size: int = 4, stride: int = 1):
        ...
        # signal_length = self.data.shape[1]
        # for idx in range(0, len(signal) - sliding_window_size + 1, stride):

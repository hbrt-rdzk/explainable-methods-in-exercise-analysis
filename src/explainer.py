import numpy as np
import torch
import tsaug as tsa
from scipy.interpolate import CubicSpline
from torch import nn


class Explainer:
    def __init__(
        self, model: nn.Module, changable_parameters: list[str], correct_label: int
    ) -> None:
        self.model = model
        self.changable_parameters = changable_parameters
        self.correct_label = correct_label

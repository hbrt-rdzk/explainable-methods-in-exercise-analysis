import os

import numpy as np
import pandas as pd
import torch
from scipy.fft import dct, idct
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ExerciseDataset

DCT_COEFFICIENTS_SIZE = 25


def encode_samples_to_latent(
    model: nn.Module, data: list[torch.Tensor]
) -> torch.Tensor:
    model.eval()
    data = torch.stack([rep for rep in data])
    return model.encoder(data)[0]


def decode_samples_from_latent(model: nn.Module, data: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model.decoder(data)


def get_data(
    dir: str, representation: str, exercise: str, batch_size: int = 8
) -> DataLoader:
    train_df = pd.read_csv(
        os.path.join(dir, "train", representation, exercise + ".csv")
    )
    test_df = pd.read_csv(os.path.join(dir, "test", representation, exercise + ".csv"))
    train_dataset = ExerciseDataset(train_df, representation=representation)
    test_dataset = ExerciseDataset(test_df, representation=representation)

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_dl, val_dl


def encode_dct(x: np.ndarray) -> np.ndarray:
    return dct(x, norm="ortho")[:DCT_COEFFICIENTS_SIZE]


def decode_dct(x: np.ndarray, length) -> np.ndarray:
    decoded_signal = []
    for feature in x.transpose(1, 0):
        x_dct = np.zeros(length, dtype=float)
        x_dct[:DCT_COEFFICIENTS_SIZE] = feature
        x_idct = idct(x_dct, norm="ortho")
        decoded_signal.append(x_idct)
    return np.array(decoded_signal).transpose(1, 0)

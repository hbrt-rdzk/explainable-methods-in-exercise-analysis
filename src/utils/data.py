import os

import numpy as np
import pandas as pd
import torch
from scipy.fft import idct
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ExerciseDataset


def generate_latent_samples(model: nn.Module, data: DataLoader) -> np.ndarray:
    model.eval()
    data = torch.stack([rep for rep in data.dataset.data])
    return model.encoder(data)[0].detach().numpy()


def get_data(
    dir: str, representation: str, exercise: str, batch_size: int
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


def decode_dct(x: torch.Tensor, length) -> torch.Tensor:
    x = x.squeeze().detach().numpy()
    decoded_signal = []
    for feature in x.transpose(1, 0):
        x_dct = np.zeros(length, dtype=float)
        x_dct[:25] = feature
        x_idct = idct(x_dct, norm="ortho")
        decoded_signal.append(x_idct)
    return np.array(decoded_signal).transpose(1, 0)

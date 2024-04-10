import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ExerciseDataset


def generate_latent_samples(model: nn.Module, data: DataLoader) -> np.ndarray:
    model.eval()
    data = torch.stack([rep for rep in data.dataset.data])
    return model.encoder(data).detach().numpy()


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
